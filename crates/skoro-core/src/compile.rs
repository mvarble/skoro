//! The compiler: a query, or a composite alone, into a program of steps over slots.
//!
//! 1. **Reach.** Walk the compilation graph backwards from the targets, stopping at quantities,
//!    constants and nodes fixed in the model.
//! 2. **Lower.** Each mapped node met is lowered for the target, and a composite is expanded into
//!    its parts, recursively, under the chain, so that a cycle is refused as `CompositeCycle`.
//! 3. **Classify.** A slot is fixed when it is a constant, a node fixed in the model, an observed
//!    quantity, or a step all of whose inputs are fixed.
//! 4. **Fold.** Fixed steps run once, forward, on the target, and become constants.
//! 5. **Prune.** Slots that no longer reach a target are dropped, with the steps that write them.
//! 6. **Finish.** Each slot records the last step that reads it.

use std::collections::{HashMap, HashSet};
use std::sync::{Arc, PoisonError};

use skoro_tensor::{HostValue, Target, Value};

use crate::model::{Model, NodeKind};
use crate::operation::Link;
use crate::primitive::{Add, Select};
use crate::program::{Free, Root, Slot, Step, expand};
use crate::query::Targets;
use crate::scope::fits;
use crate::{
    Composite, Error, Lowered, Lowering, NodeId, Observations, Operation, Program, Query, Snapshot,
    Space,
};

/// The key a program is cached under in its snapshot.
#[derive(Clone, PartialEq, Debug)]
pub(crate) struct ProgramKey {
    targets: Targets,
    observations: u64,
    target: Target,
}

/// How many programs a snapshot keeps, the most recently used.
const CACHED: usize = 32;

impl Snapshot {
    /// The program of `query` on `target`: from the snapshot's cache, under the query's targets,
    /// its observation set's identity and the target, or compiled and cached.
    pub fn compile(&self, query: &Query, target: &Target) -> Result<Arc<Program>, Error> {
        let key = ProgramKey {
            targets: query.targets.clone(),
            observations: query.observations.identity(),
            target: target.clone(),
        };
        {
            let mut programs = self
                .0
                .programs
                .lock()
                .unwrap_or_else(PoisonError::into_inner);
            if let Some(at) = programs.iter().position(|(k, _)| *k == key) {
                let entry = programs.remove(at);
                let program = entry.1.clone();
                programs.push(entry);
                return Ok(program);
            }
        }
        let program = Arc::new(self.compile_uncached(query, target)?);
        let mut programs = self
            .0
            .programs
            .lock()
            .unwrap_or_else(PoisonError::into_inner);
        programs.push((key, program.clone()));
        if programs.len() > CACHED {
            programs.remove(0);
        }
        Ok(program)
    }

    /// The number of programs the snapshot keeps.
    #[doc(hidden)]
    pub fn cached_programs(&self) -> usize {
        self.0
            .programs
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .len()
    }

    fn compile_uncached(&self, query: &Query, target: &Target) -> Result<Program, Error> {
        let observations = &query.observations;
        for id in observations.ids() {
            if self
                .0
                .nodes
                .get(&id)
                .is_some_and(|node| !node.is_quantity())
            {
                return Err(Error::NotAQuantity { id });
            }
        }
        let terms: Option<Vec<NodeId>> = match &query.targets {
            Targets::Nodes(_) => None,
            Targets::Density => Some(self.0.terms.keys().copied().collect()),
            Targets::DensityOf(terms) => {
                for &id in terms {
                    if !self.0.terms.contains_key(&id) {
                        return Err(Error::NotATerm { id });
                    }
                }
                Some(terms.iter().copied().collect())
            }
        };
        let mut lowering = Lowering::new(target);
        let mut builder = Builder::new(&mut lowering);
        let mut env = HashMap::new();
        let targets = match (&query.targets, terms) {
            (Targets::Nodes(nodes), _) => {
                let outputs = builder.emit(&self.0, nodes, &mut env, None, Some(observations))?;
                nodes.iter().copied().zip(outputs).collect()
            }
            (_, Some(mut terms)) => {
                // The fixed terms first, those whose every quantity is observed, so that their
                // sum folds.
                let observed = |id: &NodeId| {
                    self.0.terms.get(id).is_some_and(|term| {
                        term.involves.iter().all(|q| observations.get(*q).is_some())
                    })
                };
                terms.sort_by_key(|id| (!observed(id), *id));
                let slots = builder.emit(&self.0, &terms, &mut env, None, Some(observations))?;
                vec![(NodeId::DENSITY, builder.sum(&slots)?)]
            }
            (_, None) => unreachable!("a density query names its terms"),
        };
        builder.finish(targets)
    }
}

impl Composite {
    /// A program of the composite alone, whose roots are its inputs.
    pub fn compile(&self, target: &Target) -> Result<Program, Error> {
        let mut lowering = Lowering::new(target);
        lowering.enter(self)?;
        compile_body(self, &mut lowering)
    }
}

impl Lowering<'_> {
    /// `composite` compiled for this target, from the cache or compiled under the chain, which
    /// it is on for the length of the compilation (`CompositeCycle`).
    pub fn compile(&mut self, composite: &Arc<Composite>) -> Result<Arc<Program>, Error> {
        self.enter(composite)?;
        let link = Link::of(composite);
        if let Some(program) = self.cache.get(&link) {
            let program = program.clone();
            self.leave();
            return Ok(program);
        }
        let program = compile_body(composite, self);
        self.leave();
        let program = Arc::new(program?);
        self.cache.insert(link, program.clone());
        Ok(program)
    }
}

/// The program of a composite's body, its inputs free and its output the target; the composite
/// is on the chain.
fn compile_body(composite: &Composite, lowering: &mut Lowering<'_>) -> Result<Program, Error> {
    let body = composite.body();
    let mut builder = Builder::new(lowering);
    let mut env = HashMap::new();
    for (&node, space) in body.inputs.iter().zip(composite.inputs()) {
        let slot = builder.free(node, space.clone());
        env.insert(node, slot);
    }
    let outputs = builder.emit(&body.model, &[body.output], &mut env, None, None)?;
    builder.finish(vec![(body.output, outputs[0])])
}

/// The slots and steps of a program as it is emitted.
struct Builder<'l, 'a> {
    lowering: &'l mut Lowering<'a>,
    spaces: Vec<Space>,
    roots: Vec<(Slot, Root)>,
    steps: Vec<Step>,
    free: Vec<Free>,
    free_slots: Vec<Slot>,
}

impl<'l, 'a> Builder<'l, 'a> {
    fn new(lowering: &'l mut Lowering<'a>) -> Builder<'l, 'a> {
        Builder {
            lowering,
            spaces: Vec::new(),
            roots: Vec::new(),
            steps: Vec::new(),
            free: Vec::new(),
            free_slots: Vec::new(),
        }
    }

    fn slot(&mut self, space: Space) -> Slot {
        self.spaces.push(space);
        self.spaces.len() - 1
    }

    fn free(&mut self, node: NodeId, space: Space) -> Slot {
        let slot = self.slot(space.clone());
        self.roots.push((
            slot,
            Root::Free {
                index: self.free.len(),
            },
        ));
        self.free.push(Free { node, space });
        self.free_slots.push(slot);
        slot
    }

    fn constant(&mut self, space: Space, value: &HostValue) -> Result<Slot, Error> {
        let target = self.lowering.target().clone();
        let rows = value.rows();
        let loaded = value
            .load(rows.unwrap_or(1), &target)?
            .ok_or(Error::BatchSize {
                expected: rows.unwrap_or(1),
                found: rows.unwrap_or(1),
            })?;
        let slot = self.slot(space);
        self.roots.push((
            slot,
            Root::Constant {
                value: Box::new(loaded),
                rows,
            },
        ));
        Ok(slot)
    }

    /// Emits the slots and steps that compute `outputs` from `model`, with the nodes in `env`
    /// already bound to slots, and gives the outputs' slots. Steps are attributed to `origin`, the
    /// composite call they were expanded from, or to their own nodes.
    fn emit(
        &mut self,
        model: &Model,
        outputs: &[NodeId],
        env: &mut HashMap<NodeId, Slot>,
        origin: Option<NodeId>,
        observations: Option<&Observations>,
    ) -> Result<Vec<Slot>, Error> {
        for id in reach(model, outputs, env)? {
            let Some(node) = model.nodes.get(&id) else {
                unreachable!("reach found the node in the model")
            };
            let space = node.space().clone();
            let slot = match &node.kind {
                NodeKind::Quantity { .. } => match observations.and_then(|o| o.get(id)) {
                    Some(Some(value)) if fits(&space, value) => self.constant(space, value)?,
                    Some(_) => return Err(Error::ValueShape { space }),
                    None => self.free(id, space),
                },
                NodeKind::Input => return Err(Error::UnknownNode { id }),
                NodeKind::Constant { value } => self.constant(space, value)?,
                NodeKind::Mapped {
                    folded: Some(value),
                    ..
                } => self.constant(space, value)?,
                NodeKind::Mapped {
                    operation, inputs, ..
                } => {
                    let slots: Vec<Slot> = inputs
                        .iter()
                        .map(|input| match env.get(input) {
                            Some(&slot) => slot,
                            None => unreachable!("inputs are emitted before their consumers"),
                        })
                        .collect();
                    let spaces: Vec<Space> = slots
                        .iter()
                        .map(|&slot| self.spaces[slot].clone())
                        .collect();
                    match operation.lower(&spaces, self.lowering)? {
                        Lowered::Kernel(kernel) => {
                            let output = self.slot(space);
                            self.steps.push(Step {
                                kernel,
                                inputs: slots,
                                output,
                                operation: operation.name(),
                                composites: self.lowering.names(),
                                node: origin.unwrap_or(id),
                            });
                            output
                        }
                        Lowered::Composite(composite) => {
                            self.lowering.enter(&composite)?;
                            let body = composite.body();
                            let mut inner: HashMap<NodeId, Slot> =
                                body.inputs.iter().copied().zip(slots).collect();
                            let origin = origin.or(Some(id));
                            let out =
                                self.emit(&body.model, &[body.output], &mut inner, origin, None);
                            self.lowering.leave();
                            out?[0]
                        }
                    }
                }
            };
            env.insert(id, slot);
        }
        outputs
            .iter()
            .map(|id| env.get(id).copied().ok_or(Error::UnknownNode { id: *id }))
            .collect()
    }

    /// The sum of `slots`, by `skoro.add` steps in their order; a zero for none.
    fn sum(&mut self, slots: &[Slot]) -> Result<Slot, Error> {
        let Some((&first, rest)) = slots.split_first() else {
            let zero = HostValue::new(None, 1, skoro_tensor::HostData::Continuous(vec![0.0]));
            let Some(zero) = zero else {
                unreachable!("one value of storage one")
            };
            return self.constant(Space::Real, &zero);
        };
        let mut sum = first;
        for &slot in rest {
            let Lowered::Kernel(kernel) = Add.lower(&[Space::Real, Space::Real], self.lowering)?
            else {
                unreachable!("addition lowers to a kernel")
            };
            let output = self.slot(Space::Real);
            self.steps.push(Step {
                kernel,
                inputs: vec![sum, slot],
                output,
                operation: Add.name(),
                composites: Vec::new(),
                node: NodeId::DENSITY,
            });
            sum = output;
        }
        Ok(sum)
    }

    /// Classifies, folds, prunes and finishes the program computing `targets`.
    fn finish(mut self, targets: Vec<(NodeId, Slot)>) -> Result<Program, Error> {
        let target = self.lowering.target().clone();
        let slots = self.spaces.len();

        // Classify and fold: a step all of whose inputs are constants becomes one.
        let mut constants: HashMap<Slot, (Value, Option<usize>)> = HashMap::new();
        for (slot, root) in &self.roots {
            if let Root::Constant { value, rows } = root {
                constants.insert(*slot, ((**value).clone(), *rows));
            }
        }
        // A selection whose index is fixed and shared by every row is the value it chooses.
        let mut alias: HashMap<Slot, Slot> = HashMap::new();
        let mut live = Vec::with_capacity(self.steps.len());
        for mut step in std::mem::take(&mut self.steps) {
            for input in &mut step.inputs {
                if let Some(&chosen) = alias.get(input) {
                    *input = chosen;
                }
            }
            if step.operation == Select.name()
                && let Some((Value::Discrete(index), None)) = constants.get(&step.inputs[0])
                && let Some(&k) = index.to_host().first()
                && let Some(&chosen) = usize::try_from(k).ok().and_then(|k| step.inputs.get(k + 1))
            {
                alias.insert(step.output, chosen);
                continue;
            }
            let fixed: Option<Vec<&(Value, Option<usize>)>> =
                step.inputs.iter().map(|slot| constants.get(slot)).collect();
            let Some(fixed) = fixed else {
                live.push(step);
                continue;
            };
            let rows = pinned(fixed.iter().map(|(_, rows)| *rows))?;
            let inputs = fixed
                .iter()
                .map(|(value, own)| match (own, rows) {
                    (None, Some(n)) => expand(value, n, &target),
                    _ => Ok(value.clone()),
                })
                .collect::<Result<Vec<Value>, Error>>()?;
            let value = step.kernel.forward().forward(&inputs);
            constants.insert(step.output, (value.clone(), rows));
            self.roots.push((
                step.output,
                Root::Constant {
                    value: Box::new(value),
                    rows,
                },
            ));
        }

        let targets: Vec<(NodeId, Slot)> = targets
            .into_iter()
            .map(|(node, slot)| (node, alias.get(&slot).copied().unwrap_or(slot)))
            .collect();

        // Prune: keep what a target needs.
        let mut needed = vec![false; slots];
        for (_, slot) in &targets {
            needed[*slot] = true;
        }
        let mut kept = Vec::with_capacity(live.len());
        for step in live.into_iter().rev() {
            if needed[step.output] {
                for &slot in &step.inputs {
                    needed[slot] = true;
                }
                kept.push(step);
            }
        }
        kept.reverse();
        let roots: Vec<(Slot, Root)> = self
            .roots
            .into_iter()
            .filter(|(slot, root)| needed[*slot] || matches!(root, Root::Free { .. }))
            .collect();
        let batch = pinned(roots.iter().filter_map(|(_, root)| match root {
            Root::Constant { rows, .. } => Some(*rows),
            Root::Free { .. } => None,
        }))?;

        // Finish: the last reader of each slot.
        let mut last_reader = vec![None; slots];
        for (k, step) in kept.iter().enumerate() {
            for &slot in &step.inputs {
                last_reader[slot] = Some(k);
            }
        }
        Ok(Program {
            target,
            spaces: self.spaces,
            roots,
            steps: kept,
            free: self.free,
            free_slots: self.free_slots,
            targets,
            batch,
            last_reader,
        })
    }
}

/// The batch values of these rows pin, shared ones pinning none (`BatchSize` when two differ).
fn pinned(rows: impl Iterator<Item = Option<usize>>) -> Result<Option<usize>, Error> {
    let mut batch = None;
    for n in rows.flatten() {
        match batch {
            Some(b) if b != n => {
                return Err(Error::BatchSize {
                    expected: b,
                    found: n,
                });
            }
            _ => batch = Some(n),
        }
    }
    Ok(batch)
}

/// The nodes `outputs` reach, back to quantities, inputs, constants, nodes fixed in the model and
/// nodes already bound, in increasing identifier: an order in which every node follows its inputs.
fn reach(
    model: &Model,
    outputs: &[NodeId],
    env: &HashMap<NodeId, Slot>,
) -> Result<Vec<NodeId>, Error> {
    let mut seen = HashSet::new();
    let mut stack: Vec<NodeId> = outputs.to_vec();
    while let Some(id) = stack.pop() {
        if env.contains_key(&id) || !seen.insert(id) {
            continue;
        }
        let node = model.nodes.get(&id).ok_or(Error::UnknownNode { id })?;
        if !node.is_fixed() {
            stack.extend(node.inputs().iter().copied());
        }
    }
    let mut order: Vec<NodeId> = seen.into_iter().collect();
    order.sort_unstable();
    Ok(order)
}
