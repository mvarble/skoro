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
use std::sync::Arc;

use skoro_tensor::{HostValue, Target, Value};

use crate::model::{Model, NodeKind};
use crate::operation::Link;
use crate::program::{Free, Root, Slot, Step, expand};
use crate::scope::fits;
use crate::{
    Composite, Error, Lowered, Lowering, NodeId, Observations, Program, Query, Snapshot, Space,
};

impl Snapshot {
    /// The program of `query` on `target`.
    pub fn compile(&self, query: &Query, target: &Target) -> Result<Arc<Program>, Error> {
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
        let mut lowering = Lowering::new(target);
        let mut builder = Builder::new(&mut lowering);
        let mut env = HashMap::new();
        let outputs = builder.emit(&self.0, &query.targets, &mut env, None, Some(observations))?;
        let targets = query.targets.iter().copied().zip(outputs).collect();
        builder.finish(targets).map(Arc::new)
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
        let mut live = Vec::with_capacity(self.steps.len());
        for step in std::mem::take(&mut self.steps) {
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
