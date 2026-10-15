//! A compiled program: steps over slots, and the passes that run them.

use std::collections::HashMap;

use skoro_tensor::{Frame, Integers, Points, Target, Value};

use crate::{Error, Kernel, NodeId, Order, Pullback, QDyn, SecondPullback, Space};

/// A slot of a program: the value one node of the compilation graph takes.
pub(crate) type Slot = usize;

/// How a root slot gets its value.
pub(crate) enum Root {
    /// The `index`-th free quantity, from the values a pass is given.
    Free { index: usize },
    /// A constant of the program: shared by every row (`[1, storage]`) or one per row.
    Constant {
        value: Box<Value>,
        rows: Option<usize>,
    },
}

/// One step: a kernel, the slots it reads, and the slot it writes.
pub(crate) struct Step {
    pub(crate) kernel: Kernel,
    pub(crate) inputs: Vec<Slot>,
    pub(crate) output: Slot,
    pub(crate) operation: &'static str,
    /// The composites the step was expanded from, outermost first.
    pub(crate) composites: Vec<&'static str>,
    /// The node of the model the step came from: the composite call it was expanded from, if any.
    pub(crate) node: NodeId,
}

/// A free quantity of a program.
#[derive(Clone, PartialEq, Debug)]
pub struct Free {
    pub node: NodeId,
    pub space: Space,
}

/// A query compiled for one target.
pub struct Program {
    pub(crate) target: Target,
    pub(crate) spaces: Vec<Space>,
    pub(crate) roots: Vec<(Slot, Root)>,
    pub(crate) steps: Vec<Step>,
    pub(crate) free: Vec<Free>,
    pub(crate) free_slots: Vec<Slot>,
    pub(crate) targets: Vec<(NodeId, Slot)>,
    /// The batch the program's constants pin, if any does.
    pub(crate) batch: Option<usize>,
    /// The last step that reads each slot.
    pub(crate) last_reader: Vec<Option<usize>>,
}

impl std::fmt::Debug for Program {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Program")
            .field("free", &self.free)
            .field("targets", &self.targets())
            .field("steps", &self.steps.len())
            .field("precision", &self.target.precision())
            .finish_non_exhaustive()
    }
}

/// Values by node.
#[derive(Clone, Debug, Default)]
pub struct Values {
    values: HashMap<NodeId, Value>,
}

/// Frames by node.
#[derive(Clone, Debug, Default)]
pub struct Frames {
    frames: HashMap<NodeId, Frame>,
}

/// What a Hessian pass gives: the targets' values, and `α` and `β` at the free continuous
/// quantities.
#[derive(Clone, Debug)]
pub struct Second {
    pub values: Values,
    pub alpha: Frames,
    pub beta: Frames,
}

impl Values {
    pub fn new() -> Values {
        Values::default()
    }

    pub fn with(mut self, q: impl Into<QDyn>, value: Value) -> Values {
        self.insert(q, value);
        self
    }

    pub fn insert(&mut self, q: impl Into<QDyn>, value: Value) {
        self.values.insert(q.into().id(), value);
    }

    pub fn get(&self, q: impl Into<QDyn>) -> Option<&Value> {
        self.values.get(&q.into().id())
    }

    pub fn remove(&mut self, q: impl Into<QDyn>) -> Option<Value> {
        self.values.remove(&q.into().id())
    }

    pub fn len(&self) -> usize {
        self.values.len()
    }

    pub fn is_empty(&self) -> bool {
        self.values.is_empty()
    }
}

impl Frames {
    pub fn new() -> Frames {
        Frames::default()
    }

    pub fn with(mut self, q: impl Into<QDyn>, frame: Frame) -> Frames {
        self.insert(q, frame);
        self
    }

    pub fn insert(&mut self, q: impl Into<QDyn>, frame: Frame) {
        self.frames.insert(q.into().id(), frame);
    }

    pub fn get(&self, q: impl Into<QDyn>) -> Option<&Frame> {
        self.frames.get(&q.into().id())
    }

    pub fn len(&self) -> usize {
        self.frames.len()
    }

    pub fn is_empty(&self) -> bool {
        self.frames.is_empty()
    }

    fn get_id(&self, id: NodeId) -> Option<&Frame> {
        self.frames.get(&id)
    }

    fn ids(&self) -> impl Iterator<Item = NodeId> + '_ {
        self.frames.keys().copied()
    }
}

/// What a pass is given beyond the values.
enum Mode<'f> {
    Forward,
    Tangent {
        tangents: &'f Frames,
    },
    Cotangent {
        seeds: &'f Frames,
    },
    Hessian {
        velocities: &'f Frames,
        seeds: &'f Frames,
    },
}

/// What a pass gives back.
struct Outcome {
    values: Values,
    tangents: Frames,
    alpha: Frames,
    beta: Frames,
}

/// What a step recorded for the reverse sweep.
enum Recorded {
    First(Box<dyn Pullback>),
    Second(Box<dyn SecondPullback>),
}

impl Recorded {
    fn cotangent(&self, alpha: &Frame) -> Vec<Option<Frame>> {
        match self {
            Recorded::First(pullback) => pullback.cotangent(alpha),
            Recorded::Second(pullback) => pullback.cotangent(alpha),
        }
    }
}

/// What a pass asks of a step.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Need {
    /// The forward map alone: the step is not on the pass.
    Forward,
    /// The tangent map.
    Tangent,
    /// The pull-back, into the inputs reached from the differentiated roots.
    Record,
    /// The pull-back and the second-order map, at the inputs' velocities.
    Second,
}

impl Need {
    fn order(self) -> Order {
        match self {
            Need::Forward => Order::Zero,
            Need::Tangent | Need::Record => Order::One,
            Need::Second => Order::Two,
        }
    }
}

impl Program {
    /// The free quantities, in ancestral order.
    pub fn free(&self) -> &[Free] {
        &self.free
    }

    /// The free discrete quantities.
    pub fn free_discrete(&self) -> Vec<NodeId> {
        self.free
            .iter()
            .filter(|free| !free.space.is_continuous())
            .map(|free| free.node)
            .collect()
    }

    /// The program's targets, in the order the query named them.
    pub fn targets(&self) -> Vec<NodeId> {
        self.targets.iter().map(|(node, _)| *node).collect()
    }

    /// The number of steps the program runs.
    pub fn len(&self) -> usize {
        self.steps.len()
    }

    pub fn is_empty(&self) -> bool {
        self.steps.is_empty()
    }

    /// The most values a forward pass holds at once, its roots and targets among them.
    #[doc(hidden)]
    pub fn peak_values(&self) -> usize {
        let mut live: usize = self.roots.len();
        let mut peak = live;
        let held: std::collections::HashSet<Slot> =
            self.targets.iter().map(|(_, slot)| *slot).collect();
        for (k, step) in self.steps.iter().enumerate() {
            live += 1;
            peak = peak.max(live);
            let mut freed: Vec<Slot> = step
                .inputs
                .iter()
                .copied()
                .filter(|&slot| self.last_reader[slot] == Some(k) && !held.contains(&slot))
                .collect();
            freed.sort_unstable();
            freed.dedup();
            live -= freed.len();
        }
        peak
    }

    /// The targets' values.
    pub fn forward(&self, at: &Values) -> Result<Values, Error> {
        Ok(self.run(at, Mode::Forward)?.values)
    }

    /// Tangent frames at free continuous quantities in, at the targets out. A free quantity given
    /// no frame has a zero one.
    pub fn tangent(&self, at: &Values, tangents: &Frames) -> Result<(Values, Frames), Error> {
        let outcome = self.run(at, Mode::Tangent { tangents })?;
        Ok((outcome.values, outcome.tangents))
    }

    /// Cotangent frames at the targets in, at every free continuous quantity out.
    pub fn cotangent(&self, at: &Values, seeds: &Frames) -> Result<(Values, Frames), Error> {
        let outcome = self.run(at, Mode::Cotangent { seeds })?;
        Ok((outcome.values, outcome.alpha))
    }

    /// Velocities at some free continuous quantities, one cotangent column at the targets: `α`
    /// and `β` at every free continuous quantity.
    pub fn hessian(
        &self,
        at: &Values,
        velocities: &Frames,
        seeds: &Frames,
    ) -> Result<Second, Error> {
        let outcome = self.run(at, Mode::Hessian { velocities, seeds })?;
        Ok(Second {
            values: outcome.values,
            alpha: outcome.alpha,
            beta: outcome.beta,
        })
    }

    /// `cotangent` of a program's single `Real` target, seeded with one.
    pub fn gradient(&self, at: &Values) -> Result<(Values, Frames), Error> {
        let seeds = self.unit_seed(at)?;
        self.cotangent(at, &seeds)
    }

    /// `hessian` of a program's single `Real` target, seeded with one.
    pub fn hessian_products(&self, at: &Values, velocities: &Frames) -> Result<Second, Error> {
        let seeds = self.unit_seed(at)?;
        self.hessian(at, velocities, &seeds)
    }

    fn unit_seed(&self, at: &Values) -> Result<Frames, Error> {
        let [(node, slot)] = self.targets.as_slice() else {
            return Err(Error::SpaceMismatch {
                operation: "gradient",
                inputs: self
                    .targets
                    .iter()
                    .map(|(_, s)| self.spaces[*s].clone())
                    .collect(),
            });
        };
        if self.spaces[*slot] != Space::Real {
            return Err(Error::SpaceMismatch {
                operation: "gradient",
                inputs: vec![self.spaces[*slot].clone()],
            });
        }
        let rows = self.rows(at)?;
        let ones = vec![1.0; rows];
        Ok(Frames::new().with(
            QDyn::from_id(*node),
            Frame::from_host(&ones, rows, 1, 1, &self.target),
        ))
    }

    /// The batch a pass runs at: the rows of every value given, all of one number, and of the
    /// constants that pin one.
    fn rows(&self, at: &Values) -> Result<usize, Error> {
        let mut rows = None;
        for free in &self.free {
            let value = at
                .values
                .get(&free.node)
                .ok_or(Error::MissingValue { id: free.node })?;
            let fits = match (value, free.space.is_continuous()) {
                (Value::Continuous(points), true) => points.storage() == free.space.storage(),
                (Value::Discrete(integers), false) => integers.storage() == free.space.storage(),
                _ => false,
            };
            if !fits {
                return Err(Error::ValueShape {
                    space: free.space.clone(),
                });
            }
            match rows {
                None => rows = Some(value.rows()),
                Some(n) if n != value.rows() => {
                    return Err(Error::BatchSize {
                        expected: n,
                        found: value.rows(),
                    });
                }
                Some(_) => {}
            }
        }
        match (rows, self.batch) {
            (Some(n), Some(pinned)) if n != pinned => Err(Error::BatchSize {
                expected: pinned,
                found: n,
            }),
            (Some(n), _) => Ok(n),
            (None, pinned) => Ok(pinned.unwrap_or(1)),
        }
    }

    /// Checks frames given at `ids` (free quantities or targets), and gives their columns.
    fn columns(
        &self,
        frames: &Frames,
        rows: usize,
        at: &[(NodeId, Slot)],
    ) -> Result<Option<usize>, Error> {
        let mut columns = None;
        for id in frames.ids() {
            let Some((_, slot)) = at.iter().find(|(node, _)| *node == id) else {
                return Err(Error::FrameShape { id });
            };
            let Some(dimension) = self.spaces[*slot].dimension() else {
                return Err(Error::FrameShape { id });
            };
            let Some(frame) = frames.get_id(id) else {
                unreachable!("the identifier was read off the frames")
            };
            let fits = frame.rows() == rows
                && frame.dimension() == dimension
                && columns.is_none_or(|k| k == frame.columns());
            if !fits {
                return Err(Error::FrameShape { id });
            }
            columns = Some(frame.columns());
        }
        Ok(columns)
    }

    fn run(&self, at: &Values, mode: Mode<'_>) -> Result<Outcome, Error> {
        let rows = self.rows(at)?;
        let free_at: Vec<(NodeId, Slot)> = self
            .free
            .iter()
            .zip(&self.free_slots)
            .map(|(free, slot)| (free.node, *slot))
            .collect();
        let (seeded, seeds) = match &mode {
            Mode::Forward => (None, None),
            Mode::Tangent { tangents } => (Some(*tangents), None),
            Mode::Cotangent { seeds } => (None, Some(*seeds)),
            Mode::Hessian { velocities, seeds } => (Some(*velocities), Some(*seeds)),
        };
        let velocity_columns = match seeded {
            Some(frames) => self.columns(frames, rows, &free_at)?,
            None => None,
        };
        let seed_columns = match seeds {
            Some(frames) => self.columns(frames, rows, &self.targets)?,
            None => None,
        };
        if matches!(mode, Mode::Hessian { .. }) && seed_columns.is_some_and(|k| k != 1) {
            let id = self.targets.first().map_or(NodeId::FAILED, |(id, _)| *id);
            return Err(Error::FrameShape { id });
        }

        let seeded_ids: Vec<NodeId> = seeded.map(|f| f.ids().collect()).unwrap_or_default();
        let target_ids: Vec<NodeId> = seeds.map(|f| f.ids().collect()).unwrap_or_default();
        let needs = match mode {
            Mode::Forward => vec![Need::Forward; self.steps.len()],
            Mode::Tangent { .. } => {
                let pass = Pass::Tangent {
                    seeded: &seeded_ids,
                };
                self.admits(&pass)?;
                self.needs(&pass)
            }
            Mode::Cotangent { .. } => {
                let pass = Pass::Cotangent {
                    targets: &target_ids,
                };
                self.admits(&pass)?;
                self.needs(&pass)
            }
            Mode::Hessian { .. } => {
                let pass = Pass::Hessian {
                    seeded: &seeded_ids,
                    targets: &target_ids,
                };
                self.admits(&pass)?;
                self.needs(&pass)
            }
        };
        let from_free = self.reached(&self.free_slots_continuous());

        let slots = self.spaces.len();
        let mut values: Vec<Option<Value>> = vec![None; slots];
        let mut velocities: Vec<Option<Frame>> = vec![None; slots];
        for (slot, root) in &self.roots {
            values[*slot] = Some(match root {
                Root::Free { index } => {
                    let node = self.free[*index].node;
                    let Some(value) = at.values.get(&node) else {
                        unreachable!("rows() checked every free quantity has a value")
                    };
                    if let Some(frame) = seeded.and_then(|frames| frames.get_id(node)) {
                        velocities[*slot] = Some(frame.clone());
                    }
                    value.clone()
                }
                Root::Constant { value, rows: None } => expand(value, rows, &self.target)?,
                Root::Constant { value, .. } => (**value).clone(),
            });
        }
        let held: std::collections::HashSet<Slot> = self.targets.iter().map(|(_, s)| *s).collect();
        let mut records: Vec<Option<Recorded>> = Vec::with_capacity(self.steps.len());
        for (k, (step, need)) in self.steps.iter().zip(&needs).enumerate() {
            let inputs: Vec<Value> = step
                .inputs
                .iter()
                .map(|&slot| match &values[slot] {
                    Some(value) => value.clone(),
                    None => unreachable!("a slot is freed only after its last reader"),
                })
                .collect();
            let differentiable: Vec<bool> = step
                .inputs
                .iter()
                .map(|&slot| self.spaces[slot].is_continuous() && from_free[slot])
                .collect();
            let input_velocities: Vec<Option<Frame>> = step
                .inputs
                .iter()
                .map(|&slot| velocities[slot].clone())
                .collect();
            let (value, velocity, record) = match need {
                Need::Forward => (step.kernel.forward().forward(&inputs), None, None),
                Need::Tangent => {
                    let Some(first) = step.kernel.first() else {
                        unreachable!("orders were checked before the pass ran")
                    };
                    let (value, velocity) = first.tangent(&inputs, &input_velocities);
                    (value, velocity, None)
                }
                Need::Record => {
                    let Some(first) = step.kernel.first() else {
                        unreachable!("orders were checked before the pass ran")
                    };
                    let (value, pullback) = first.record(&inputs, &differentiable);
                    (value, None, Some(Recorded::First(pullback)))
                }
                Need::Second => {
                    let Some(second) = step.kernel.second() else {
                        unreachable!("orders were checked before the pass ran")
                    };
                    let (value, velocity, pullback) =
                        second.record_second(&inputs, &differentiable, &input_velocities);
                    (value, velocity, Some(Recorded::Second(pullback)))
                }
            };
            values[step.output] = Some(value);
            velocities[step.output] = velocity;
            records.push(record);
            for &slot in &step.inputs {
                if self.last_reader[slot] == Some(k) && !held.contains(&slot) {
                    values[slot] = None;
                    velocities[slot] = None;
                }
            }
        }

        let mut outcome = Outcome {
            values: Values::new(),
            tangents: Frames::new(),
            alpha: Frames::new(),
            beta: Frames::new(),
        };
        for (node, slot) in &self.targets {
            let Some(value) = values[*slot].clone() else {
                unreachable!("a target's slot is never freed")
            };
            outcome.values.values.insert(*node, value);
            if let Mode::Tangent { .. } = mode
                && let Some(dimension) = self.spaces[*slot].dimension()
            {
                let columns = velocity_columns.unwrap_or(1);
                let frame = match &velocities[*slot] {
                    Some(frame) => frame.clone(),
                    None => Frame::zeros(rows, dimension, columns, &self.target),
                };
                outcome.tangents.frames.insert(*node, frame);
            }
        }
        let Some(seeds) = seeds else {
            return Ok(outcome);
        };

        let mut alpha: Vec<Option<Frame>> = vec![None; slots];
        let mut beta: Vec<Option<Frame>> = vec![None; slots];
        for (node, slot) in &self.targets {
            if let Some(seed) = seeds.get_id(*node) {
                alpha[*slot] = Some(add(alpha[*slot].take(), seed.clone(), &self.target));
            }
        }
        let hessian = matches!(mode, Mode::Hessian { .. });
        for (step, record) in self.steps.iter().zip(records).rev() {
            let Some(record) = record else { continue };
            let (a, b) = (alpha[step.output].take(), beta[step.output].take());
            if let Some(a) = &a {
                for (&slot, cotangent) in step.inputs.iter().zip(record.cotangent(a)) {
                    if let Some(cotangent) = cotangent {
                        alpha[slot] = Some(add(alpha[slot].take(), cotangent, &self.target));
                    }
                }
                if let (true, Recorded::Second(pullback)) = (hessian, &record) {
                    for (&slot, hvp) in step.inputs.iter().zip(pullback.second(a)) {
                        if let Some(hvp) = hvp {
                            beta[slot] = Some(add(beta[slot].take(), hvp, &self.target));
                        }
                    }
                }
            }
            if let Some(b) = &b {
                for (&slot, cotangent) in step.inputs.iter().zip(record.cotangent(b)) {
                    if let Some(cotangent) = cotangent {
                        beta[slot] = Some(add(beta[slot].take(), cotangent, &self.target));
                    }
                }
            }
        }
        for (free, slot) in self.free.iter().zip(&self.free_slots) {
            let Some(dimension) = free.space.dimension() else {
                continue;
            };
            let zeros = |columns| Frame::zeros(rows, dimension, columns, &self.target);
            let a = alpha[*slot]
                .take()
                .unwrap_or_else(|| zeros(seed_columns.unwrap_or(1)));
            outcome.alpha.frames.insert(free.node, a);
            if hessian {
                let b = beta[*slot]
                    .take()
                    .unwrap_or_else(|| zeros(velocity_columns.unwrap_or(1)));
                outcome.beta.frames.insert(free.node, b);
            }
        }
        Ok(outcome)
    }

    fn free_slots_continuous(&self) -> Vec<Slot> {
        self.free
            .iter()
            .zip(&self.free_slots)
            .filter(|(free, _)| free.space.is_continuous())
            .map(|(_, slot)| *slot)
            .collect()
    }

    /// The slots reached from `roots` through continuous values.
    fn reached(&self, roots: &[Slot]) -> Vec<bool> {
        let mut reached = vec![false; self.spaces.len()];
        for &slot in roots {
            reached[slot] = true;
        }
        for step in &self.steps {
            if self.spaces[step.output].is_continuous()
                && step
                    .inputs
                    .iter()
                    .any(|&slot| reached[slot] && self.spaces[slot].is_continuous())
            {
                reached[step.output] = true;
            }
        }
        reached
    }

    /// The slots that reach `targets` through continuous values.
    fn reaching(&self, targets: &[Slot]) -> Vec<bool> {
        let mut reaches = vec![false; self.spaces.len()];
        for &slot in targets {
            reaches[slot] = true;
        }
        for step in self.steps.iter().rev() {
            if reaches[step.output] && self.spaces[step.output].is_continuous() {
                for &slot in &step.inputs {
                    if self.spaces[slot].is_continuous() {
                        reaches[slot] = true;
                    }
                }
            }
        }
        reaches
    }

    /// What the pass asks of each step.
    /// Whether `pass` can run: every step on it has a kernel of the order it requires
    /// (`OrderTooLow`, naming the first that does not, the composites it was expanded from and its
    /// node). Computed before anything runs.
    pub fn admits(&self, pass: &Pass<'_>) -> Result<(), Error> {
        for (step, need) in self.steps.iter().zip(self.needs(pass)) {
            if step.kernel.order() < need.order() {
                return Err(Error::OrderTooLow {
                    operation: step.operation,
                    composites: step.composites.clone(),
                    node: step.node,
                });
            }
        }
        Ok(())
    }

    /// The free continuous quantities' slots among `ids`.
    fn slots_of(&self, ids: &[NodeId], at: &[(NodeId, Slot)]) -> Vec<Slot> {
        at.iter()
            .filter(|(node, _)| ids.contains(node))
            .map(|(_, slot)| *slot)
            .collect()
    }

    /// What `pass` asks of each step. A step is on the pass when its output is continuous and
    /// reaches a target, and one of its continuous inputs is reached from a free continuous
    /// quantity (from a seeded one, for a tangent pass); it then requires order two when the pass
    /// is a Hessian pass and one of its inputs is reached from a seeded quantity, and order one
    /// otherwise.
    fn needs(&self, pass: &Pass<'_>) -> Vec<Need> {
        let free_at: Vec<(NodeId, Slot)> = self
            .free
            .iter()
            .zip(&self.free_slots)
            .filter(|(free, _)| free.space.is_continuous())
            .map(|(free, slot)| (free.node, *slot))
            .collect();
        let (seeded, targets): (&[NodeId], Vec<Slot>) = match pass {
            Pass::Tangent { seeded } => {
                (seeded, self.targets.iter().map(|(_, slot)| *slot).collect())
            }
            Pass::Cotangent { targets } => (&[], self.slots_of(targets, &self.targets)),
            Pass::Hessian { seeded, targets } => (seeded, self.slots_of(targets, &self.targets)),
        };
        let from_free = self.reached(&self.free_slots_continuous());
        let from_seeded = self.reached(&self.slots_of(seeded, &free_at));
        let reaches = self.reaching(&targets);
        self.steps
            .iter()
            .map(|step| {
                let continuous = self.spaces[step.output].is_continuous();
                let fed = |reached: &[bool]| {
                    step.inputs
                        .iter()
                        .any(|&slot| reached[slot] && self.spaces[slot].is_continuous())
                };
                let on = |reached: &[bool]| continuous && reaches[step.output] && fed(reached);
                match pass {
                    Pass::Tangent { .. } if on(&from_seeded) => Need::Tangent,
                    Pass::Cotangent { .. } if on(&from_free) => Need::Record,
                    Pass::Hessian { .. } if on(&from_free) && fed(&from_seeded) => Need::Second,
                    Pass::Hessian { .. } if on(&from_free) => Need::Record,
                    _ => Need::Forward,
                }
            })
            .collect()
    }
}

/// A pass over a program, by the roots it seeds with tangents or velocities and the targets it
/// seeds with cotangents.
#[derive(Clone, Copy, Debug)]
pub enum Pass<'a> {
    Tangent {
        seeded: &'a [NodeId],
    },
    Cotangent {
        targets: &'a [NodeId],
    },
    Hessian {
        seeded: &'a [NodeId],
        targets: &'a [NodeId],
    },
}

/// A value shared by every row, `[1, storage]`, expanded to `rows`.
pub(crate) fn expand(value: &Value, rows: usize, target: &Target) -> Result<Value, Error> {
    Ok(match value {
        Value::Continuous(points) => {
            let storage = points.storage();
            let tensor = points.clone().into_burn().expand([rows, storage]);
            Value::Continuous(Points::from_burn(tensor, target)?)
        }
        Value::Discrete(integers) => {
            let storage = integers.storage();
            let tensor = integers.clone().into_burn().expand([rows, storage]);
            Value::Discrete(Integers::from_burn(tensor, target)?)
        }
    })
}

fn add(sum: Option<Frame>, frame: Frame, target: &Target) -> Frame {
    let Some(sum) = sum else {
        return frame;
    };
    match Frame::from_burn(sum.into_burn() + frame.into_burn(), target) {
        Ok(frame) => frame,
        Err(error) => unreachable!("frames of one pass share its target: {error}"),
    }
}
