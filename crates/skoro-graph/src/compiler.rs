use std::{collections::HashMap, marker::PhantomData, sync::Arc};

use skoro_backend::{Backend, CotangentFrame, TangentFrame, TangentVector, Tensor};
use skoro_core::{
    any_tensor::QuantityValue,
    error::{SkoroError, SkoroResult},
    id::Id,
    kind::QuantityKind,
    sampler::DynSequenceSampler,
};
use skoro_operation::{DynSequenceOperation, Input, tape::Tape};

use crate::{point_map::PointMap, quantity::DynQuantityType};

#[derive(Debug)]
pub enum CompiledStep<B: Backend> {
    Constant {
        id: Id,
        value: QuantityValue<B>,
        quantity_type: Arc<dyn DynQuantityType<B>>,
    },
    Parameter {
        id: Id,
        quantity_type: Arc<dyn DynQuantityType<B>>,
    },
    /// A cut boundary: a node whose value the caller supplies in place of computing it.
    ///
    /// It stands for a node that *does* have parents -- a mapped quantity, a log-density, a
    /// stochastic draw -- which the cut has pruned from the sequence. Supplied like a
    /// [`Constant`](Self::Constant) and differentiated like a [`Parameter`](Self::Parameter):
    /// the cotangent passes collect at it, because from the compiled program's point of view it
    /// is a free variable.
    ///
    /// Lowering a cut to `Constant` instead is the mistake this kind exists to prevent. A
    /// constant is not a differentiable input, so every consumer below one gets
    /// `constant_inputs`, which keeps it off the tape -- and the pass then returns correct
    /// values with the derivatives silently missing.
    Given {
        id: Id,
        quantity_type: Arc<dyn DynQuantityType<B>>,
    },
    Sampler {
        id: Id,
        sampler: Box<dyn DynSequenceSampler<B>>,
        quantity_type: Arc<dyn DynQuantityType<B>>,
    },
    Operation {
        id: Id,
        op: Box<dyn DynSequenceOperation<B>>,
        value: Option<QuantityValue<B>>,
        quantity_type: Arc<dyn DynQuantityType<B>>,
        /// The operation has no derivatives at all -- a discrete operation.
        ///
        /// No pass branches on this: `constant_inputs` is implied by it and is what the three
        /// differentiating passes actually gate on. It is kept because it says something
        /// `constant_inputs` does not, and that a reader of a compiled sequence wants: whether
        /// the derivative is *undefined* here or merely *unneeded*.
        forward_only: bool,
        /// No input of the operation is differentiable, so the pass reads its fold instead of
        /// re-entering it. Implied by `forward_only`, and true besides whenever every input is
        /// constant.
        constant_inputs: bool,
    },
}

impl<B: Backend> CompiledStep<B> {
    /// The graph node this step evaluates.
    pub fn id(&self) -> Id {
        match self {
            Self::Constant { id, .. }
            | Self::Parameter { id, .. }
            | Self::Given { id, .. }
            | Self::Sampler { id, .. }
            | Self::Operation { id, .. } => *id,
        }
    }
}

pub struct CompiledSequence<B: Backend> {
    pub steps: Vec<CompiledStep<B>>,
    /// For each slot, the index of the last step that reads it. See [`Self::last_use`].
    last_use: Vec<usize>,
    _phantom: PhantomData<B>,
}

impl<B: Backend> std::fmt::Debug for CompiledSequence<B> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CompiledSequence")
            .field("steps", &self.steps)
            .finish_non_exhaustive()
    }
}

/// Options for forward operation on a [`CompiledSequence`].
#[derive(Debug, Clone, Copy)]
pub struct ForwardOptions {
    /// Batch size of the evaluation. It must match axis 0 dimension of each value in
    /// `points`; it is used for constants and samplers.
    pub batch_size: usize,

    /// If true, the execution will sample from the model's internal sampling mechanisms for
    /// stochastic quantities which are not already present in `points`. If false, it panics.
    pub use_model_samplers: bool,
}

impl Default for ForwardOptions {
    fn default() -> Self {
        Self {
            batch_size: 1,
            use_model_samplers: false,
        }
    }
}

/// Input data for performing a forward operation on a [`CompiledSequence`].
#[derive(Debug, Clone)]
pub struct ForwardInput<'a, B: Backend> {
    /// Values at which we are evaluating the graph operation, continuous or
    /// discrete, keyed by node id.
    pub points: &'a HashMap<Id, Input<B>>,
    /// Options for forward evaluation.
    pub options: ForwardOptions,
}

/// The values produced by a forward pass over a [`CompiledSequence`].
///
/// Observed quantities lower to [`CompiledStep::Constant`], so their values arrive in
/// `constants` alongside genuine constants — there is no separate observation map.
#[derive(Debug, Clone, Default)]
pub struct ForwardOutput<B: Backend> {
    pub variables: PointMap<B>,
    pub constants: PointMap<B>,
    pub evaluations: PointMap<B>,
}

/// Input data for performing a JVP forward operation on a [`CompiledSequence`].
#[derive(Debug, Clone)]
pub struct JvpForwardInput<'a, B: Backend> {
    /// Values at which we are evaluating the graph operation.
    pub points: &'a HashMap<Id, Input<B>>,
    /// Tangent frames at points at which we are evaluating the JVP graph operation.
    pub tangents: &'a HashMap<Id, TangentFrame<B>>,
    /// Options for forward evaluation.
    pub options: ForwardOptions,
}

#[derive(Debug, Clone, Default)]
pub struct JvpForwardOutput<B: Backend> {
    pub primal: ForwardOutput<B>,
    pub tangents: HashMap<Id, TangentFrame<B>>,
}

/// Input data for performing a VJP forward operation on a [`CompiledSequence`].
#[derive(Debug, Clone)]
pub struct VjpForwardInput<'a, B: Backend> {
    /// Values at which we are evaluating the graph operation.
    pub points: &'a HashMap<Id, Input<B>>,
    /// Cotangent frames at points at which we are evaluating the VJP graph operation.
    pub cotangents: &'a HashMap<Id, CotangentFrame<B>>,
    /// Options for forward evaluation.
    pub options: ForwardOptions,
}

#[derive(Debug, Clone, Default)]
pub struct VjpForwardOutput<B: Backend> {
    pub primal: ForwardOutput<B>,
    pub cotangents: HashMap<Id, CotangentFrame<B>>,
}

/// Input data for performing a HVP forward operation on a [`CompiledSequence`].
#[derive(Debug, Clone)]
pub struct HvpForwardInput<'a, B: Backend> {
    /// Values at which we are evaluating the graph operation.
    pub points: &'a HashMap<Id, Input<B>>,
    /// Tangent vectors at points at which we are evaluating the HVP graph operation.
    pub tangents: &'a HashMap<Id, TangentVector<B>>,
    /// Order-1 cotangent frames at points at which we are evaluating the HVP graph operation.
    pub ord1_cotangents: &'a HashMap<Id, CotangentFrame<B>>,
    /// Order-2 cotangent frames at points at which we are evaluating the HVP graph operation.
    pub ord2_cotangents: &'a HashMap<Id, CotangentFrame<B>>,
    /// Options for forward evaluation.
    pub options: ForwardOptions,
}

#[derive(Debug, Clone, Default)]
pub struct HvpForwardOutput<B: Backend> {
    pub primal: ForwardOutput<B>,
    pub tangents: HashMap<Id, TangentVector<B>>,
    pub ord1_cotangents: HashMap<Id, CotangentFrame<B>>,
    pub ord2_cotangents: HashMap<Id, CotangentFrame<B>>,
}

/// The index of the last step that reads each slot.
///
/// Reads are gathered from *every* step kind, not only operations: `input_indices` is on
/// `DynSequenceSampler` as well as `DynSequenceOperation`, and a sampler step reads the point
/// sequence when it draws. A slot nothing reads gets its own index, which is right -- its value
/// is cloned into the output as the step runs, and the slot is dead immediately after.
fn compute_last_use<B: Backend>(steps: &[CompiledStep<B>]) -> Vec<usize> {
    let mut last_use: Vec<usize> = (0..steps.len()).collect();
    for (index, step) in steps.iter().enumerate() {
        let inputs: &[usize] = match step {
            CompiledStep::Operation { op, .. } => op.input_indices(),
            CompiledStep::Sampler { sampler, .. } => sampler.input_indices(),
            CompiledStep::Constant { .. }
            | CompiledStep::Parameter { .. }
            | CompiledStep::Given { .. } => &[],
        };
        for &input in inputs {
            // Steps are visited in ascending order and an input always precedes its consumer,
            // so the last write wins without a `max`.
            last_use[input] = index;
        }
    }
    last_use
}

/// Free every slot that step `index` was the last to read.
///
/// Restricted to operation slots. A leaf's slot is read again after `tape.backward`, by the
/// cotangent collection that shapes the zero cotangent of a leaf reaching no target from the
/// supplied point's dims and device -- so releasing one would panic there rather than save
/// anything, since a leaf's value is the caller's tensor and is alive regardless.
#[inline(always)]
fn release_dead_points<B: Backend>(
    steps: &[CompiledStep<B>],
    last_use: &[usize],
    index: usize,
    point_seq: &mut [Option<Input<B>>],
) {
    let inputs: &[usize] = match &steps[index] {
        CompiledStep::Operation { op, .. } => op.input_indices(),
        CompiledStep::Sampler { sampler, .. } => sampler.input_indices(),
        CompiledStep::Constant { .. }
        | CompiledStep::Parameter { .. }
        | CompiledStep::Given { .. } => &[],
    };
    for &slot in inputs.iter().chain(std::iter::once(&index)) {
        if last_use[slot] == index && matches!(steps[slot], CompiledStep::Operation { .. }) {
            point_seq[slot] = None;
        }
    }
}

impl<B: Backend> CompiledSequence<B> {
    pub fn new(steps: Vec<CompiledStep<B>>) -> Self {
        Self {
            last_use: compute_last_use(&steps),
            steps,
            _phantom: PhantomData,
        }
    }

    /// For each slot, the index of the last step that reads it.
    ///
    /// Indices are into [`steps`](Self::steps): slot `i` holds the value step `i` produces. A
    /// slot whose entry is its own index is read by nothing.
    pub fn last_use(&self) -> &[usize] {
        &self.last_use
    }

    pub fn forward(&self, input: ForwardInput<'_, B>) -> SkoroResult<ForwardOutput<B>> {
        let mut point_seq: Vec<Option<Input<B>>> = vec![None; self.steps.len()];
        let mut out = ForwardOutput::default();
        for (index, step) in self.steps.iter().enumerate() {
            match step {
                CompiledStep::Operation {
                    id,
                    op,
                    value,
                    quantity_type,
                    ..
                } => {
                    forward_only_op(
                        *id,
                        op,
                        value,
                        quantity_type,
                        index,
                        &input.options,
                        &mut point_seq,
                        &mut out,
                    )?;
                }
                _ => {
                    non_op_forward(
                        index,
                        step,
                        input.points,
                        &input.options,
                        &mut point_seq,
                        &mut out,
                        None,
                        None,
                        None,
                        None,
                    )?;
                }
            };
            release_dead_points(&self.steps, &self.last_use, index, &mut point_seq);
        }
        Ok(out)
    }

    pub fn jvp_forward(&self, input: JvpForwardInput<'_, B>) -> SkoroResult<JvpForwardOutput<B>> {
        let steps_len = self.steps.len();
        let mut point_seq: Vec<Option<Input<B>>> = vec![None; steps_len];
        let mut tangent_seq: Vec<Option<TangentFrame<B>>> = vec![None; steps_len];
        let mut out = JvpForwardOutput::default();
        let mut frame_count: Option<usize> = None;
        for (index, step) in self.steps.iter().enumerate() {
            match step {
                CompiledStep::Operation {
                    id,
                    op,
                    value,
                    quantity_type,
                    constant_inputs,
                    ..
                } => {
                    // Gated on `constant_inputs` rather than `forward_only`, matching
                    // `vjp_forward` and `hvp_forward`. A node whose every input is constant has
                    // already folded, and re-entering its operation in JVP mode would demand a
                    // pushforward from an operation the graph never has to ask one of.
                    if *constant_inputs {
                        forward_only_op(
                            *id,
                            op,
                            value,
                            quantity_type,
                            index,
                            &input.options,
                            &mut point_seq,
                            &mut out.primal,
                        )?;
                        if tangent_seq[index].is_none()
                            && let Some(frame) = input.tangents.get(id)
                        {
                            tangent_seq[index] = Some(frame.clone());
                            out.tangents.insert(*id, frame.clone());
                        }
                    } else {
                        op.jvp_forward(&mut point_seq, &mut tangent_seq);
                        out.primal.evaluations.insert(
                            *id,
                            point_seq[index].clone().unwrap(),
                            quantity_type.clone(),
                        );
                        if let Some(tangent) = &tangent_seq[index] {
                            out.tangents.insert(*id, tangent.clone());
                        }
                    }
                    check_tensor_frames(&mut frame_count, tangent_seq[index].as_ref())?;
                }
                _ => {
                    // Seeds are checked here, before any kernel sees them, for the same reason
                    // `vjp_forward` checks its supplied cotangents: two seeds of differing width
                    // is a caller error with a name, and letting it reach the operation turns it
                    // into a broadcast panic from inside burn that names neither quantity.
                    check_tensor_frames(&mut frame_count, input.tangents.get(&step.id()))?;
                    non_op_forward(
                        index,
                        step,
                        input.points,
                        &input.options,
                        &mut point_seq,
                        &mut out.primal,
                        None,
                        None,
                        Some(input.tangents),
                        Some(&mut tangent_seq),
                    )?;
                }
            };
            release_dead_points(&self.steps, &self.last_use, index, &mut point_seq);
        }
        Ok(out)
    }

    pub fn vjp_forward(&self, input: VjpForwardInput<'_, B>) -> SkoroResult<VjpForwardOutput<B>> {
        let steps_len = self.steps.len();
        let mut point_seq: Vec<Option<Input<B>>> = vec![None; steps_len];
        let mut cotangent_seq: Vec<Option<CotangentFrame<B>>> = vec![None; steps_len];
        let mut out = VjpForwardOutput::default();
        // The seeds fix the pass's frame count even when no operation consumes one: a leaf's
        // zero cotangent below must still carry that many frames.
        let mut frame_count: Option<usize> = None;
        for cotangent in input.cotangents.values() {
            check_tensor_frames(&mut frame_count, Some(cotangent))?;
        }
        let mut tape = Tape::new();
        for (index, step) in self.steps.iter().enumerate() {
            match step {
                CompiledStep::Operation {
                    id,
                    op,
                    value,
                    quantity_type,
                    constant_inputs,
                    ..
                } => {
                    if *constant_inputs {
                        forward_only_op(
                            *id,
                            op,
                            value,
                            quantity_type,
                            index,
                            &input.options,
                            &mut point_seq,
                            &mut out.primal,
                        )?;
                    } else {
                        let cotangent = input.cotangents.get(id);
                        check_tensor_frames(&mut frame_count, cotangent)?;
                        if cotangent_seq[index].is_none()
                            && let Some(cotangent) = cotangent
                        {
                            cotangent_seq[index] = Some(cotangent.clone());
                        }
                        op.vjp_forward(&mut point_seq, &mut tape);
                        out.primal.evaluations.insert(
                            *id,
                            point_seq[index].clone().unwrap(),
                            quantity_type.clone(),
                        );
                    }
                }
                _ => {
                    non_op_forward(
                        index,
                        step,
                        input.points,
                        &input.options,
                        &mut point_seq,
                        &mut out.primal,
                        None,
                        None,
                        None,
                        None,
                    )?;
                }
            };
            release_dead_points(&self.steps, &self.last_use, index, &mut point_seq);
        }
        tape.backward(&mut cotangent_seq);
        for (index, step) in self.steps.iter().enumerate() {
            match step {
                CompiledStep::Parameter { id, quantity_type }
                | CompiledStep::Given { id, quantity_type }
                | CompiledStep::Sampler {
                    id, quantity_type, ..
                } => {
                    // Discrete variables receive no cotangents by construction.
                    if quantity_type.kind() == QuantityKind::Continuous {
                        if let Some(cotangent) = cotangent_seq[index].take() {
                            out.cotangents.insert(*id, cotangent);
                        } else {
                            // A leaf that never reaches the targets has zero derivative
                            // with respect to them. That is an answer, not a failure, and
                            // it is reachable from any Jacobian whose column set is wider
                            // than the subgraph feeding its rows.
                            let point = point_seq[index]
                                .as_ref()
                                .ok_or(SkoroError::MissingPoint { id: *id })?;
                            let dimension = quantity_type
                                .as_manifold()
                                .ok_or(SkoroError::NotContinuous { id: *id })?
                                .dimension();
                            let shape = [point.dims()[0], dimension, frame_count.unwrap_or(1)];
                            out.cotangents
                                .insert(*id, Tensor::zeros(shape, &point.device()));
                        }
                    }
                }
                _ => {}
            };
        }
        Ok(out)
    }

    pub fn hvp_forward(&self, input: HvpForwardInput<'_, B>) -> SkoroResult<HvpForwardOutput<B>> {
        let steps_len = self.steps.len();
        let mut point_seq: Vec<Option<Input<B>>> = vec![None; steps_len];
        let mut tangent_seq: Vec<Option<TangentVector<B>>> = vec![None; steps_len];
        let mut ord1_cotangent_seq: Vec<Option<CotangentFrame<B>>> = vec![None; steps_len];
        let mut ord2_cotangent_seq: Vec<Option<CotangentFrame<B>>> = vec![None; steps_len];
        let mut out = HvpForwardOutput::default();
        let mut frame_count: Option<usize> = None;
        let mut tape = Tape::new();
        for (index, step) in self.steps.iter().enumerate() {
            match step {
                CompiledStep::Operation {
                    id,
                    op,
                    value,
                    quantity_type,
                    constant_inputs,
                    ..
                } => {
                    if *constant_inputs {
                        forward_only_op(
                            *id,
                            op,
                            value,
                            quantity_type,
                            index,
                            &input.options,
                            &mut point_seq,
                            &mut out.primal,
                        )?;
                        if tangent_seq[index].is_none()
                            && let Some(vector) = input.tangents.get(id)
                        {
                            tangent_seq[index] = Some(vector.clone());
                            out.tangents.insert(*id, vector.clone());
                        }
                    } else {
                        let ord1_cotangent = input.ord1_cotangents.get(id);
                        check_tensor_frames(&mut frame_count, ord1_cotangent)?;
                        if ord1_cotangent_seq[index].is_none()
                            && let Some(cotangent) = ord1_cotangent
                        {
                            ord1_cotangent_seq[index] = Some(cotangent.clone());
                        }

                        let ord2_cotangent = input.ord2_cotangents.get(id);
                        check_tensor_frames(&mut frame_count, ord2_cotangent)?;
                        if ord2_cotangent_seq[index].is_none()
                            && let Some(cotangent) = ord2_cotangent
                        {
                            ord2_cotangent_seq[index] = Some(cotangent.clone());
                        }

                        if ord1_cotangent.is_some() ^ ord2_cotangent.is_some() {
                            return Err(SkoroError::InconsistentCotangents);
                        }

                        op.hvp_forward(&mut point_seq, &mut tangent_seq, &mut tape);
                        out.primal.evaluations.insert(
                            *id,
                            point_seq[index].clone().unwrap(),
                            quantity_type.clone(),
                        );
                        if let Some(tangent) = &tangent_seq[index] {
                            out.tangents.insert(*id, tangent.clone());
                        }
                    }
                }
                _ => {
                    non_op_forward(
                        index,
                        step,
                        input.points,
                        &input.options,
                        &mut point_seq,
                        &mut out.primal,
                        Some(input.tangents),
                        Some(&mut tangent_seq),
                        None,
                        None,
                    )?;
                }
            };
            release_dead_points(&self.steps, &self.last_use, index, &mut point_seq);
        }
        tape.backward(&mut ord1_cotangent_seq, &mut ord2_cotangent_seq);
        for (index, step) in self.steps.iter().enumerate() {
            match step {
                CompiledStep::Parameter { id, .. }
                | CompiledStep::Given { id, .. }
                | CompiledStep::Sampler { id, .. } => {
                    if let Some(cotangent) = ord1_cotangent_seq[index].take() {
                        out.ord1_cotangents.insert(*id, cotangent);
                    }
                    if let Some(cotangent) = ord2_cotangent_seq[index].take() {
                        out.ord2_cotangents.insert(*id, cotangent);
                    }
                }
                _ => {}
            };
        }
        Ok(out)
    }
}

/// Realize a leaf step -- constant, parameter, or sampler -- into the sequence slots.
///
/// The four trailing `Option`s are the seeding channels, and they are separate because the
/// passes carry different ranks: JVP threads rank-3 tangent *frames* (`tensors`), HVP threads a
/// single rank-2 tangent vector per quantity (`vectors`), and a plain forward pass threads
/// neither. Each pass hands in only the pair it has, so a `None` here means "this pass has no
/// such channel", not "nothing was seeded for this node" -- the latter is the inner
/// `get(id)` miss, which leaves the slot empty and lets the operation treat it as zero.
#[inline(always)]
#[allow(clippy::too_many_arguments)]
fn non_op_forward<B: Backend>(
    index: usize,
    step: &CompiledStep<B>,
    points: &HashMap<Id, Input<B>>,
    options: &ForwardOptions,
    point_seq: &mut [Option<Input<B>>],
    out: &mut ForwardOutput<B>,
    vectors: Option<&HashMap<Id, Tensor<B, 2>>>,
    vector_seq: Option<&mut [Option<Tensor<B, 2>>]>,
    tensors: Option<&HashMap<Id, Tensor<B, 3>>>,
    tensor_seq: Option<&mut [Option<Tensor<B, 3>>]>,
) -> SkoroResult<()> {
    match step {
        CompiledStep::Constant {
            id,
            value,
            quantity_type,
        } => {
            let input = value.materialize(options.batch_size)?;
            out.constants
                .insert(*id, input.clone(), quantity_type.clone());
            point_seq[index] = Some(input);
        }
        // A cut boundary reads exactly as a parameter does. It is supplied, it is reported as
        // a variable rather than a constant, and it takes a seed -- the only difference is
        // which graph node it stands for, which the sequence has already forgotten.
        CompiledStep::Parameter { id, quantity_type }
        | CompiledStep::Given { id, quantity_type } => {
            let input = if let Some(parameter) = points.get(id) {
                parameter.ensure_batch(options.batch_size)?
            } else {
                return Err(SkoroError::MissingPoint { id: *id });
            };

            out.variables
                .insert(*id, input.clone(), quantity_type.clone());
            point_seq[index] = Some(input);

            if let (Some(vectors), Some(vector_seq)) = (vectors, vector_seq)
                && let Some(vector) = vectors.get(id)
            {
                vector_seq[index] = Some(vector.clone());
            }
            if let (Some(tensors), Some(tensor_seq)) = (tensors, tensor_seq)
                && let Some(tensor) = tensors.get(id)
            {
                tensor_seq[index] = Some(tensor.clone());
            }
        }
        CompiledStep::Sampler {
            id,
            sampler,
            quantity_type,
        } => {
            if let Some(sample) = points.get(id) {
                let input = sample.ensure_batch(options.batch_size)?;
                out.variables
                    .insert(*id, input.clone(), quantity_type.clone());
                point_seq[index] = Some(input);
            } else if options.use_model_samplers {
                sampler.sample(options.batch_size, point_seq);
                out.variables.insert(
                    *id,
                    point_seq[index].clone().unwrap(),
                    quantity_type.clone(),
                );
            } else {
                return Err(SkoroError::MissingPoint { id: *id });
            };

            if let (Some(vectors), Some(vector_seq)) = (vectors, vector_seq)
                && let Some(vector) = vectors.get(id)
            {
                vector_seq[index] = Some(vector.clone());
            }
            if let (Some(tensors), Some(tensor_seq)) = (tensors, tensor_seq)
                && let Some(tensor) = tensors.get(id)
            {
                tensor_seq[index] = Some(tensor.clone());
            }
        }
        CompiledStep::Operation { .. } => unreachable!(),
    };
    Ok(())
}

#[inline(always)]
#[allow(clippy::borrowed_box)]
#[allow(clippy::too_many_arguments)]
fn forward_only_op<B: Backend>(
    id: Id,
    op: &Box<dyn DynSequenceOperation<B>>,
    value: &Option<QuantityValue<B>>,
    quantity_type: &Arc<dyn DynQuantityType<B>>,
    index: usize,
    options: &ForwardOptions,
    tensor_seq: &mut [Option<Input<B>>],
    out: &mut ForwardOutput<B>,
) -> SkoroResult<()> {
    if let Some(value) = value {
        let input = value.materialize(options.batch_size)?;
        out.evaluations
            .insert(id, input.clone(), quantity_type.clone());
        tensor_seq[index] = Some(input);
    } else {
        op.forward(tensor_seq);
        out.evaluations.insert(
            id,
            tensor_seq[index].clone().unwrap(),
            quantity_type.clone(),
        );
    }
    Ok(())
}

#[inline(always)]
fn check_tensor_frames<B: Backend>(
    frame_count: &mut Option<usize>,
    tensor: Option<&Tensor<B, 3>>,
) -> SkoroResult<()> {
    if let Some(tensor) = tensor {
        let tensor_frame_count = tensor.dims()[2];
        match frame_count {
            None => {
                *frame_count = Some(tensor_frame_count);
            }
            Some(frame_count) if *frame_count != tensor_frame_count => {
                return Err(SkoroError::InconsistentFrameCount {
                    expected: *frame_count,
                    found: tensor_frame_count,
                });
            }
            _ => {}
        }
    }
    Ok(())
}
