use std::{collections::HashMap, marker::PhantomData, sync::Arc};

use skoro_backend::{Backend, CotangentFrame, TangentFrame, TangentVector, Tensor};
use skoro_core::{
    constant_value::ConstantValue,
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
        value: ConstantValue<B>,
        quantity_type: Arc<dyn DynQuantityType<B>>,
    },
    Parameter {
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
        value: Option<ConstantValue<B>>,
        quantity_type: Arc<dyn DynQuantityType<B>>,
        /// If true, the operation is non-differentiable (e.g. a discrete
        /// kernel): it is evaluated forward-only in every pass, and its inputs
        /// receive no tangents/cotangents.
        forward_only: bool,
        constant_inputs: bool,
    },
}

pub struct CompiledSequence<B: Backend> {
    pub steps: Vec<CompiledStep<B>>,
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

#[derive(Debug, Clone, Default)]
pub struct ForwardOutput<B: Backend> {
    pub variables: PointMap<B>,
    pub constants: PointMap<B>,
    pub evaluations: PointMap<B>,
    pub observations: PointMap<B>,
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

impl<B: Backend> CompiledSequence<B> {
    pub fn new(steps: Vec<CompiledStep<B>>) -> Self {
        Self {
            steps,
            _phantom: PhantomData,
        }
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
                    );
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
                    forward_only,
                    ..
                } => {
                    if *forward_only {
                        forward_only_op(
                            *id,
                            op,
                            value,
                            quantity_type,
                            index,
                            &input.options,
                            &mut point_seq,
                            &mut out.primal,
                        );
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
        }
        Ok(out)
    }

    pub fn vjp_forward(&self, input: VjpForwardInput<'_, B>) -> SkoroResult<VjpForwardOutput<B>> {
        let steps_len = self.steps.len();
        let mut point_seq: Vec<Option<Input<B>>> = vec![None; steps_len];
        let mut cotangent_seq: Vec<Option<CotangentFrame<B>>> = vec![None; steps_len];
        let mut out = VjpForwardOutput::default();
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
                        );
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
        }
        tape.backward(&mut cotangent_seq);
        for (index, step) in self.steps.iter().enumerate() {
            match step {
                CompiledStep::Parameter { id, quantity_type }
                | CompiledStep::Sampler {
                    id, quantity_type, ..
                } => {
                    // Discrete variables receive no cotangents by construction.
                    if quantity_type.kind() == QuantityKind::Continuous {
                        if let Some(cotangent) = cotangent_seq[index].take() {
                            out.cotangents.insert(*id, cotangent);
                        } else {
                            panic!("Point should have a cotangent vector.");
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
                        );
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
        }
        tape.backward(&mut ord1_cotangent_seq, &mut ord2_cotangent_seq);
        for (index, step) in self.steps.iter().enumerate() {
            match step {
                CompiledStep::Parameter { id, .. } | CompiledStep::Sampler { id, .. } => {
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

fn ensure_batch_size<B: Backend>(
    id: Id,
    point: &Input<B>,
    batch_size: usize,
) -> SkoroResult<Input<B>> {
    let found = point.dims()[0];
    if found == batch_size {
        Ok(point.clone())
    } else {
        Err(SkoroError::BatchSizeMismatch {
            id,
            expected: batch_size,
            found,
        })
    }
}

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
            let input = value.expand(options.batch_size);
            out.constants
                .insert(*id, input.clone(), quantity_type.clone());
            point_seq[index] = Some(input);
        }
        CompiledStep::Parameter { id, quantity_type } => {
            let input = if let Some(parameter) = points.get(id) {
                ensure_batch_size(*id, parameter, options.batch_size)?
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
                let input = ensure_batch_size(*id, sample, options.batch_size)?;
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
    value: &Option<ConstantValue<B>>,
    quantity_type: &Arc<dyn DynQuantityType<B>>,
    index: usize,
    options: &ForwardOptions,
    tensor_seq: &mut [Option<Input<B>>],
    out: &mut ForwardOutput<B>,
) {
    if let Some(value) = value {
        let input = value.expand(options.batch_size);
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
