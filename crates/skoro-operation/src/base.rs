use skoro_backend::{Backend, TangentFrame, TangentVector, Tensor};
use skoro_core::sequence::SequenceMutator;

pub use skoro_core::kind_erased_tensor::{KindErasedTensor, KindErasedTensorMap};

use crate::{
    step::{BoxDynHvpStep, BoxDynVjpStep},
    tape::Tape,
};

pub type Input<B> = KindErasedTensor<B, 2>;
pub type ValueMap<B> = KindErasedTensorMap<B, 2>;

/// Operation which mutates an output value array and pushes backward operations onto tape for
/// backpropagation.
///
/// This can be seen as the dyn-compatible analog of [`DiffOperation`](crate::diff::DiffOperation)
/// which is aware of the states where relevant objects are held. The value sequence holds [`Input`]
/// instances so that discrete (integer) values can flow through it alongside continuous ones.
/// [`DiscreteOperation`](crate::discrete::DiscreteOperation) also opaquely follows this interface,
/// though the differentiable operations are non-operational.
pub trait DynSequenceOperation<B: Backend>: SequenceMutator {
    fn input_diffs(&self) -> &[bool];
    fn forward(&self, points: &mut [Option<Input<B>>]);
    fn jvp_forward(
        &self,
        points: &mut [Option<Input<B>>],
        tangents: &mut [Option<TangentFrame<B>>],
    );
    fn vjp_forward(&self, points: &mut [Option<Input<B>>], tape: &mut Tape<BoxDynVjpStep<B>>);
    fn hvp_forward(
        &self,
        points: &mut [Option<Input<B>>],
        tangents: &mut [Option<TangentVector<B>>],
        tape: &mut Tape<BoxDynHvpStep<B>>,
    );
}

#[inline]
pub fn sum_opt_tensors<I, B, const D: usize>(opt_tensors: I) -> Option<Tensor<B, D>>
where
    B: Backend,
    I: IntoIterator<Item = Option<Tensor<B, D>>>,
{
    opt_tensors
        .into_iter()
        .fold(None, |tangent, df| match (tangent, df) {
            (Some(tangent), Some(df)) => Some(tangent + df),
            (Some(tangent), None) => Some(tangent),
            (None, Some(df)) => Some(df),
            (None, None) => None,
        })
}
