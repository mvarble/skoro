use std::fmt::Debug;

use skoro_backend::Backend;
use skoro_operation::{
    DynSequenceOperation,
    diff::{DiffOperation, SequencedDiffOperation},
    discrete::{DiscreteOperation, SequencedDiscreteOperation},
};

/// A deferred graph operation, lowerable into a concrete [`DynSequenceOperation`].
///
/// Both differentiable operations ([`DiffOperation`]) and forward-only discrete operations
/// [`DiscreteOperation`] implement this via adapters.
pub trait DynGraphOperation<B: Backend>: 'static + Send + Sync + Debug {
    /// Lowers this deferred operation into a concrete [`DynSequenceOperation`].
    fn lower(
        &self,
        input_indices: Vec<usize>,
        input_diffs: Vec<bool>,
        output_index: usize,
    ) -> Box<dyn DynSequenceOperation<B>>;

    /// Whether this operation propagates derivatives (i.e. returns continuous tensors).
    fn is_differentiable(&self) -> bool;

    /// Returns a new boxed copy of this operation.
    fn clone_box(&self) -> Box<dyn DynGraphOperation<B>>;
}

impl<B: Backend> Clone for Box<dyn DynGraphOperation<B>> {
    fn clone(&self) -> Self {
        self.clone_box()
    }
}

/// Adapter which connects the [`DiffOperation`] and [`DynGraphOperation`] traits.
pub struct DiffGraphOperation<Op>(pub Op);

impl<Op: Debug> Debug for DiffGraphOperation<Op> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_tuple("DiffGraphOperation").field(&self.0).finish()
    }
}

impl<B: Backend, Op: DiffOperation<B>> DynGraphOperation<B> for DiffGraphOperation<Op> {
    fn lower(
        &self,
        input_indices: Vec<usize>,
        input_diffs: Vec<bool>,
        output_index: usize,
    ) -> Box<dyn DynSequenceOperation<B>> {
        Box::new(SequencedDiffOperation::new(
            input_indices,
            input_diffs,
            output_index,
            self.0.clone(),
        ))
    }

    fn is_differentiable(&self) -> bool {
        true
    }

    fn clone_box(&self) -> Box<dyn DynGraphOperation<B>> {
        Box::new(DiffGraphOperation(self.0.clone()))
    }
}

/// Adapter which connects the [`DiscreteOperation`] and [`DynGraphOperation`] traits.
pub struct DiscreteGraphOperation<Op>(pub Op);

impl<Op: Debug> Debug for DiscreteGraphOperation<Op> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_tuple("DiscreteGraphOperation")
            .field(&self.0)
            .finish()
    }
}

impl<B: Backend, Op: DiscreteOperation<B>> DynGraphOperation<B> for DiscreteGraphOperation<Op> {
    fn lower(
        &self,
        input_indices: Vec<usize>,
        _input_diffs: Vec<bool>,
        output_index: usize,
    ) -> Box<dyn DynSequenceOperation<B>> {
        Box::new(SequencedDiscreteOperation::new(
            input_indices,
            output_index,
            self.0.clone(),
        ))
    }

    fn is_differentiable(&self) -> bool {
        false
    }

    fn clone_box(&self) -> Box<dyn DynGraphOperation<B>> {
        Box::new(DiscreteGraphOperation(self.0.clone()))
    }
}
