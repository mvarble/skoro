use std::fmt::Debug;

use skoro_backend::Backend;
use skoro_operation::{
    DynSequenceOperation,
    diff::{DiffOperation, SequencedDiffOperation},
    discrete::{DiscreteOperation, SequencedDiscreteOperation},
};

/// A deferred graph operation, lowerable into a concrete [`DynSequenceOperation`].
///
/// Both differentiable operations ([`DiffOperation`]) and forward-only discrete
/// operations ([`DiscreteOperation`], via the [`ForwardOnly`] adapter) lower into the
/// same dyn sequence-operation interface, since the value sequence is
/// kind-erased ([`Input`](skoro_operation::Input)).
pub trait GraphOperation<B: Backend>: 'static + Send + Sync + Debug {
    /// Lowers this deferred operation into a concrete [`DynSequenceOperation`].
    fn lower(
        &self,
        input_indices: Vec<usize>,
        input_diffs: Vec<bool>,
        output_index: usize,
    ) -> Box<dyn DynSequenceOperation<B>>;

    /// Whether this operation propagates derivatives.
    ///
    /// Forward-only operations (discrete outputs) return `false`; the compiler
    /// then evaluates them without differentiation even in JVP/VJP/HVP passes,
    /// and treats all of their inputs as non-differentiable.
    fn is_differentiable(&self) -> bool {
        true
    }

    /// Returns a new boxed copy of this operation.
    fn clone_box(&self) -> Box<dyn GraphOperation<B>>;
}

impl<B: Backend> Clone for Box<dyn GraphOperation<B>> {
    fn clone(&self) -> Self {
        self.clone_box()
    }
}

impl<B: Backend, Op: DiffOperation<B>> GraphOperation<B> for Op {
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
            self.clone(),
        ))
    }

    fn clone_box(&self) -> Box<dyn GraphOperation<B>> {
        Box::new(self.clone())
    }
}

/// Adapter presenting a forward-only [`DiscreteOperation`] as a [`GraphOperation`].
///
/// The wrapped operation is non-differentiable: it lowers into a
/// [`SequenceForwardOnlyOperation`] and reports
/// [`is_differentiable`](GraphOperation::is_differentiable) as `false`.
pub struct ForwardOnly<K>(pub K);

impl<K: Debug> Debug for ForwardOnly<K> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_tuple("ForwardOnly").field(&self.0).finish()
    }
}

impl<B: Backend, K: DiscreteOperation<B>> GraphOperation<B> for ForwardOnly<K> {
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

    fn clone_box(&self) -> Box<dyn GraphOperation<B>> {
        Box::new(ForwardOnly(self.0.clone()))
    }
}
