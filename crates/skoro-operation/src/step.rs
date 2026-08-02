use skoro_backend::{Backend, CotangentFrame};

/// A single backward step, consumed during first-order tape traversal.
///
/// Do not implement this trait; it is handled by the wrapper struct [`VjpStep`](crate::diff::VjpStep).
pub trait DynVjpStep<B: Backend>: Send + Sync + 'static {
    /// Apply the local pullback `ω ↦ dF_p*(ω)`, consuming cotangent vectors in the codomain and
    /// inserting those in the domain.
    fn step(&self, cotangents: &mut [Option<CotangentFrame<B>>]);

    /// Get the input indices for this operation.
    fn input_indices(&self) -> &[usize];

    /// Get the output index.
    fn output_index(&self) -> usize;
}

/// A single backward step, consumed during first-order tape traversal.
///
/// Do not implement this trait; it is handled by the wrapper struct [`HvpStep`](crate::diff::HvpStep).
pub trait DynHvpStep<B: Backend>: Send + Sync + 'static {
    /// Apply the local pullback `ω ↦ dF_p*(ω)` to first- and second- order cotangents and the
    /// sensitivity `ω ↦ ∇_v dF_p^*(ω)` to the first- order cotangent, consuming
    /// these cotangent vectors in the codomain and inserting those in the domain.
    fn step(
        &self,
        ord1_cotangents: &mut [Option<CotangentFrame<B>>],
        ord2_cotangents: &mut [Option<CotangentFrame<B>>],
    );

    /// Get the input indices for this operation.
    fn input_indices(&self) -> &[usize];

    /// Get the output index.
    fn output_index(&self) -> usize;
}

pub type BoxDynVjpStep<B> = Box<dyn DynVjpStep<B>>;
pub type BoxDynHvpStep<B> = Box<dyn DynHvpStep<B>>;
