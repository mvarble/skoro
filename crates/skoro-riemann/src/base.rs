use skoro_backend::{Backend, Point, TangentVector};
use skoro_operation::Input;
use skoro_operation::diff::{DiffModeState, DiffOperation, DiffOperationForward};

/// A trait which implements the manifold operations associated with some type.
///
/// This struct should be light-weight, as it is frequently cloned during storage operations.
/// Should it hold any state, it might be useful to use references.
pub trait Manifold<B: Backend> {
    /// Dimension of the manifold.
    fn dimension(&self) -> usize;

    /// Operation for retracting along a tangent vector from a point on the manifold.
    /// This method returns an operation `R` where the tensor shapes associated with
    /// `output = R(point, tangent)` are as follows.
    ///
    /// - `point`: `[batch, storage-size]`
    /// - `tangent`: `[batch, dimension]`
    /// - `output`: `[batch, storage-size]`
    fn retraction(&self) -> impl DiffOperation<B>;

    /// Operation for obtaining the tangent vector along which we retract to get between two points
    /// on the manifold. This method returns an operation `L` where the tensor shapes associated
    /// with `output = L(point, other_point)` are as follows.
    ///
    /// - `point`: `[batch, storage-size]`
    /// - `other_point`: `[batch, storage-size]`
    /// - `output`: `[batch, dimension]`
    fn inverse_retraction(&self) -> impl DiffOperation<B>;
}

mod private {
    pub trait Sealed<B: super::Backend> {}
}

/// The dyn-compatible analogue of [`Manifold`]. It must not be implemented for any specific type,
/// as there is a blanket implementation.
pub trait DynManifold<B: Backend>: private::Sealed<B> {
    /// Dimension of the manifold.
    fn dimension(&self) -> usize;

    /// Perform forward-only retraction.
    fn retract(&self, point: Point<B>, tangent: TangentVector<B>) -> Point<B>;

    /// Perform forward-only inverse-retraction.
    fn inverse_retract(&self, point: Point<B>, other_point: Point<B>) -> TangentVector<B>;
}

impl<B: Backend, M: Manifold<B>> private::Sealed<B> for M {}

impl<B: Backend, M: Manifold<B>> DynManifold<B> for M {
    #[inline(always)]
    fn dimension(&self) -> usize {
        <M as Manifold<B>>::dimension(self)
    }

    fn retract(&self, point: Point<B>, tangent: TangentVector<B>) -> Point<B> {
        match self.retraction().forward(
            vec![Input::Float(point), Input::Float(tangent)],
            DiffModeState::ForwardOnly,
        ) {
            DiffOperationForward::ForwardOnly { output } => output,
            _ => unreachable!(
                "`DiffOperation` must pass `DiffModeState::ForwardOnly` to `DiffOperationForward::ForwardOnly`."
            ),
        }
    }

    fn inverse_retract(&self, point: Point<B>, other_point: Point<B>) -> TangentVector<B> {
        match self.inverse_retraction().forward(
            vec![Input::Float(point), Input::Float(other_point)],
            DiffModeState::ForwardOnly,
        ) {
            DiffOperationForward::ForwardOnly { output } => output,
            _ => unreachable!(
                "`DiffOperation` must pass `DiffModeState::ForwardOnly` to `DiffOperationForward::ForwardOnly`."
            ),
        }
    }
}
