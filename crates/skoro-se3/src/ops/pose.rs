//! Splitting a pose into its rotation and translation, and its tangents into their blocks.
//!
//! A pose is stored `[batch, 12]`: the rotation's 9 numbers, then the translation's 3. Its tangent
//! is `(a, b)`, six coordinates with the rotation block first, in the body-frame chart
//! `(r, p) ⊕ (a, b) = (r Exp(a), p + r b)`.

use skoro_backend::{Backend, Point, Tensor};

/// `(r, p)` from a stored pose.
pub(crate) fn split_pose<B: Backend>(x: Point<B>) -> (Point<B>, Tensor<B, 2>) {
    (x.clone().narrow(1, 0, 9), x.narrow(1, 9, 3))
}

/// A stored pose from `(r, p)`.
pub(crate) fn join_pose<B: Backend>(r: Point<B>, p: Tensor<B, 2>) -> Point<B> {
    Tensor::cat(vec![r, p], 1)
}

/// The rotation and translation blocks of a pose tangent or cotangent frame `[batch, 6, f]`.
pub(crate) fn split_frame<B: Backend>(t: Tensor<B, 3>) -> (Tensor<B, 3>, Tensor<B, 3>) {
    (t.clone().narrow(1, 0, 3), t.narrow(1, 3, 3))
}

/// The same for a tangent vector `[batch, 6]`.
pub(crate) fn split_vector<B: Backend>(t: Tensor<B, 2>) -> (Tensor<B, 2>, Tensor<B, 2>) {
    (t.clone().narrow(1, 0, 3), t.narrow(1, 3, 3))
}

/// A pose frame from its two blocks, a missing one read as zero; `None` only when both are.
pub(crate) fn join_frame<B: Backend>(
    a: Option<Tensor<B, 3>>,
    b: Option<Tensor<B, 3>>,
) -> Option<Tensor<B, 3>> {
    match (a, b) {
        (None, None) => None,
        (Some(a), Some(b)) => Some(Tensor::cat(vec![a, b], 1)),
        (Some(a), None) => {
            let zeros = a.zeros_like();
            Some(Tensor::cat(vec![a, zeros], 1))
        }
        (None, Some(b)) => {
            let zeros = b.zeros_like();
            Some(Tensor::cat(vec![zeros, b], 1))
        }
    }
}

/// A tangent vector `[batch, d]` as a one-column frame.
pub(crate) fn one_column<B: Backend>(v: Tensor<B, 2>) -> Tensor<B, 3> {
    v.unsqueeze_dim(2)
}
