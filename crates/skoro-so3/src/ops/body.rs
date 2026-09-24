//! Shared arithmetic for the group operations' derivatives in the body-frame chart
//! `r ⊕ a = r Exp(a)`.
//!
//! Tensors are `[batch, 3, m]`: axis 1 is the component axis and axis 2 runs over columns, which
//! may be the columns of a matrix operand, the columns of a frame, or both, flattened. Everything
//! here is a plain tensor operation, so the group operations built on it serve every backend
//! from one implementation.

use skoro_backend::{Backend, Point, Tensor};

/// `x × y` column by column, along axis 1.
pub(crate) fn cross<B: Backend>(x: Tensor<B, 3>, y: Tensor<B, 3>) -> Tensor<B, 3> {
    let part = |t: &Tensor<B, 3>, i: usize| t.clone().narrow(1, i, 1);
    let (x0, x1, x2) = (part(&x, 0), part(&x, 1), part(&x, 2));
    let (y0, y1, y2) = (part(&y, 0), part(&y, 1), part(&y, 2));
    Tensor::cat(
        vec![
            x1.clone() * y2.clone() - x2.clone() * y1.clone(),
            x2 * y0.clone() - x0.clone() * y2,
            x0 * y1 - x1 * y0,
        ],
        1,
    )
}

/// The stored rotation `[batch, 9]` as `[batch, 3, 3]`.
pub(crate) fn matrix<B: Backend>(r: &Point<B>) -> Tensor<B, 3> {
    let [batch, _] = r.dims();
    r.clone().reshape([batch, 3, 3])
}

/// `r x`.
pub(crate) fn apply<B: Backend>(r: &Point<B>, x: Tensor<B, 3>) -> Tensor<B, 3> {
    matrix(r).matmul(x)
}

/// `rᵀ x`.
pub(crate) fn apply_transpose<B: Backend>(r: &Point<B>, x: Tensor<B, 3>) -> Tensor<B, 3> {
    matrix(r).swap_dims(1, 2).matmul(x)
}

/// A `[batch, 3]` vector repeated as `m` columns: `[batch, 3, m]`.
pub(crate) fn repeat<B: Backend>(v: &Tensor<B, 2>, m: usize) -> Tensor<B, 3> {
    v.clone().unsqueeze_dim::<3>(2).repeat_dim(2, m)
}

/// `a + b` where either may be absent, `None` only when both are.
pub(crate) fn sum<B: Backend>(
    a: Option<Tensor<B, 3>>,
    b: Option<Tensor<B, 3>>,
) -> Option<Tensor<B, 3>> {
    match (a, b) {
        (Some(a), Some(b)) => Some(a + b),
        (a, None) => a,
        (None, b) => b,
    }
}
