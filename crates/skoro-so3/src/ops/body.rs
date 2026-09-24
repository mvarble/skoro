//! Shared arithmetic for the group operations' derivatives in the body-frame chart
//! `r ⊕ a = r Exp(a)`.
//!
//! Tensors are `[batch, 3, m]`: axis 1 is the component axis and axis 2 runs over columns, which
//! may be the columns of a matrix operand, the columns of a frame, or both, flattened. Everything
//! here is a plain tensor operation, so the group operations built on it serve every backend
//! from one implementation. `skoro-se3` builds the pose operations on the same arithmetic.

use skoro_backend::{Backend, Point, Tensor};

/// `x × y` column by column, along axis 1.
pub fn cross<B: Backend>(x: Tensor<B, 3>, y: Tensor<B, 3>) -> Tensor<B, 3> {
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
pub fn matrix<B: Backend>(r: &Point<B>) -> Tensor<B, 3> {
    let [batch, _] = r.dims();
    r.clone().reshape([batch, 3, 3])
}

/// `r x`.
pub fn apply<B: Backend>(r: &Point<B>, x: Tensor<B, 3>) -> Tensor<B, 3> {
    matrix(r).matmul(x)
}

/// `rᵀ x`.
pub fn apply_transpose<B: Backend>(r: &Point<B>, x: Tensor<B, 3>) -> Tensor<B, 3> {
    matrix(r).swap_dims(1, 2).matmul(x)
}

/// A `[batch, 3]` vector repeated as `m` columns: `[batch, 3, m]`.
pub fn repeat<B: Backend>(v: &Tensor<B, 2>, m: usize) -> Tensor<B, 3> {
    v.clone().unsqueeze_dim::<3>(2).repeat_dim(2, m)
}

/// `a + b` where either may be absent, `None` only when both are.
pub fn sum<B: Backend>(a: Option<Tensor<B, 3>>, b: Option<Tensor<B, 3>>) -> Option<Tensor<B, 3>> {
    match (a, b) {
        (Some(a), Some(b)) => Some(a + b),
        (a, None) => a,
        (None, b) => b,
    }
}

// A matrix operand `V ∈ ℝ^{3×n}` (a vector is `n = 1`) is stored row-major as `[batch, 3n]`, and
// its tangent and cotangent frames as `[batch, 3n, f]`. Derivatives work on `[batch, 3, n·f]`,
// column `c` of the operand and column `j` of the frame at `c·f + j`, so one cross product or one
// matmul covers every pair.

/// `V` as `[batch, 3, n·f]`, each operand column repeated across the `f` frame columns.
pub fn spread<B: Backend>(v: &Tensor<B, 2>, n: usize, f: usize) -> Tensor<B, 3> {
    let [batch, _] = v.dims();
    v.clone()
        .reshape([batch, 3, n])
        .unsqueeze_dim::<4>(3)
        .repeat_dim(3, f)
        .reshape([batch, 3, n * f])
}

/// A frame `[batch, 3, f]` as `[batch, 3, n·f]`, repeated across the `n` operand columns.
pub fn tile<B: Backend>(a: Tensor<B, 3>, n: usize) -> Tensor<B, 3> {
    let [batch, _, f] = a.dims();
    a.unsqueeze_dim::<4>(2)
        .repeat_dim(2, n)
        .reshape([batch, 3, n * f])
}

/// An operand frame `[batch, 3n, f]` as `[batch, 3, n·f]`.
pub fn unstack<B: Backend>(x: Tensor<B, 3>, n: usize) -> Tensor<B, 3> {
    let [batch, _, f] = x.dims();
    x.reshape([batch, 3, n * f])
}

/// `[batch, 3, n·f]` back to an operand frame `[batch, 3n, f]`.
pub fn stack<B: Backend>(x: Tensor<B, 3>, n: usize) -> Tensor<B, 3> {
    let [batch, _, nf] = x.dims();
    x.reshape([batch, 3 * n, nf / n])
}

/// The sum over the `n` operand columns: `[batch, 3, n·f]` to a rotation frame `[batch, 3, f]`.
pub fn fold<B: Backend>(x: Tensor<B, 3>, n: usize) -> Tensor<B, 3> {
    let [batch, _, nf] = x.dims();
    let f = nf / n;
    x.reshape([batch, 3, n, f])
        .sum_dim(2)
        .reshape([batch, 3, f])
}
