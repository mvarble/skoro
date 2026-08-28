//! Shared fixtures and the numerical-derivative harness for `skoro-operation`.
//!
//! This is the same forward-difference convention as `skoro_graph::test_utils` and
//! `skoro-model`'s `test_utils`, one level further down. The difference is what it can reach:
//! `skoro-operation` sits *below* `skoro-riemann`, so there is no `Manifold` here to retract
//! along. Perturbation is plain `x + δ dᵢ` and comparison plain `f(x') - f(x)`, which is exactly
//! what retraction and inverse-retraction reduce to on a flat manifold -- so a result checked
//! here means the same thing as one checked upstream.
//!
//! Forward, not central: the truncation error is `O(δ)`, so `EPSILON` is deliberately loose
//! against `DELTA`. A failure by a small multiple of `DELTA` is the method; a failure by a
//! *factor* is a dropped term.

#![allow(dead_code)]

use skoro_backend::{NdArrayBackend, Tensor, TensorData};
use skoro_operation::{
    Input,
    diff::{DiffModeState, DiffOperation, DiffOperationForward},
};

pub type B = NdArrayBackend<f64, i32>;

pub const DELTA: f64 = 1e-6;
pub const EPSILON: f64 = 1e-5;

// ---------------------------------------------------------------------------------------------
// Tensor helpers
// ---------------------------------------------------------------------------------------------

pub fn tensor2(data: Vec<f64>, batch: usize, components: usize) -> Tensor<B, 2> {
    Tensor::from_data(
        TensorData::new(data, [batch, components]),
        &Default::default(),
    )
}

pub fn tensor3(data: Vec<f64>, dims: [usize; 3]) -> Tensor<B, 3> {
    Tensor::from_data(TensorData::new(data, dims), &Default::default())
}

/// A batch-1 row vector.
pub fn row(data: Vec<f64>) -> Tensor<B, 2> {
    let n = data.len();
    tensor2(data, 1, n)
}

/// A batch-1 scalar.
pub fn scalar(v: f64) -> Tensor<B, 2> {
    tensor2(vec![v], 1, 1)
}

pub fn into_vec<const D: usize>(t: Tensor<B, D>) -> Vec<f64> {
    t.into_data().to_vec::<f64>().unwrap()
}

/// A `[batch, dim, dim]` identity.
pub fn identity(dim: usize, batch: usize) -> Tensor<B, 3> {
    let mut data = vec![0.0; dim * dim];
    for i in 0..dim {
        data[i * (dim + 1)] = 1.0;
    }
    tensor3(data, [1, dim, dim]).repeat_dim(0, batch)
}

/// The `i`-th basis direction, shaped `[batch, dim]`.
pub fn basis(dim: usize, i: usize, batch: usize) -> Tensor<B, 2> {
    let mut data = vec![0.0; dim];
    data[i] = 1.0;
    tensor2(data, 1, dim).repeat_dim(0, batch)
}

// ---------------------------------------------------------------------------------------------
// Mode accessors
// ---------------------------------------------------------------------------------------------
//
// Each of these asserts the contract every `DiffOperation` owes: the variant it returns must
// match the variant it was given. Routing that assertion through one place is what keeps a
// mistyped `else` branch from reading as a shape error twenty lines later.

pub fn forward_only<Op: DiffOperation<B>>(op: &Op, inputs: Vec<Input<B>>) -> Tensor<B, 2> {
    match op.forward(inputs, DiffModeState::ForwardOnly) {
        DiffOperationForward::ForwardOnly { output } => output,
        _ => panic!("ForwardOnly mode must return the ForwardOnly variant"),
    }
}

pub fn jvp<Op: DiffOperation<B>>(
    op: &Op,
    inputs: Vec<Input<B>>,
    tangents: Vec<Option<Tensor<B, 3>>>,
) -> (Tensor<B, 2>, Option<Tensor<B, 3>>) {
    match op.forward(inputs, DiffModeState::Jvp(tangents)) {
        DiffOperationForward::Jvp { output, tangent } => (output, tangent),
        _ => panic!("Jvp mode must return the Jvp variant"),
    }
}

pub fn vjp_state<Op: DiffOperation<B>>(
    op: &Op,
    inputs: Vec<Input<B>>,
    diffs: Vec<bool>,
) -> (Tensor<B, 2>, Op::VjpState) {
    match op.forward(inputs, DiffModeState::Vjp(diffs)) {
        DiffOperationForward::Vjp { output, vjp_state } => (output, vjp_state),
        _ => panic!("Vjp mode must return the Vjp variant"),
    }
}

pub type HvpParts<Op> = (
    Tensor<B, 2>,
    Option<Tensor<B, 2>>,
    <Op as DiffOperation<B>>::VjpState,
    <Op as DiffOperation<B>>::HvpState,
);

pub fn hvp_state<Op: DiffOperation<B>>(
    op: &Op,
    inputs: Vec<Input<B>>,
    diffs: Vec<bool>,
    tangents: Vec<Option<Tensor<B, 2>>>,
) -> HvpParts<Op> {
    match op.forward(inputs, DiffModeState::Hvp { diffs, tangents }) {
        DiffOperationForward::Hvp {
            output,
            tangent,
            vjp_state,
            hvp_state,
        } => (output, tangent, vjp_state, hvp_state),
        _ => panic!("Hvp mode must return the Hvp variant"),
    }
}

/// Wrap float tensors as operation inputs.
pub fn floats(tensors: &[Tensor<B, 2>]) -> Vec<Input<B>> {
    tensors.iter().cloned().map(Input::Float).collect()
}

/// `inputs`, with input `wrt` stepped by `delta` along basis direction `i`.
fn stepped(inputs: &[Tensor<B, 2>], wrt: usize, i: usize, delta: f64) -> Vec<Tensor<B, 2>> {
    let mut out = inputs.to_vec();
    let [batch, dim] = out[wrt].dims();
    out[wrt] = out[wrt].clone() + basis(dim, i, batch).mul_scalar(delta);
    out
}

// ---------------------------------------------------------------------------------------------
// Numerical checks
// ---------------------------------------------------------------------------------------------

/// Check the pushforward with respect to input `wrt` against a forward difference.
///
/// Every input is declared differentiable, but only `wrt` is seeded, so the reported tangent is
/// the column block belonging to that input alone.
pub fn test_jvp_numerically<Op: DiffOperation<B>>(
    op: &Op,
    inputs: &[Tensor<B, 2>],
    wrt: usize,
    delta: f64,
    epsilon: f64,
) {
    let [batch, indim] = inputs[wrt].dims();
    let mut seeds: Vec<Option<Tensor<B, 3>>> = vec![None; inputs.len()];
    seeds[wrt] = Some(identity(indim, batch));

    let (y0, tangent) = jvp(op, floats(inputs), seeds);
    let tangent = tangent.expect("a seeded differentiable input must produce a tangent");
    let outdim = y0.dims()[1];
    assert_eq!(tangent.dims(), [batch, outdim, indim], "JVP frame shape");

    for i in 0..indim {
        let yi = forward_only(op, floats(&stepped(inputs, wrt, i, delta)));
        let expected = into_vec((yi - y0.clone()).div_scalar(delta));
        let actual = into_vec(tangent.clone().slice_dim(2, i).squeeze_dim::<2>(2));
        approx::assert_abs_diff_eq!(actual.as_slice(), expected.as_slice(), epsilon = epsilon);
    }
}

/// Check the pullback into input `wrt` against a forward difference.
///
/// Seeds the identity cotangent frame on the output, so row `i` of the reported cotangent is the
/// same quantity `test_jvp_numerically` reports as column `i`. A transposed pullback passes one
/// of these and fails the other.
pub fn test_vjp_numerically<Op: DiffOperation<B>>(
    op: &Op,
    inputs: &[Tensor<B, 2>],
    wrt: usize,
    delta: f64,
    epsilon: f64,
) {
    let diffs = vec![true; inputs.len()];
    test_vjp_numerically_gated(op, inputs, wrt, diffs, delta, epsilon)
}

/// The same check with the differentiability flags chosen by the caller.
///
/// An operation whose `VjpState` splits by gate computes each slot through a *different* arm, so
/// the all-differentiable check above only ever reads one of them. `wrt` must still be gated on,
/// since a slot that is not differentiable receives no cotangent to compare.
pub fn test_vjp_numerically_gated<Op: DiffOperation<B>>(
    op: &Op,
    inputs: &[Tensor<B, 2>],
    wrt: usize,
    diffs: Vec<bool>,
    delta: f64,
    epsilon: f64,
) {
    assert!(diffs[wrt], "the slot under test must be differentiable");
    let [batch, indim] = inputs[wrt].dims();
    let (y0, state) = vjp_state(op, floats(inputs), diffs);
    let outdim = y0.dims()[1];

    let cotangents = op.vjp(&state, identity(outdim, batch));
    let block = cotangents[wrt]
        .clone()
        .expect("a differentiable input must receive a cotangent");
    assert_eq!(block.dims(), [batch, indim, outdim], "VJP frame shape");

    for i in 0..indim {
        let yi = forward_only(op, floats(&stepped(inputs, wrt, i, delta)));
        let expected = into_vec((yi - y0.clone()).div_scalar(delta));
        let actual = into_vec(block.clone().slice_dim(1, i));
        approx::assert_abs_diff_eq!(actual.as_slice(), expected.as_slice(), epsilon = epsilon);
    }
}

/// Check the mixed second-order block `∂²F / ∂read ∂seed` against a forward difference of the VJP.
///
/// `hvp` is only the sensitivity term `∇_v dF*_p(ω)` -- the Leibniz sum with `vjp(β)` is the
/// tape's job, not the operation's -- so the reference is exactly
/// `(vjp(x + δv) - vjp(x)) / δ`, with no second term to subtract off.
///
/// Passing `seed == read` recovers the diagonal block.
pub fn test_hvp_numerically<Op: DiffOperation<B>>(
    op: &Op,
    inputs: &[Tensor<B, 2>],
    seed: usize,
    read: usize,
    delta: f64,
    epsilon: f64,
) {
    let [batch, seeddim] = inputs[seed].dims();
    let readdim = inputs[read].dims()[1];
    let diffs = vec![true; inputs.len()];
    let outdim = forward_only(op, floats(inputs)).dims()[1];
    let omega = identity(outdim, batch);

    for i in 0..seeddim {
        let mut tangents: Vec<Option<Tensor<B, 2>>> = vec![None; inputs.len()];
        tangents[seed] = Some(basis(seeddim, i, batch));

        let (_, _, vjp_st, hvp_st) = hvp_state(op, floats(inputs), diffs.clone(), tangents);
        // An absent slot means "no contribution", which is a legitimate answer whenever the seed
        // cannot move that input -- so it is read as zero and still compared. Treating `None` as
        // "skip" instead would let a genuinely dropped term through unnoticed.
        let exact = match &op.hvp(&vjp_st, &hvp_st, omega.clone())[read] {
            Some(block) => {
                assert_eq!(block.dims(), [batch, readdim, outdim], "HVP frame shape");
                into_vec(block.clone())
            }
            None => vec![0.0; batch * readdim * outdim],
        };

        // The first-order pullback at the base point and one step along the seed direction.
        let (_, base_state) = vjp_state(op, floats(inputs), diffs.clone());
        let vjp0 = op.vjp(&base_state, omega.clone())[read].clone().unwrap();

        let moved = stepped(inputs, seed, i, delta);
        let (_, moved_state) = vjp_state(op, floats(&moved), diffs.clone());
        let vjp1 = op.vjp(&moved_state, omega.clone())[read].clone().unwrap();

        let expected = into_vec((vjp1 - vjp0).div_scalar(delta));
        approx::assert_abs_diff_eq!(exact.as_slice(), expected.as_slice(), epsilon = epsilon);
    }
}

/// Run all three checks for every input of a single-output operation.
pub fn check_all_derivatives<Op: DiffOperation<B>>(op: &Op, inputs: &[Tensor<B, 2>]) {
    for wrt in 0..inputs.len() {
        test_jvp_numerically(op, inputs, wrt, DELTA, EPSILON);
        test_vjp_numerically(op, inputs, wrt, DELTA, EPSILON);
        for read in 0..inputs.len() {
            test_hvp_numerically(op, inputs, wrt, read, DELTA, EPSILON);
        }
    }
}
