//! Shared fixtures and the numerical-derivative harness for `skoro-riemann`.
//!
//! Same forward-difference convention as everywhere else in the workspace: perturb along one
//! basis direction of the tangent space, re-evaluate, compare against `(f(x + δdᵢ) - f(x))/δ`.
//! Unlike `skoro-operation`, this crate *has* manifolds, so the perturbation goes through
//! `retract` and the comparison through `inverse_retract` -- which is what makes the harness
//! carry over unchanged to the curved manifolds in `skoro-so3` and `skoro-se3`.

#![allow(dead_code)]

use skoro_backend::{NdArrayBackend, Tensor, TensorData};
use skoro_operation::{
    Input,
    diff::{DiffModeState, DiffOperation, DiffOperationForward},
};
use skoro_riemann::DynManifold;

pub type B = NdArrayBackend<f64, i32>;

pub const DELTA: f64 = 1e-6;
pub const EPSILON: f64 = 1e-5;

pub fn tensor2(data: Vec<f64>, batch: usize, components: usize) -> Tensor<B, 2> {
    Tensor::from_data(
        TensorData::new(data, [batch, components]),
        &Default::default(),
    )
}

pub fn tensor3(data: Vec<f64>, dims: [usize; 3]) -> Tensor<B, 3> {
    Tensor::from_data(TensorData::new(data, dims), &Default::default())
}

pub fn row(data: Vec<f64>) -> Tensor<B, 2> {
    let n = data.len();
    tensor2(data, 1, n)
}

pub fn into_vec<const D: usize>(t: Tensor<B, D>) -> Vec<f64> {
    t.into_data().to_vec::<f64>().unwrap()
}

pub fn identity(dim: usize, batch: usize) -> Tensor<B, 3> {
    let mut data = vec![0.0; dim * dim];
    for i in 0..dim {
        data[i * (dim + 1)] = 1.0;
    }
    tensor3(data, [1, dim, dim]).repeat_dim(0, batch)
}

pub fn basis(dim: usize, i: usize, batch: usize) -> Tensor<B, 2> {
    let mut data = vec![0.0; dim];
    data[i] = 1.0;
    tensor2(data, 1, dim).repeat_dim(0, batch)
}

pub fn floats(tensors: &[Tensor<B, 2>]) -> Vec<Input<B>> {
    tensors.iter().cloned().map(Input::Float).collect()
}

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

fn stepped(inputs: &[Tensor<B, 2>], wrt: usize, i: usize, delta: f64) -> Vec<Tensor<B, 2>> {
    let mut out = inputs.to_vec();
    let [batch, dim] = out[wrt].dims();
    out[wrt] = out[wrt].clone() + basis(dim, i, batch).mul_scalar(delta);
    out
}

/// Check the pushforward with respect to input `wrt` against a forward difference.
pub fn test_jvp_numerically<Op: DiffOperation<B>>(op: &Op, inputs: &[Tensor<B, 2>], wrt: usize) {
    let [batch, indim] = inputs[wrt].dims();
    let mut seeds: Vec<Option<Tensor<B, 3>>> = vec![None; inputs.len()];
    seeds[wrt] = Some(identity(indim, batch));

    let (y0, tangent) = jvp(op, floats(inputs), seeds);
    let tangent = tangent.expect("a seeded differentiable input must produce a tangent");

    for i in 0..indim {
        let yi = forward_only(op, floats(&stepped(inputs, wrt, i, DELTA)));
        let expected = into_vec((yi - y0.clone()).div_scalar(DELTA));
        let actual = into_vec(tangent.clone().slice_dim(2, i).squeeze_dim::<2>(2));
        approx::assert_abs_diff_eq!(actual.as_slice(), expected.as_slice(), epsilon = EPSILON);
    }
}

/// Check the pullback into input `wrt` against a forward difference.
pub fn test_vjp_numerically<Op: DiffOperation<B>>(op: &Op, inputs: &[Tensor<B, 2>], wrt: usize) {
    let [batch, indim] = inputs[wrt].dims();
    let diffs = vec![true; inputs.len()];
    let (y0, state) = vjp_state(op, floats(inputs), diffs);
    let outdim = y0.dims()[1];

    let block = op.vjp(&state, identity(outdim, batch))[wrt]
        .clone()
        .expect("a differentiable input must receive a cotangent");

    for i in 0..indim {
        let yi = forward_only(op, floats(&stepped(inputs, wrt, i, DELTA)));
        let expected = into_vec((yi - y0.clone()).div_scalar(DELTA));
        let actual = into_vec(block.clone().slice_dim(1, i));
        approx::assert_abs_diff_eq!(actual.as_slice(), expected.as_slice(), epsilon = EPSILON);
    }
}

/// Check the mixed second-order block against a forward difference of the VJP.
pub fn test_hvp_numerically<Op: DiffOperation<B>>(
    op: &Op,
    inputs: &[Tensor<B, 2>],
    seed: usize,
    read: usize,
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
        // An absent slot means "no contribution", which is read as zero and still compared.
        let exact = match &op.hvp(&vjp_st, &hvp_st, omega.clone())[read] {
            Some(block) => into_vec(block.clone()),
            None => vec![0.0; batch * readdim * outdim],
        };

        let (_, base) = vjp_state(op, floats(inputs), diffs.clone());
        let vjp0 = op.vjp(&base, omega.clone())[read].clone().unwrap();
        let moved = stepped(inputs, seed, i, DELTA);
        let (_, moved_state) = vjp_state(op, floats(&moved), diffs.clone());
        let vjp1 = op.vjp(&moved_state, omega.clone())[read].clone().unwrap();

        let expected = into_vec((vjp1 - vjp0).div_scalar(DELTA));
        approx::assert_abs_diff_eq!(exact.as_slice(), expected.as_slice(), epsilon = EPSILON);
    }
}

pub fn check_all_derivatives<Op: DiffOperation<B>>(op: &Op, inputs: &[Tensor<B, 2>]) {
    for wrt in 0..inputs.len() {
        test_jvp_numerically(op, inputs, wrt);
        test_vjp_numerically(op, inputs, wrt);
        for read in 0..inputs.len() {
            test_hvp_numerically(op, inputs, wrt, read);
        }
    }
}

/// `inverse_retract(p, retract(p, v)) == v`, the relation that defines the pair.
///
/// This is the one property every manifold owes regardless of its geometry, and the only check
/// that stays meaningful when the two operations stop being addition and subtraction.
pub fn check_retraction_round_trip<M: DynManifold<B>>(
    manifold: &M,
    point: Tensor<B, 2>,
    tangent: Tensor<B, 2>,
) {
    assert_eq!(
        tangent.dims()[1],
        manifold.dimension(),
        "a tangent has the manifold's dimension, not its storage size"
    );

    let moved = manifold.retract(point.clone(), tangent.clone());
    assert_eq!(
        moved.dims(),
        point.dims(),
        "retraction stays on the manifold"
    );

    let recovered = manifold.inverse_retract(point, moved);
    approx::assert_abs_diff_eq!(
        into_vec(recovered).as_slice(),
        into_vec(tangent).as_slice(),
        epsilon = EPSILON
    );
}
