//! Numerical checks for the derivatives a [`Graph`] reports.
//!
//! Every derivative in this workspace is verified the same way: perturb an input by `δ` along one
//! basis direction of its tangent space, re-evaluate, and compare the analytic answer against the
//! **forward** difference. `skoro-model`'s `test_utils` is the same harness one level up, taking
//! its quantity handles where this one takes `(&Graph, Id)` pairs; keeping the two in step is why
//! this lives in `src` behind a feature rather than in some crate's `tests/`.
//!
//! Forward rather than central differences on purpose. The truncation error is therefore `O(δ)`,
//! not `O(δ²)`, so `epsilon` has to stay loose relative to `delta` -- `DELTA = 1e-6` with
//! `EPSILON = 1e-5` is the pairing the rest of the workspace uses. A check that fails by a small
//! multiple of `delta` is the method, not the operation; a check that fails by a *factor* is a
//! dropped term.
//!
//! Perturbation goes through [`DynManifold::retract`] and comparison through
//! [`DynManifold::inverse_retract`], never through `+` and `-`. On the flat manifolds reachable
//! from this crate the two coincide, but writing it this way is what lets the same harness check
//! `SO3` and `SE3` unchanged.

use std::collections::HashMap;

use skoro_backend::{Backend, CotangentFrame, TangentFrame, Tensor, TensorData};
use skoro_core::id::Id;
use skoro_operation::{Input, ValueMap};

use crate::graph::Graph;

/// A `[1, dim, dim]` identity, batched to `batches`.
fn identity_frames<B: Backend>(dim: usize, batches: usize) -> Tensor<B, 3> {
    let mut data = vec![0.0f64; dim * dim];
    for i in 0..dim {
        data[i * (dim + 1)] = 1.0;
    }
    Tensor::from_data(TensorData::new(data, [1, dim, dim]), &B::Device::default())
        .repeat_dim(0, batches)
}

/// The `i`-th basis tangent vector, shaped `[batches, dim]`.
fn basis_vector<B: Backend>(dim: usize, i: usize, batches: usize) -> Tensor<B, 2> {
    let mut data = vec![0.0f64; dim];
    data[i] = 1.0;
    Tensor::from_data(TensorData::new(data, [1, dim]), &B::Device::default()).repeat_dim(0, batches)
}

/// Flatten a tensor to its row-major elements, for comparison.
///
/// Shapes are asserted separately by the callers; once they agree, the flat layouts do too, so
/// the comparison does not need to carry the shape.
pub fn into_vec<B: Backend, const D: usize>(t: Tensor<B, D>) -> Vec<f64> {
    t.into_data().convert::<f64>().to_vec().unwrap()
}

/// The batch axis of the point stored for `id`.
fn batches_of<B: Backend>(point: &ValueMap<B>, id: Id, who: &str) -> usize {
    point
        .get(&id)
        .unwrap_or_else(|| panic!("{who}: no point supplied for {id:?}"))
        .dims()[0]
}

/// Check `d(out)/d(in)` as reported by [`Graph::evaluate_jvp`] against a forward difference.
///
/// Seeds the identity tangent frame on `in_id`, so column `i` of the reported tangent is the
/// pushforward of basis direction `i`, and compares it to
/// `Rinv(f(x), f(R(x, δ dᵢ))) / δ`.
pub fn test_jvp_numerically<B>(
    graph: &Graph<B>,
    in_id: Id,
    out_id: Id,
    mut point: ValueMap<B>,
    delta: f64,
    epsilon: f64,
) where
    B: Backend,
{
    let intype = graph.get_manifold(in_id).expect("input must be continuous");
    let outtype = graph
        .get_manifold(out_id)
        .expect("output must be continuous");
    let indim = intype.dimension();
    let outdim = outtype.dimension();
    let batches = batches_of(&point, in_id, "test_jvp_numerically");

    let ident = identity_frames::<B>(indim, batches);
    let mut tangents: HashMap<Id, TangentFrame<B>> = HashMap::new();
    tangents.insert(in_id, ident.clone());

    let mut eval = graph
        .evaluate_jvp(&point, &tangents, out_id)
        .expect("test_jvp_numerically: JVP evaluation failed");
    let y0 = eval.value;
    let dydx = eval
        .tangents
        .remove(&out_id)
        .expect("test_jvp_numerically: no tangent reported at the output");
    assert_eq!(dydx.dims(), [batches, outdim, indim], "JVP frame shape");

    let dx = ident.mul_scalar(delta);
    let x0 = point
        .remove(&in_id)
        .expect("test_jvp_numerically: input point vanished")
        .unwrap_float();
    for i in 0..indim {
        let vi = dx.clone().slice_dim(2, i).squeeze_dim(2);
        let xi = intype.retract(x0.clone(), vi);
        point.insert(in_id, Input::Float(xi));
        let yi = graph
            .evaluate(&point, out_id)
            .expect("test_jvp_numerically: perturbed evaluation failed");
        let approx_col = into_vec(outtype.inverse_retract(y0.clone(), yi).div_scalar(delta));
        let exact_col = into_vec(dydx.clone().slice_dim(2, i).squeeze_dim::<2>(2));
        approx::assert_abs_diff_eq!(
            exact_col.as_slice(),
            approx_col.as_slice(),
            epsilon = epsilon
        );
    }
}

/// Check `d(out)/d(in)ᵀ` as reported by [`Graph::evaluate_vjp`] against a forward difference.
///
/// Seeds the identity cotangent frame on `out_id`, so row `i` of the reported cotangent is the
/// same quantity the JVP reports as column `i`. The two therefore have to agree, which is what
/// catches a silently transposed Jacobian.
pub fn test_vjp_numerically<B>(
    graph: &Graph<B>,
    in_id: Id,
    out_id: Id,
    mut point: ValueMap<B>,
    delta: f64,
    epsilon: f64,
) where
    B: Backend,
{
    let intype = graph.get_manifold(in_id).expect("input must be continuous");
    let outtype = graph
        .get_manifold(out_id)
        .expect("output must be continuous");
    let indim = intype.dimension();
    let outdim = outtype.dimension();
    let batches = batches_of(&point, in_id, "test_vjp_numerically");

    let mut cotangents: HashMap<Id, CotangentFrame<B>> = HashMap::new();
    cotangents.insert(out_id, identity_frames::<B>(outdim, batches));

    let mut eval = graph
        .evaluate_vjp(&point, &cotangents, out_id)
        .expect("test_vjp_numerically: VJP evaluation failed");
    let y0 = eval.value;
    let dydxt = eval
        .cotangents
        .remove(&in_id)
        .expect("test_vjp_numerically: no cotangent reported at the input");
    assert_eq!(dydxt.dims(), [batches, indim, outdim], "VJP frame shape");

    let dx = identity_frames::<B>(indim, batches).mul_scalar(delta);
    let x0 = point
        .remove(&in_id)
        .expect("test_vjp_numerically: input point vanished")
        .unwrap_float();
    for i in 0..indim {
        let vi = dx.clone().slice_dim(2, i).squeeze_dim(2);
        let xi = intype.retract(x0.clone(), vi);
        point.insert(in_id, Input::Float(xi));
        let yi = graph
            .evaluate(&point, out_id)
            .expect("test_vjp_numerically: perturbed evaluation failed");
        let approx_row = into_vec(outtype.inverse_retract(y0.clone(), yi).div_scalar(delta));
        let exact_row = into_vec(dydxt.clone().slice_dim(1, i));
        approx::assert_abs_diff_eq!(
            exact_row.as_slice(),
            approx_row.as_slice(),
            epsilon = epsilon
        );
    }
}

/// Check the diagonal Hessian block `∂²out / ∂in²`.
///
/// Only ever reaches the diagonal; use [`test_hvp_cross_numerically`] for the mixed blocks of a
/// multi-input operation, which is where an asymmetric mistake hides.
pub fn test_hvp_numerically<B>(
    graph: &Graph<B>,
    in_id: Id,
    out_id: Id,
    point: ValueMap<B>,
    delta: f64,
    epsilon: f64,
) where
    B: Backend,
{
    test_hvp_cross_numerically(graph, in_id, in_id, out_id, point, delta, epsilon)
}

/// Check the mixed Hessian block `∂²out / ∂read ∂seed`.
///
/// The order-1 tangent is seeded on `seed_id` (and the numerical perturbation applied there),
/// while the order-2 cotangent compared is the one accumulated at `read_id`. Passing the same id
/// for both recovers [`test_hvp_numerically`].
///
/// The reference is a forward difference *of the VJP*: `(vjp(x + δv) - vjp(x)) / δ`. That is the
/// definition the `hvp` contract states, so a `hvp` that forgot the Leibniz term shows up here
/// and nowhere in a first-order test.
pub fn test_hvp_cross_numerically<B>(
    graph: &Graph<B>,
    seed_id: Id,
    read_id: Id,
    out_id: Id,
    point: ValueMap<B>,
    delta: f64,
    epsilon: f64,
) where
    B: Backend,
{
    let seedtype = graph
        .get_manifold(seed_id)
        .expect("seed must be continuous");
    let outtype = graph
        .get_manifold(out_id)
        .expect("output must be continuous");
    let seeddim = seedtype.dimension();
    let outdim = outtype.dimension();
    let readdim = graph
        .get_manifold(read_id)
        .expect("read quantity must be continuous")
        .dimension();
    let batches = batches_of(&point, seed_id, "test_hvp_cross_numerically");

    let mut ord1: HashMap<Id, CotangentFrame<B>> = HashMap::new();
    ord1.insert(out_id, identity_frames::<B>(outdim, batches));
    let mut ord2: HashMap<Id, CotangentFrame<B>> = HashMap::new();
    ord2.insert(
        out_id,
        Tensor::zeros([batches, outdim, outdim], &B::Device::default()),
    );

    let mut perturbed = point.clone();
    let x0 = perturbed
        .remove(&seed_id)
        .expect("test_hvp_cross_numerically: seed point vanished")
        .unwrap_float();
    let dx = identity_frames::<B>(seeddim, batches).mul_scalar(delta);

    for i in 0..seeddim {
        let mut tangents: HashMap<Id, Tensor<B, 2>> = HashMap::new();
        tangents.insert(seed_id, basis_vector::<B>(seeddim, i, batches));
        let mut eval = graph
            .evaluate_hvp(&point, &tangents, &ord1, &ord2, out_id)
            .expect("test_hvp_cross_numerically: HVP evaluation failed");
        let vjp0 = eval
            .ord1_cotangents
            .remove(&read_id)
            .expect("no order-1 cotangent reported at the read quantity");
        let hvp = eval
            .ord2_cotangents
            .remove(&read_id)
            .expect("no order-2 cotangent reported at the read quantity");
        assert_eq!(hvp.dims(), [batches, readdim, outdim], "HVP frame shape");

        let vi = dx.clone().slice_dim(2, i).squeeze_dim(2);
        let xi = seedtype.retract(x0.clone(), vi);
        perturbed.insert(seed_id, Input::Float(xi));
        let mut shifted = graph
            .evaluate_vjp(&perturbed, &ord1, out_id)
            .expect("test_hvp_cross_numerically: perturbed VJP failed");
        let vjp1 = shifted
            .cotangents
            .remove(&read_id)
            .expect("no cotangent reported at the read quantity after perturbation");

        let approx_block = into_vec((vjp1 - vjp0).div_scalar(delta));
        let exact_block = into_vec(hvp);
        approx::assert_abs_diff_eq!(
            exact_block.as_slice(),
            approx_block.as_slice(),
            epsilon = epsilon
        );
    }
}
