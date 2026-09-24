//! The bridge from a model to host elimination, on a manifold whose storage is not its dimension.
//!
//! `SE3` is 6-dimensional over 12 components. The host tests in `skoro-dense` use key dimensions of
//! 3 and 6, but only as numbers; storage and dimension can only be confused where a model meets the
//! host — splitting a Jacobian's columns per key, reading a gradient per key, and retracting an
//! anchor by a marginal's mean. These tests put an `SE3` chain through each of those.
//!
//! Here because the umbrella crate is the one that sees `SE3`, `Normal` and `skoro-linearize`
//! together.

use approx::assert_abs_diff_eq;

use skoro_backend::{Flex, Tensor, TensorData};
use skoro_core::id::Id;
use skoro_dense::dense::Dense;
use skoro_linearize::{Curvature, joint_normal_from, linearize, marginalize_out};
use skoro_model::{
    ModelBuilderOps, ModelReaderOps,
    hash_map_builders::{point_map, points},
    jacobian::jacobian_vjp,
    least_squares::{residual_layout, residual_vector},
    model::ModelClient,
    q::Q,
    test_utils::into_vec,
};
use skoro_normal::{distribution::Normal, facade::constant_covariance_diagonal};
use skoro_operation::ValueMap;
use skoro_se3::quantity::{SE3, SE3Pose};
use skoro_so3::quantity::SO3Matrix;

type B = Flex;
skoro_backend::f64_test_binary!(Flex);

fn row(values: &[f64]) -> Tensor<B, 2> {
    Tensor::from_data(
        TensorData::new(values.to_vec(), [1, values.len()]),
        &Default::default(),
    )
}

/// A rotation about z by `theta`, then about x by `phi`, translated to `t`: far from the identity
/// on two axes, so a Jacobian that ignored the rotation would show.
fn rotation(theta: f64, phi: f64) -> [f64; 9] {
    let (c, s) = (theta.cos(), theta.sin());
    let (cp, sp) = (phi.cos(), phi.sin());
    // Rz(theta) · Rx(phi), row-major.
    [c, -s * cp, s * sp, s, c * cp, -c * sp, 0.0, sp, cp]
}

fn pose(theta: f64, phi: f64, t: [f64; 3]) -> SE3Pose<f64> {
    SE3Pose::new(SO3Matrix(rotation(theta, phi)), t)
}

fn pose_row(theta: f64, phi: f64, t: [f64; 3]) -> Tensor<B, 2> {
    let r = rotation(theta, phi);
    let mut v = r.to_vec();
    v.extend_from_slice(&t);
    row(&v)
}

/// `p0 ~ N(prior, P0)`, `p1 ~ N(p0, Q)`, and a pose measurement of `p1`.
fn chain() -> (ModelClient<B>, Q<B, SE3>, Q<B, SE3>) {
    let model = ModelClient::<B>::default();
    let prior = model.add_constant(SE3, &pose(0.2, -0.1, [0.5, -1.0, 0.25]));
    let p0_cov = constant_covariance_diagonal(&model, &[0.5, 0.4, 0.3, 1.0, 1.5, 0.8]);
    let q_cov = constant_covariance_diagonal(&model, &[0.2, 0.3, 0.25, 0.6, 0.5, 0.7]);
    let r_cov = constant_covariance_diagonal(&model, &[0.3, 0.2, 0.4, 0.5, 0.9, 0.6]);
    let p0 = model.add_quantity(Normal::new(prior, p0_cov)).unwrap();
    let p1 = model.add_quantity(Normal::new(p0.clone(), q_cov)).unwrap();
    let z = model.add_quantity(Normal::new(p1.clone(), r_cov)).unwrap();
    model
        .observe(&z, &pose(0.9, 0.35, [2.0, -0.5, 1.5]))
        .unwrap();
    (model, p0, p1)
}

/// Both poses away from the identity and from each other, and away from the optimum.
fn at(p0: &Q<B, SE3>, p1: &Q<B, SE3>) -> ValueMap<B> {
    points()
        .with_quantity(p0, pose_row(0.4, 0.3, [1.5, -2.25, 0.75]))
        .with_quantity(p1, pose_row(-0.3, 0.6, [0.5, 1.0, -0.5]))
        .build()
}

#[test]
fn an_se3_linearization_is_keyed_by_dimension_and_matches_the_dense_rows() {
    let (model, p0, p1) = chain();
    let values = at(&p0, &p1);
    let ids: Vec<Id> = vec![p0.id(), p1.id()];
    let free = model.tangent_layout(&ids).unwrap();

    let graph = linearize(&model, &values, &free, Curvature::Residual)
        .unwrap()
        .remove(0)
        .graph;
    assert_eq!(
        graph.dim(p0.id()),
        Some(6),
        "a dimension, not the 12-wide storage"
    );
    assert_eq!(graph.dim(p1.id()), Some(6));
    assert_eq!(graph.total_dim(), 12);

    // Factors follow `residual_layout`'s row order, so the assembled rows are the dense ones.
    let (a, b) = graph.to_dense(&ids).unwrap();
    let rows = residual_layout(&model).unwrap();
    let j = Dense::from_tensor(jacobian_vjp(&model, &values, &free, &rows).unwrap()).unwrap();
    let r = Dense::from_vector(residual_vector(&model, &values).unwrap()).unwrap();
    assert_eq!((a.rows(), a.cols()), (18, 12));
    assert!(a.max_abs_diff(&j).unwrap() < 1e-12);
    assert!(b.max_abs_diff(&r.neg()).unwrap() < 1e-12);

    // The Fisher route reads the gradient per key, also by dimension.
    let fisher = linearize(&model, &values, &free, Curvature::Fisher)
        .unwrap()
        .remove(0);
    let gradient = fisher.gradient.unwrap();
    assert_eq!(gradient.get(p0.id()).unwrap().rows(), 6);
    assert_eq!(gradient.get(p1.id()).unwrap().rows(), 6);
}

/// The joint factor built from a marginal keeps the linearization point and the right-hand side,
/// `r = W (x ⊖ at) − d`, so at `at` its residual is exactly `−d` — the eliminated rows' value at
/// `δ = 0` — on `SE3` as on a flat space.
///
/// Anchoring at the marginal's mean `at ⊕ W⁻¹d` instead would not be: skoro's `SE3` retraction is
/// not a group exponential, so `at ⊖ (at ⊕ o)` is `−o` only to first order, and with `‖o‖ ≈ 1` here
/// the gap is of the size of `d` itself. This test is what caught that.
#[test]
fn a_joint_factor_from_an_se3_marginal_is_anchored_at_the_marginal_mean() {
    let (model, p0, p1) = chain();
    let values = at(&p0, &p1);
    let free = model.tangent_layout(&[p0.id(), p1.id()]).unwrap();

    let marginal = marginalize_out(&model, &values, &free, &[p0.id()], Curvature::Residual)
        .unwrap()
        .remove(0);
    assert_eq!(marginal.keys(), &[p1.id()]);
    assert_eq!(marginal.dims(), &[6]);
    let offset = marginal.offset().unwrap();
    assert!(
        offset.norm() > 0.1,
        "linearized off the optimum, the marginal's mean is a real move: {}",
        offset.norm()
    );

    let anchor_at = point_map()
        .with_quantity(&p1, pose_row(-0.3, 0.6, [0.5, 1.0, -0.5]))
        .build();
    let joint = joint_normal_from(&model, &marginal, &anchor_at).unwrap();
    assert_eq!(joint.ids(), &[p1.id()]);

    let residual = joint.whitened_residual(&model).unwrap();
    // A stochastic `p1` compiles with its sampler, so its parent `p0` needs a point too.
    let r = into_vec(model.evaluate(&values, &residual).unwrap());
    let d = marginal.rhs().to_row_major();
    assert_eq!(r.len(), 6);
    for (ri, di) in r.iter().zip(&d) {
        assert_abs_diff_eq!(*ri, -di, epsilon = 1e-9);
    }
}
