//! Gradients of the proposal and of the log-weight at given points, against closed forms.
//!
//! `z ~ N(μ, σ)` is proposed and scored by the target as `c·z`, and `w ~ N(μ, σ₂)` is in both
//! densities, so its term must cancel out of the weight, in the gradient as in the value. `μ` is a
//! free parameter: `∇_μ log q` at a fixed draw is the score a score-function estimator needs.

use std::collections::HashMap;

use skoro_backend::{Flex, Tensor, TensorData};
use skoro_core::id::Id;
use skoro_model::{
    ModelBuilderOps, ModelReaderOps, distribution::Factor, hash_map_builders::points,
    map::register_map1, model::ModelClient, test_utils::into_vec,
};
use skoro_normal::distribution::Normal;
use skoro_operation::ops::LiteralScalarMultiplication;
use skoro_riemann::{euclidean::VectorD, real::Real};

type B = Flex;
skoro_backend::f64_test_binary!(Flex);

const MU: f64 = 0.5;
const SIGMA: f64 = 1.3;
const SIGMA_W: f64 = 0.7;
const C: f64 = -2.0;
const Z: f64 = 0.75;
const W: f64 = -0.4;

fn row(value: f64) -> Tensor<B, 2> {
    Tensor::from_data(TensorData::new(vec![value], [1, 1]), &Default::default())
}

fn scalar(cotangents: &HashMap<Id, Tensor<B, 2>>, id: Id) -> f64 {
    cotangents.get(&id).map_or(0.0, |g| into_vec(g.clone())[0])
}

#[test]
fn the_proposal_and_weight_gradients_are_the_scores_at_a_fixed_draw() {
    let model = ModelClient::<B>::default();
    let mu = model.add_parameter(Real);
    let sigma = model.add_constant(skoro_linalg::tril::Tril(1), &VectorD::new(&[SIGMA]));
    let sigma_w = model.add_constant(skoro_linalg::tril::Tril(1), &VectorD::new(&[SIGMA_W]));
    let z = model.add_proposal(Normal::new(mu.clone(), sigma)).unwrap();
    let target = register_map1(Real, LiteralScalarMultiplication::new(C), &z).unwrap();
    model.add_factor(Factor::new(target), &[z.id()]).unwrap();
    let w = model
        .add_quantity(Normal::new(mu.clone(), sigma_w))
        .unwrap();

    let values = points()
        .with_quantity(&mu, row(MU))
        .with_quantity(&z, row(Z))
        .with_quantity(&w, row(W))
        .build();

    // log q = ℓ(z; μ, σ) + ℓ(w; μ, σ₂).
    let q = model.log_proposal_ord1(&values).unwrap();
    assert!(
        (into_vec(q.value)[0] - into_vec(model.log_proposal(&values).unwrap())[0]).abs() < 1e-12
    );
    let (dz, dw) = ((Z - MU) / SIGMA.powi(2), (W - MU) / SIGMA_W.powi(2));
    assert!((scalar(&q.cotangents, mu.id()) - (dz + dw)).abs() < 1e-12);
    assert!((scalar(&q.cotangents, z.id()) + dz).abs() < 1e-12);
    assert!((scalar(&q.cotangents, w.id()) + dw).abs() < 1e-12);

    // log w = c·z + ℓ(w) − ℓ(z) − ℓ(w) = c·z − ℓ(z; μ, σ): `w`'s term cancels.
    let weight = model.log_weight_ord1(&values).unwrap();
    assert!(
        (into_vec(weight.value)[0] - into_vec(model.log_weight(&values).unwrap())[0]).abs() < 1e-12
    );
    assert!((scalar(&weight.cotangents, mu.id()) + dz).abs() < 1e-12);
    assert!((scalar(&weight.cotangents, z.id()) - (C + dz)).abs() < 1e-12);
    assert_eq!(
        scalar(&weight.cotangents, w.id()),
        0.0,
        "a term in both cancels exactly"
    );
}
