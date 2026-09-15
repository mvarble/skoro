//! Importance sampling end to end: draw from a proposal the model can sample, score a target it
//! can only evaluate, and let the log-weight reconcile the two.
//!
//! The point of `add_proposal` is that these need not be the same distribution. Here the target
//! is a free-standing factor over `z`, and `z` itself is generated from a Gaussian that the
//! posterior never scores.

use skoro_backend::{NdArrayBackend, Tensor, TensorData};
use skoro_model::{
    ModelBuilderOps, ModelClientOps, distribution::Factor, hash_map_builders::points,
    map::register_map1, model::ModelClient, test_utils::into_vec,
};
use skoro_normal::distribution::Normal;
use skoro_operation::ops::LiteralScalarMultiplication;
use skoro_riemann::{euclidean::VectorD, real::Real};

type B = NdArrayBackend<f64, i32>;

fn row(values: &[f64]) -> Tensor<B, 2> {
    let n = values.len();
    Tensor::<B, 1>::from_data(TensorData::new(values.to_vec(), [n]), &Default::default())
        .reshape([1, n])
}

/// A model whose only quantity is proposed: `z ~ N(mu, sigma)` under `q`, and `log p(z) = c * z`
/// under the target. Returns the model and `z`.
fn proposed_model(mu: f64, sigma: f64, c: f64) -> (ModelClient<B>, skoro_model::q::Q<B, Real>) {
    let model = ModelClient::<B>::default();
    let mean = model.add_constant(Real, &mu);
    let whitening = model.add_constant(skoro_linalg::tril::Tril(1), &VectorD::new(&[sigma]));
    let z = model.add_proposal(Normal::new(mean, whitening)).unwrap();

    let target = register_map1(Real, LiteralScalarMultiplication::new(c), &z).unwrap();
    model.add_factor(Factor::new(target), &[z.id()]).unwrap();
    (model, z)
}

/// `log N(z; mu, sigma)` for a scalar, on the host.
fn log_normal(z: f64, mu: f64, sigma: f64) -> f64 {
    let r = (z - mu) / sigma;
    -0.5 * r * r - (2.0 * std::f64::consts::PI).sqrt().ln() - sigma.ln()
}

#[test]
fn the_weight_corrects_a_proposal_to_the_target() {
    let (mu, sigma, c) = (0.5, 1.3, -2.0);
    let (model, z) = proposed_model(mu, sigma, c);

    // The target term is the whole log-density; the Gaussian is the whole proposal.
    let flags: Vec<_> = model
        .terms()
        .into_iter()
        .map(|t| (t.in_density, t.in_proposal))
        .collect();
    assert_eq!(flags, vec![(false, true), (true, false)]);

    let z_val = 0.75;
    let values = points().with_quantity(&z, row(&[z_val])).build();

    let log_target = c * z_val;
    let log_q = log_normal(z_val, mu, sigma);

    approx::assert_abs_diff_eq!(
        into_vec(model.log_density(&values).unwrap())[0],
        log_target,
        epsilon = 1e-12
    );
    approx::assert_abs_diff_eq!(
        into_vec(model.log_proposal(&values).unwrap())[0],
        log_q,
        epsilon = 1e-12
    );
    approx::assert_abs_diff_eq!(
        into_vec(model.log_weight(&values).unwrap())[0],
        log_target - log_q,
        epsilon = 1e-12
    );
}

/// Generation keys off the node being `Stochastic`, not off term membership: a proposal-only
/// quantity draws exactly as a scored one does, and the weight can be read at the draws.
#[test]
fn a_proposal_only_quantity_still_generates() {
    let (mu, sigma, c) = (0.5, 1.3, -2.0);
    let (model, z) = proposed_model(mu, sigma, c);

    let n = 64;
    let generated = model.generate(n).unwrap();
    let (drawn, _) = generated.variables.get(&z.id()).expect("z is generated");
    let drawn = drawn.clone().unwrap_float();
    assert_eq!(drawn.dims(), [n, 1]);

    let values = points().with_quantity(&z, drawn.clone()).build();
    let weights = into_vec(model.log_weight(&values).unwrap());
    let draws = into_vec(drawn);
    assert_eq!(weights.len(), n);
    for (w, zi) in weights.iter().zip(&draws) {
        approx::assert_abs_diff_eq!(*w, c * zi - log_normal(*zi, mu, sigma), epsilon = 1e-12);
    }
    // A degenerate draw would make the loop above vacuous.
    assert!(
        draws.windows(2).any(|p| p[0] != p[1]),
        "the proposal must actually sample"
    );
}
