//! Adding Gaussians through any builder, and the covariance helpers.
//!
//! There is no Gaussian-specific entry point: a Gaussian enters a model through
//! `add_quantity(Normal::new(mean, std_dev))` on any `ModelBuilderOps`, so an adapter that builds
//! Gaussian sub-models is generic over the builder rather than over a capability trait. What the
//! old `GaussianModelClientOps` bound expressed — "this client can add Gaussians" — was never a
//! claim about what a model *contained*, and every client that can grow a model can add one.

use std::collections::HashMap;

use skoro_backend::{Flex, Point, TensorData};
use skoro_linalg::tril::Tril;
use skoro_model::{ModelBuilderOps, ModelReaderOps, model::ModelClient, q::Q};
use skoro_normal::{
    distribution::Normal,
    facade::{constant_covariance_diagonal, constant_covariance_identity},
    structure::GaussianBlock,
};
use skoro_operation::Input;
use skoro_riemann::{
    Manifold,
    euclidean::{EuclideanD, VectorD},
};

type B = Flex;

/// An adapter generic over *any* builder, which is the shape a producer of sub-models takes.
fn add_unit_gaussian<C: ModelBuilderOps<B>>(
    client: &C,
    mean: &Q<B, EuclideanD>,
) -> Q<B, EuclideanD> {
    let cov = constant_covariance_identity(client, 3);
    client.add_quantity(Normal::new(mean.clone(), cov)).unwrap()
}

#[test]
fn a_gaussian_quantity_scores_the_peak_at_the_mean() {
    let model = ModelClient::<B>::new(Default::default());
    let mean = model.add_constant(EuclideanD(3), &VectorD::new(&[10.0, 20.0, 30.0]));
    let std_dev = constant_covariance_diagonal(&model, &[1.0, 2.0, 3.0]);

    let x = model.add_quantity(Normal::new(mean, std_dev)).unwrap();

    // Evaluate the model log-density at the mean: should be the Gaussian peak.
    let mut points = HashMap::new();
    points.insert(
        x.id(),
        Input::float(Point::<B>::from_data(
            TensorData::new(vec![10.0, 20.0, 30.0], [1, 3]),
            &Default::default(),
        )),
    );
    let ld = model.log_density(&points).unwrap();
    let value = ld.into_data().to_vec::<f32>().unwrap()[0];

    // At the mean the whitened residual is zero, so log density = constant +
    // log|Sigma^-1|. constant = -0.5 * 3 * ln(2*pi); log_det = -ln(1*2*3).
    let expected = -0.5 * 3.0 * (std::f64::consts::TAU.ln()) as f32 - (1.0f32 * 2.0 * 3.0).ln();
    approx::assert_abs_diff_eq!(value, expected, epsilon = 1e-4);
}

/// The adapter is generic over the builder, and what it built is observable and scoring.
#[test]
fn a_builder_generic_adapter_grows_a_real_model() {
    let model = ModelClient::<B>::new(Default::default());
    let mean = model.add_constant(EuclideanD(3), &VectorD::new(&[1.0, 2.0, 3.0]));
    let x = add_unit_gaussian(&model, &mean);

    model.observe(&x, &VectorD::new(&[1.0, 2.0, 3.0])).unwrap();
    let ld = model.log_density(&HashMap::new()).unwrap();
    assert_eq!(ld.dims(), [1, 1]);

    // And its factor binds the mean it was given.
    assert_eq!(model.factors().len(), 1);
    let blocks = GaussianBlock::all(&model);
    assert_eq!(blocks.len(), 1);
    assert_eq!(blocks[0].mean, mean.id());
}

#[test]
fn covariance_helpers_shapes() {
    let model = ModelClient::<B>::new(Default::default());
    let cov_diag: Q<B, Tril> = constant_covariance_diagonal(&model, &[2.0, 3.0]);
    let cov_id: Q<B, Tril> = constant_covariance_identity(&model, 2);
    // Tril(2) packs into n(n+1)/2 = 3 components.
    assert_eq!(
        <Tril as Manifold<B>>::dimension(&cov_diag.get_quantity_type()),
        3
    );
    assert_eq!(
        <Tril as Manifold<B>>::dimension(&cov_id.get_quantity_type()),
        3
    );
}
