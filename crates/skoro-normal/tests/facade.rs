//! Tests for the Gaussian model facades.

use std::collections::HashMap;

use skoro_backend::{NdArrayBackend, Point, TensorData};
use skoro_linalg::tril::Tril;
use skoro_model::{ModelClientOps, model::ModelClient, q::Q};
use skoro_normal::facade::{
    GaussianModelClient, GaussianModelClientOps, constant_covariance_diagonal,
    constant_covariance_identity,
};
use skoro_operation::Input;
use skoro_riemann::{
    Manifold,
    euclidean::{EuclideanD, VectorD},
};

type B = NdArrayBackend<f32, i32>;

/// An adapter that is generic over *any* client that can add Gaussian
/// quantities. Demonstrates the capability-trait (contravariant) use: it works
/// against both a general `ModelClient` and a restricted `GaussianModelClient`.
fn add_unit_gaussian<C: GaussianModelClientOps<B>>(
    client: &C,
    mean: &Q<B, EuclideanD>,
) -> Q<B, EuclideanD> {
    let cov = constant_covariance_identity(client, 3);
    client.add_gaussian_quantity(mean, &cov).unwrap()
}

#[test]
fn gaussian_client_add_gaussian_quantity_density() {
    let model = GaussianModelClient::<B>::new(Default::default());
    let mean = model.add_constant(EuclideanD(3), &VectorD::new(&[10.0, 20.0, 30.0]));
    let std_dev = constant_covariance_diagonal(&model, &[1.0, 2.0, 3.0]);

    let x = model.add_gaussian_quantity(&mean, &std_dev).unwrap();

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

#[test]
fn capability_trait_works_on_both_clients() {
    // General client: the trait is implemented, so the adapter accepts it.
    let general = ModelClient::<B>::new(Default::default());
    let mean1 = general.add_constant(EuclideanD(3), &VectorD::new(&[0.0, 0.0, 0.0]));
    let _x1 = add_unit_gaussian(&general, &mean1);

    // Restricted Gaussian client: also accepted by the same adapter.
    let gaussian = GaussianModelClient::<B>::new(Default::default());
    let mean2 = gaussian.add_constant(EuclideanD(3), &VectorD::new(&[1.0, 2.0, 3.0]));
    let x2 = add_unit_gaussian(&gaussian, &mean2);

    // The restricted client can still observe / evaluate via `ModelClientOps`.
    gaussian
        .observe(&x2, &VectorD::new(&[1.0, 2.0, 3.0]))
        .unwrap();
    let ld = gaussian.log_density(&HashMap::new()).unwrap();
    assert_eq!(ld.dims(), [1, 1]);
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
