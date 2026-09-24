//! `abs` pushes a tangent forward as `sign(x) · dx`.
//!
//! The fixture has non-unit tangents and a two-column frame on purpose: an implementation that
//! returned `sign(x)` alone, ignoring `dx`, passes any test seeded with unit vectors on a scalar.

use std::collections::HashMap;

use skoro_backend::{Flex, Tensor, TensorData};
use skoro_model::{hash_map_builders::points, model::ModelClient};
use skoro_pointwise::PointwiseOps;
use skoro_riemann::euclidean::EuclideanD;

type B = Flex;
skoro_backend::f64_test_binary!(Flex);

fn tensor<const D: usize>(values: Vec<f64>, shape: [usize; D]) -> Tensor<B, D> {
    Tensor::from_data(TensorData::new(values, shape), &Default::default())
}

#[test]
fn the_tangent_of_abs_is_the_sign_times_the_tangent() {
    let model = ModelClient::<B>::default();
    let x = model.add_parameter(EuclideanD(2));
    let y = x.abs().unwrap();
    let values = points()
        .with_quantity(&x, tensor(vec![2.0, -3.0], [1, 2]))
        .build();

    // JVP with a two-column frame: columns (0.5, 0.25) and (-1.5, 4.0).
    let frame = tensor(vec![0.5, -1.5, 0.25, 4.0], [1, 2, 2]);
    let jvp = model
        .evaluate_ids_jvp(&values, &HashMap::from([(x.id(), frame)]), &[y.id()])
        .unwrap();
    let got = jvp.tangents[&y.id()]
        .clone()
        .into_data()
        .to_vec::<f64>()
        .unwrap();
    assert_eq!(got, vec![0.5, -1.5, -0.25, -4.0]);

    // The HVP pass pushes one tangent vector forward the same way.
    let hvp = model
        .evaluate_ids_hvp(
            &values,
            &HashMap::from([(x.id(), tensor(vec![0.5, 0.25], [1, 2]))]),
            &HashMap::from([(y.id(), tensor(vec![1.0, 1.0], [1, 2, 1]))]),
            &HashMap::from([(y.id(), tensor(vec![0.0, 0.0], [1, 2, 1]))]),
            &[y.id()],
        )
        .unwrap();
    let got = hvp.tangents[&y.id()]
        .clone()
        .into_data()
        .to_vec::<f64>()
        .unwrap();
    assert_eq!(got, vec![0.5, -0.25]);
}
