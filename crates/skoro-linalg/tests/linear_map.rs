//! `FixedLinearMap`: a host matrix applied as a graph operation. `SparseLinearMap`'s shape check
//! lives here too; its values and derivatives are checked on every backend in `common/matrix.rs`.
//!
//! One backend for the binary (`Flex` under the `f64` float policy), and tensors built through
//! `TensorData`, so the exact comparisons below are not undone by an `f32` round trip.

use std::collections::HashMap;

use skoro_backend::{Flex, Tensor, TensorData};
use skoro_core::error::SkoroError;
use skoro_linalg::matrix::{FixedLinearMap, SparseLinearMap, linear_map, sparse_linear_map};
use skoro_model::{
    ModelClientOps,
    hash_map_builders::points,
    jacobian::{jacobian_jvp, jacobian_vjp},
    model::ModelClient,
    test_utils::{test_hvp_numerically, test_jvp_numerically, test_vjp_numerically},
};
use skoro_riemann::euclidean::EuclideanD;

type B = Flex;
skoro_backend::f64_test_binary!(Flex);

/// `K`, row-major, `3 × 2`.
const K: [f64; 6] = [0.7, -1.3, 2.1, 0.4, -0.25, 1.9];

fn k() -> FixedLinearMap<B> {
    FixedLinearMap::from_row_major(3, 2, &K, &Default::default())
}

fn tensor<const D: usize>(data: Vec<f64>, dims: [usize; D]) -> Tensor<B, D> {
    Tensor::from_data(TensorData::new(data, dims), &Default::default())
}

fn into_vec<const D: usize>(t: Tensor<B, D>) -> Vec<f64> {
    t.into_data().to_vec::<f64>().unwrap()
}

/// Two batch rows of `x`.
const X: [f64; 4] = [1.5, -0.5, -2.0, 0.75];

#[test]
fn forward_matches_the_host_product() {
    let model = ModelClient::<B>::default();
    let x = model.add_parameter(EuclideanD(2));
    let y = linear_map(&x, EuclideanD(3), k()).unwrap();

    let values = points()
        .with_quantity(&x, tensor(X.to_vec(), [2, 2]))
        .build();
    let got = y.evaluate(&values).unwrap();
    assert_eq!(got.dims(), [2, 3]);
    let got = into_vec(got);
    for batch in 0..2 {
        for row in 0..3 {
            let want: f64 = (0..2)
                .map(|col| K[row * 2 + col] * X[batch * 2 + col])
                .sum();
            assert!((got[batch * 3 + row] - want).abs() < 1e-12);
        }
    }
}

#[test]
fn jacobian_is_the_matrix() {
    let model = ModelClient::<B>::default();
    let x = model.add_parameter(EuclideanD(2));
    let y = linear_map(&x, EuclideanD(3), k()).unwrap();

    let values = points()
        .with_quantity(&x, tensor(X.to_vec(), [2, 2]))
        .build();
    let inputs = model.tangent_layout(&[x.id()]).unwrap();
    let outputs = model.tangent_layout(&[y.id()]).unwrap();

    let vjp = jacobian_vjp(&model, &values, &inputs, &outputs).unwrap();
    let jvp = jacobian_jvp(&model, &values, &inputs, &outputs).unwrap();
    assert_eq!(vjp.dims(), [2, 3, 2]);
    assert_eq!(jvp.dims(), [2, 3, 2]);
    let (vjp, jvp) = (into_vec(vjp), into_vec(jvp));
    for batch in 0..2 {
        for (entry, want) in K.iter().enumerate() {
            assert!((vjp[batch * 6 + entry] - want).abs() < 1e-12);
            assert!((jvp[batch * 6 + entry] - want).abs() < 1e-12);
        }
    }
}

#[test]
fn hvp_is_zero() {
    let model = ModelClient::<B>::default();
    let x = model.add_parameter(EuclideanD(2));
    let y = linear_map(&x, EuclideanD(3), k()).unwrap();

    let values = points()
        .with_quantity(&x, tensor(X.to_vec(), [2, 2]))
        .build();
    let tangents = HashMap::from([(x.id(), tensor(vec![0.3, -1.1, 2.0, 0.5], [2, 2]))]);
    let ord1 = HashMap::from([(
        y.id(),
        tensor(vec![1.0, -2.0, 0.5, 3.0, 0.25, -1.0], [2, 3, 1]),
    )]);
    let ord2 = HashMap::from([(
        y.id(),
        Tensor::<B, 3>::zeros([2, 3, 1], &Default::default()),
    )]);

    let out = y.evaluate_hvp(&values, &tangents, &ord1, &ord2).unwrap();
    let second = out
        .ord2_cotangents
        .get(&x.id())
        .expect("a differentiable input receives a second-order cotangent");
    assert_eq!(second.dims(), [2, 2, 1]);
    assert!(into_vec(second.clone()).iter().all(|v| *v == 0.0));

    // The first-order cotangent is still Kᵀ ω.
    let first = into_vec(out.ord1_cotangents[&x.id()].clone());
    let omega = [1.0, -2.0, 0.5, 3.0, 0.25, -1.0];
    for batch in 0..2 {
        for col in 0..2 {
            let want: f64 = (0..3)
                .map(|row| K[row * 2 + col] * omega[batch * 3 + row])
                .sum();
            assert!((first[batch * 2 + col] - want).abs() < 1e-12);
        }
    }
}

#[test]
fn derivatives_agree_with_finite_differences() {
    let model = ModelClient::<B>::default();
    let x = model.add_parameter(EuclideanD(2));
    let y = linear_map(&x, EuclideanD(3), k()).unwrap();

    let values = points()
        .with_quantity(&x, tensor(X.to_vec(), [2, 2]))
        .build();
    test_jvp_numerically(&x, &y, values.clone(), 1e-6, 1e-5);
    test_vjp_numerically(&x, &y, values.clone(), 1e-6, 1e-5);
    test_hvp_numerically(&x, &y, values, 1e-6, 1e-5);
}

#[test]
fn shape_mismatch_is_reported() {
    let model = ModelClient::<B>::default();
    let x = model.add_parameter(EuclideanD(2));
    assert!(matches!(
        linear_map(&x, EuclideanD(4), k()),
        Err(SkoroError::ShapeMismatch {
            expected: 4,
            found: 3
        })
    ));

    let wide = model.add_parameter(EuclideanD(5));
    assert!(matches!(
        linear_map(&wide, EuclideanD(3), k()),
        Err(SkoroError::ShapeMismatch {
            expected: 5,
            found: 2
        })
    ));
}

#[test]
fn sparse_shape_mismatch_is_reported() {
    let model = ModelClient::<B>::default();
    let x = model.add_parameter(EuclideanD(2));
    // A 3 × 2 map; the check reports rows first, then columns.
    let map = || SparseLinearMap::from_triplets(3, 2, [(0, 0, 1.0), (2, 1, -1.0)]);
    assert!(matches!(
        sparse_linear_map(&x, EuclideanD(4), map()),
        Err(SkoroError::ShapeMismatch {
            expected: 4,
            found: 3
        })
    ));

    let wide = model.add_parameter(EuclideanD(5));
    assert!(matches!(
        sparse_linear_map(&wide, EuclideanD(3), map()),
        Err(SkoroError::ShapeMismatch {
            expected: 5,
            found: 2
        })
    ));
}
