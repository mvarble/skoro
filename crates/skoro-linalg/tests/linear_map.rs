use skoro_backend::{NdArrayBackend, Tensor, TensorData};
use skoro_core::error::SkoroError;
use skoro_linalg::matrix::{FixedLinearMap, linear_map};
use skoro_model::{
    ModelClientOps,
    hash_map_builders::points,
    jacobian::{jacobian_jvp, jacobian_vjp},
    model::ModelClient,
    test_utils::{into_vec, test_hvp_numerically, test_jvp_numerically, test_vjp_numerically},
};
use skoro_riemann::euclidean::EuclideanD;

type B = NdArrayBackend<f64, i32>;

fn row(values: &[f64]) -> Tensor<B, 2> {
    let n = values.len();
    Tensor::<B, 1>::from_data(TensorData::new(values.to_vec(), [n]), &Default::default())
        .reshape([1, n])
}

/// A `[batch, n]` tensor from row-major host data, for exercising `FixedLinearMap`'s batched
/// broadcast path (`batched`/`batched_transpose`), which a batch-1 point leaves untested.
fn rows(values: &[f64], batch: usize, n: usize) -> Tensor<B, 2> {
    Tensor::<B, 2>::from_data(
        TensorData::new(values.to_vec(), [batch, n]),
        &Default::default(),
    )
}

/// `K = [[1, 2], [3, 4], [5, 6]]`, row-major, `3x2`.
fn k() -> FixedLinearMap<B> {
    FixedLinearMap::from_row_major(3, 2, &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &Default::default())
}

#[test]
fn forward_matches_the_host_product() {
    let model = ModelClient::<B>::default();
    let x = model.add_parameter(EuclideanD(2));
    let y = linear_map(&x, EuclideanD(3), k()).unwrap();

    let values = points().with_quantity(&x, row(&[1.0, 1.0])).build();
    let out = into_vec(model.evaluate(&values, &y).unwrap());
    // K @ [1,1]^T = [3, 7, 11]
    assert_eq!(out, vec![3.0, 7.0, 11.0]);
}

#[test]
fn jacobian_is_the_matrix() {
    let model = ModelClient::<B>::default();
    let x = model.add_parameter(EuclideanD(2));
    let y = linear_map(&x, EuclideanD(3), k()).unwrap();

    let values = points().with_quantity(&x, row(&[1.0, 1.0])).build();
    let inputs = model.tangent_layout(&[x.id()]).unwrap();
    let outputs = model.tangent_layout(&[y.id()]).unwrap();

    let jvp = into_vec(
        jacobian_jvp(&model, &values, &inputs, &outputs)
            .unwrap()
            .reshape([3, 2]),
    );
    let vjp = into_vec(
        jacobian_vjp(&model, &values, &inputs, &outputs)
            .unwrap()
            .reshape([3, 2]),
    );
    let expected = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    for (got, want) in [(&jvp, &expected), (&vjp, &expected)] {
        for (g, w) in got.iter().zip(want) {
            approx::assert_abs_diff_eq!(g, w, epsilon = 1e-12);
        }
    }
}

#[test]
fn hvp_is_zero() {
    let model = ModelClient::<B>::default();
    let x = model.add_parameter(EuclideanD(2));
    let y = linear_map(&x, EuclideanD(3), k()).unwrap();

    let values = points().with_quantity(&x, row(&[1.0, 1.0])).build();
    let tangents = [(x.id(), row(&[1.0, 0.0]))].into_iter().collect();
    let ord1 = [(y.id(), Tensor::<B, 3>::ones([1, 3, 1], &Default::default()))]
        .into_iter()
        .collect();
    // The order-2 seed (`β` in the graph's Leibniz composition, `hvp(...) + vjp(state, β)`) must
    // be zero at the requested node: nothing further downstream contributes a second-order
    // cotangent into `y`, matching the convention in `hvp_diff_gating.rs`. A nonzero `β` here
    // would flow straight through `vjp` (K^T·β) and swamp the genuinely-zero Hessian term.
    let ord2 = [(
        y.id(),
        Tensor::<B, 3>::zeros([1, 3, 1], &Default::default()),
    )]
    .into_iter()
    .collect();
    let out = model
        .evaluate_hvp(&values, &tangents, &ord1, &ord2, &y)
        .unwrap();
    // No second-order cotangent should land on x: a linear map has zero Hessian.
    assert!(
        out.ord2_cotangents
            .get(&x.id())
            .is_none_or(|t| into_vec(t.clone()).iter().all(|v| *v == 0.0))
    );
}

#[test]
fn shape_mismatch_is_reported() {
    let model = ModelClient::<B>::default();
    let x = model.add_parameter(EuclideanD(2));
    // k() is 3x2; a 4-wide output needs 4 rows, and the check reports rows first.
    assert!(matches!(
        linear_map(&x, EuclideanD(4), k()),
        Err(SkoroError::ShapeMismatch {
            expected: 4,
            found: 3
        })
    ));
}

#[test]
fn derivatives_agree_with_finite_differences() {
    let model = ModelClient::<B>::default();
    let x = model.add_parameter(EuclideanD(2));
    let y = linear_map(&x, EuclideanD(3), k()).unwrap();
    let point = points().with_quantity(&x, row(&[0.3, -1.2])).build();
    test_jvp_numerically(&x, &y, point.clone(), 1e-6, 1e-8);
    test_vjp_numerically(&x, &y, point.clone(), 1e-6, 1e-8);
    test_hvp_numerically(&x, &y, point, 1e-6, 1e-8);
}

/// Same check at `batch = 2`. `derivatives_agree_with_finite_differences` only ever builds a
/// batch-1 point, so `repeat_dim(0, 1)` inside `batched`/`batched_transpose` is a no-op and never
/// exercises the broadcast those helpers exist for; a wrong batch axis there would still give a
/// correct forward value (the forward pass uses a plain, unbatched matmul) and only show up in
/// the derivatives at batch > 1.
#[test]
fn derivatives_agree_with_finite_differences_batched() {
    let model = ModelClient::<B>::default();
    let x = model.add_parameter(EuclideanD(2));
    let y = linear_map(&x, EuclideanD(3), k()).unwrap();
    let point = points()
        .with_quantity(&x, rows(&[0.3, -1.2, 1.7, 0.4], 2, 2))
        .build();
    test_jvp_numerically(&x, &y, point.clone(), 1e-6, 1e-8);
    test_vjp_numerically(&x, &y, point.clone(), 1e-6, 1e-8);
    test_hvp_numerically(&x, &y, point, 1e-6, 1e-8);
}
