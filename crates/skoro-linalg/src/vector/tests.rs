use approx::assert_abs_diff_eq;
use skoro_backend::{NdArrayBackend, Tensor};
use skoro_graph::point_map::{points, tangent_frames};
use skoro_model::model::ModelClient;
use skoro_riemann::euclidean::Euclidean;

use super::{CrossProductOps, LinalgVectorOps};

type B = NdArrayBackend<f32, i32>;

fn into_f32_vec<const D: usize>(tensor: Tensor<B, D>) -> Vec<f32> {
    tensor.into_data().to_vec::<f32>().unwrap()
}

#[test]
pub fn norm_squared() {
    let model = ModelClient::<B>::default();

    let v = model.add_parameter(Euclidean::<3>);
    let f = v.norm_squared();

    // ForwardOnly
    let points = points().with(
        v.id,
        Tensor::from_floats([[1.0, 2.0, 3.0]], &Default::default()),
    );

    let eval = f.evaluate(&points).unwrap();
    assert_abs_diff_eq!(
        into_f32_vec(eval).as_slice(),
        [14.0].as_slice(),
        epsilon = 1e-6
    );

    // JVP
    let tangents = tangent_frames().with(
        v.id,
        Tensor::<_, 1>::from_floats([0.1, 0.2, 0.3], &Default::default()).reshape([1, 3, 1]),
    );

    let jvp = f.evaluate_jvp(&points, &tangents).unwrap();
    assert_abs_diff_eq!(
        into_f32_vec(jvp.value).as_slice(),
        [14.0].as_slice(),
        epsilon = 1e-6
    );
    assert_abs_diff_eq!(
        into_f32_vec(jvp.tangents.get(&f.id).unwrap().clone()).as_slice(),
        [2.8].as_slice(),
        epsilon = 1e-6
    );

    // VJP
    let vjp = f.evaluate_ord1(&points).unwrap();
    assert_abs_diff_eq!(
        into_f32_vec(vjp.value).as_slice(),
        [14.0].as_slice(),
        epsilon = 1e-6
    );
    assert_abs_diff_eq!(
        into_f32_vec(vjp.cotangents.get(&v.id).unwrap().clone()).as_slice(),
        [2.0, 4.0, 6.0].as_slice(),
        epsilon = 1e-6
    );

    // HVP
    let tangents = tangents
        .build()
        .into_iter()
        .map(|(id, frame)| (id, frame.squeeze_dim(2)))
        .collect();
    let hvp = f.evaluate_ord2(&points, &tangents).unwrap();
    assert_abs_diff_eq!(
        into_f32_vec(hvp.value).as_slice(),
        [14.0].as_slice(),
        epsilon = 1e-6
    );
    assert_abs_diff_eq!(
        into_f32_vec(hvp.tangents.get(&f.id).unwrap().clone()).as_slice(),
        [2.8].as_slice(),
        epsilon = 1e-6
    );
    assert_abs_diff_eq!(
        into_f32_vec(hvp.ord1_cotangents.get(&v.id).unwrap().clone()).as_slice(),
        [2.0, 4.0, 6.0].as_slice(),
        epsilon = 1e-6
    );
    assert_abs_diff_eq!(
        into_f32_vec(hvp.ord2_cotangents.get(&v.id).unwrap().clone()).as_slice(),
        [0.2, 0.4, 0.6].as_slice(),
        epsilon = 1e-6
    );
}

#[test]
pub fn norm() {
    let model = ModelClient::<B>::default();

    let v = model.add_parameter(Euclidean::<3>);
    let f = v.norm();

    // ForwardOnly
    let points = points().with(
        v.id,
        Tensor::from_floats([[1.0, 2.0, 3.0]], &Default::default()),
    );

    let sqrt14 = 14.0f32.sqrt();
    let eval = f.evaluate(&points).unwrap();
    assert_abs_diff_eq!(
        into_f32_vec(eval).as_slice(),
        [sqrt14].as_slice(),
        epsilon = 1e-6
    );

    // JVP
    let tangents = tangent_frames().with(
        v.id,
        Tensor::<_, 1>::from_floats([0.1, 0.2, 0.3], &Default::default()).reshape([1, 3, 1]),
    );

    let jvp = f.evaluate_jvp(&points, &tangents).unwrap();
    assert_abs_diff_eq!(
        into_f32_vec(jvp.value).as_slice(),
        [sqrt14].as_slice(),
        epsilon = 1e-6
    );
    assert_abs_diff_eq!(
        into_f32_vec(jvp.tangents.get(&f.id).unwrap().clone()).as_slice(),
        [1.4 / sqrt14].as_slice(),
        epsilon = 1e-6
    );

    // VJP
    let vjp = f.evaluate_ord1(&points).unwrap();
    assert_abs_diff_eq!(
        into_f32_vec(vjp.value).as_slice(),
        [sqrt14].as_slice(),
        epsilon = 1e-6
    );
    assert_abs_diff_eq!(
        into_f32_vec(vjp.cotangents.get(&v.id).unwrap().clone()).as_slice(),
        [1.0 / sqrt14, 2.0 / sqrt14, 3.0 / sqrt14].as_slice(),
        epsilon = 1e-6
    );

    // HVP
    let tangents = tangents
        .build()
        .into_iter()
        .map(|(id, frame)| (id, frame.squeeze_dim(2)))
        .collect();
    let hvp = f.evaluate_ord2(&points, &tangents).unwrap();
    assert_abs_diff_eq!(
        into_f32_vec(hvp.value).as_slice(),
        [sqrt14].as_slice(),
        epsilon = 1e-6
    );
    assert_abs_diff_eq!(
        into_f32_vec(hvp.tangents.get(&f.id).unwrap().clone()).as_slice(),
        [1.4 / sqrt14].as_slice(),
        epsilon = 1e-6
    );
    assert_abs_diff_eq!(
        into_f32_vec(hvp.ord1_cotangents.get(&v.id).unwrap().clone()).as_slice(),
        [1.0 / sqrt14, 2.0 / sqrt14, 3.0 / sqrt14].as_slice(),
        epsilon = 1e-6
    );
    assert_abs_diff_eq!(
        into_f32_vec(hvp.ord2_cotangents.get(&v.id).unwrap().clone()).as_slice(),
        [0.0, 0.0, 0.0].as_slice(),
        epsilon = 1e-6
    );
}

#[test]
fn sum() {
    let model = ModelClient::<B>::default();

    let v = model.add_parameter(Euclidean::<3>);
    let s = v.sum();

    // ForwardOnly
    let points = points().with(
        v.id,
        Tensor::from_floats([[1.0, 2.0, 3.0]], &Default::default()),
    );

    let eval = s.evaluate(&points).unwrap();
    assert_abs_diff_eq!(
        into_f32_vec(eval).as_slice(),
        [6.0].as_slice(),
        epsilon = 1e-6
    );

    // JVP
    let tangents = tangent_frames().with(
        v.id,
        Tensor::<_, 1>::from_floats([0.1, 0.2, 0.3], &Default::default()).reshape([1, 3, 1]),
    );

    let jvp = s.evaluate_jvp(&points, &tangents).unwrap();
    assert_abs_diff_eq!(
        into_f32_vec(jvp.value).as_slice(),
        [6.0].as_slice(),
        epsilon = 1e-6
    );
    assert_abs_diff_eq!(
        into_f32_vec(jvp.tangents.get(&s.id).unwrap().clone()).as_slice(),
        [0.6].as_slice(),
        epsilon = 1e-6
    );

    // VJP
    let vjp = s.evaluate_ord1(&points).unwrap();
    assert_abs_diff_eq!(
        into_f32_vec(vjp.value).as_slice(),
        [6.0].as_slice(),
        epsilon = 1e-6
    );
    assert_abs_diff_eq!(
        into_f32_vec(vjp.cotangents.get(&v.id).unwrap().clone()).as_slice(),
        [1.0, 1.0, 1.0].as_slice(),
        epsilon = 1e-6
    );

    // HVP
    let tangents = tangents
        .build()
        .into_iter()
        .map(|(id, frame)| (id, frame.squeeze_dim(2)))
        .collect();
    let hvp = s.evaluate_ord2(&points, &tangents).unwrap();
    assert_abs_diff_eq!(
        into_f32_vec(hvp.value).as_slice(),
        [6.0].as_slice(),
        epsilon = 1e-6
    );
    assert_abs_diff_eq!(
        into_f32_vec(hvp.tangents.get(&s.id).unwrap().clone()).as_slice(),
        [0.6].as_slice(),
        epsilon = 1e-6
    );
    assert_abs_diff_eq!(
        into_f32_vec(hvp.ord1_cotangents.get(&v.id).unwrap().clone()).as_slice(),
        [1.0, 1.0, 1.0].as_slice(),
        epsilon = 1e-6
    );
    assert_abs_diff_eq!(
        into_f32_vec(hvp.ord2_cotangents.get(&v.id).unwrap().clone()).as_slice(),
        [0.0, 0.0, 0.0].as_slice(),
        epsilon = 1e-6
    );
}

#[test]
pub fn cross_product_ndarray() {
    // Smoke test ensuring `CrossProductOps` is usable on the NdArray backend.
    let model = ModelClient::<B>::default();

    let a = model.add_parameter(Euclidean::<3>);
    let b = model.add_parameter(Euclidean::<3>);
    let c = a.cross_product(&b);

    let points = points()
        .with(
            a.id,
            Tensor::from_floats([[1.0, 0.0, 0.0]], &Default::default()),
        )
        .with(
            b.id,
            Tensor::from_floats([[0.0, 1.0, 0.0]], &Default::default()),
        );

    // e_x x e_y = e_z
    let eval = c.evaluate(&points).unwrap();
    assert_abs_diff_eq!(
        into_f32_vec(eval).as_slice(),
        [0.0, 0.0, 1.0].as_slice(),
        epsilon = 1e-6
    );
}
