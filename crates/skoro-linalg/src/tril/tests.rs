use approx::assert_abs_diff_eq;
use skoro_backend::{NdArrayBackend, Tensor};
use skoro_graph::point_map::{points, tangent_frames};
use skoro_model::model::ModelClient;
use skoro_riemann::euclidean::Euclidean;

use crate::tril::Tril;

use super::TrilOps;

type B = NdArrayBackend<f32, i32>;

fn into_f32_vec<const D: usize>(tensor: Tensor<B, D>) -> Vec<f32> {
    tensor.into_data().to_vec::<f32>().unwrap()
}

#[test]
pub fn vector_product() {
    let model = ModelClient::<B>::default();

    let l = model.add_parameter(Tril(2));
    let v = model.add_parameter(Euclidean::<2>);
    let lv = l.vector_product(&v).unwrap();

    // ForwardOnly
    let points = points()
        .with(
            l.id,
            Tensor::from_floats([[1.0, 2.0, 3.0]], &Default::default()),
        )
        .with(v.id, Tensor::from_floats([[4.0, 5.0]], &Default::default()));

    let eval = lv.evaluate(&points).unwrap();
    assert_abs_diff_eq!(into_f32_vec(eval).as_slice(), [4.0, 23.0].as_slice());

    // JVP
    let tangents = tangent_frames()
        .with(
            l.id,
            Tensor::<_, 1>::from_floats([0.1, 0.2, 0.3], &Default::default()).reshape([1, 3, 1]),
        )
        .with(
            v.id,
            Tensor::<_, 1>::from_floats([0.4, 0.5], &Default::default()).reshape([1, 2, 1]),
        );

    let jvp = lv.evaluate_jvp(&points, &tangents).unwrap();
    assert_abs_diff_eq!(into_f32_vec(jvp.value).as_slice(), [4.0, 23.0].as_slice());
    assert_abs_diff_eq!(
        into_f32_vec(jvp.tangents.get(&lv.id).unwrap().clone()).as_slice(),
        [0.8, 4.6].as_slice(),
        epsilon = 1e-6
    );

    // VJP
    let vjp = lv.evaluate_ord1(&points).unwrap();
    assert_abs_diff_eq!(into_f32_vec(vjp.value).as_slice(), [4.0, 23.0].as_slice());
    assert_abs_diff_eq!(
        into_f32_vec(vjp.cotangents.get(&l.id).unwrap().clone()).as_slice(),
        [4.0, 4.0, 5.0].as_slice(),
        epsilon = 1e-6
    );
    assert_abs_diff_eq!(
        into_f32_vec(vjp.cotangents.get(&v.id).unwrap().clone()).as_slice(),
        [3.0, 3.0].as_slice(),
        epsilon = 1e-6
    );

    // HVP
    let tangents = tangents
        .build()
        .into_iter()
        .map(|(id, frame)| (id, frame.squeeze_dim(2)))
        .collect();
    let hvp = lv.evaluate_ord2(&points, &tangents).unwrap();
    assert_abs_diff_eq!(into_f32_vec(hvp.value).as_slice(), [4.0, 23.0].as_slice());
    assert_abs_diff_eq!(
        into_f32_vec(hvp.tangents.get(&lv.id).unwrap().clone()).as_slice(),
        [0.8, 4.6].as_slice(),
        epsilon = 1e-6
    );
    assert_abs_diff_eq!(
        into_f32_vec(hvp.ord1_cotangents.get(&l.id).unwrap().clone()).as_slice(),
        [4.0, 4.0, 5.0].as_slice(),
        epsilon = 1e-6
    );
    assert_abs_diff_eq!(
        into_f32_vec(hvp.ord1_cotangents.get(&v.id).unwrap().clone()).as_slice(),
        [3.0, 3.0].as_slice(),
        epsilon = 1e-6
    );

    // hd_L = pack(omega * dv^T) = pack([[1.0, 1.0]]^T * [[0.4, 0.5]]) = pack([[0.4, 0.5], [0.4, 0.5]])
    // L11=0.4, L21=0.4, L22=0.5 => [0.4, 0.4, 0.5]
    assert_abs_diff_eq!(
        into_f32_vec(hvp.ord2_cotangents.get(&l.id).unwrap().clone()).as_slice(),
        [0.4, 0.4, 0.5].as_slice(),
        epsilon = 1e-6
    );
    // hd_v = dL^T * omega = [[0.1, 0.2], [0.0, 0.3]] * [1.0, 1.0]^T = [0.3, 0.3]
    assert_abs_diff_eq!(
        into_f32_vec(hvp.ord2_cotangents.get(&v.id).unwrap().clone()).as_slice(),
        [0.3, 0.3].as_slice(),
        epsilon = 1e-6
    );
}

#[test]
fn invert() {
    let model = ModelClient::<B>::default();

    let l = model.add_parameter(Tril(2));
    let linv = l.invert();

    // ForwardOnly
    let points = points().with(
        l.id,
        Tensor::from_floats([[1.0, 2.0, 3.0]], &Default::default()),
    );
    let eval = linv.evaluate(&points).unwrap();
    assert_abs_diff_eq!(
        into_f32_vec(eval).as_slice(),
        [1.0, -2.0 / 3.0, 1.0 / 3.0].as_slice()
    );

    // JVP
    let tangents = tangent_frames().with(
        l.id,
        Tensor::<_, 1>::from_floats([0.2, 0.1, 0.4], &Default::default()).reshape([1, 3, 1]),
    );
    let jvp = linv.evaluate_jvp(&points, &tangents).unwrap();
    assert_abs_diff_eq!(
        into_f32_vec(jvp.value).as_slice(),
        [1.0, -2.0 / 3.0, 1.0 / 3.0].as_slice()
    );
    assert_abs_diff_eq!(
        into_f32_vec(jvp.tangents.get(&linv.id).unwrap().clone()).as_slice(),
        [-0.2, 17.0 / 90.0, -2.0 / 45.0].as_slice(),
        epsilon = 1e-6
    );

    // VJP
    let vjp = linv.evaluate_ord1(&points).unwrap();
    assert_abs_diff_eq!(
        into_f32_vec(vjp.value).as_slice(),
        [1.0, -2.0 / 3.0, 1.0 / 3.0].as_slice()
    );
    assert_abs_diff_eq!(
        into_f32_vec(vjp.cotangents.get(&l.id).unwrap().clone()).as_slice(),
        [-1.0, 5.0 / 9.0, -1.0 / 9.0].as_slice(),
        epsilon = 1e-6
    );

    // HVP
    let tangents = tangents
        .build()
        .into_iter()
        .map(|(id, frame)| (id, frame.squeeze_dim(2)))
        .collect();
    let hvp = linv.evaluate_ord2(&points, &tangents).unwrap();
    assert_abs_diff_eq!(
        into_f32_vec(hvp.value).as_slice(),
        [1.0, -2.0 / 3.0, 1.0 / 3.0].as_slice()
    );
    assert_abs_diff_eq!(
        into_f32_vec(hvp.tangents.get(&linv.id).unwrap().clone()).as_slice(),
        [-0.2, 17.0 / 90.0, -2.0 / 45.0].as_slice(),
        epsilon = 1e-6
    );
    assert_abs_diff_eq!(
        into_f32_vec(hvp.ord1_cotangents.get(&l.id).unwrap().clone()).as_slice(),
        [-1.0, 5.0 / 9.0, -1.0 / 9.0].as_slice(),
        epsilon = 1e-6
    );
    assert_abs_diff_eq!(
        into_f32_vec(hvp.ord2_cotangents.get(&l.id).unwrap().clone()).as_slice(),
        [-0.4, 41.0 / 135.0, -4.0 / 135.0].as_slice(),
        epsilon = 1e-6
    );
}

#[test]
fn diag() {
    let model = ModelClient::<B>::default();

    let l = model.add_parameter(Tril(2));
    let d = l.diag();

    // ForwardOnly
    let points = points().with(
        l.id,
        Tensor::from_floats([[1.0, 2.0, 3.0]], &Default::default()),
    );
    let eval = d.evaluate(&points).unwrap();
    assert_abs_diff_eq!(into_f32_vec(eval).as_slice(), [1.0, 3.0].as_slice());

    // Jvp
    let tangents = tangent_frames().with(
        l.id,
        Tensor::<_, 1>::from_floats([0.2, 0.1, 0.4], &Default::default()).reshape([1, 3, 1]),
    );
    let jvp = d.evaluate_jvp(&points, &tangents).unwrap();
    assert_abs_diff_eq!(into_f32_vec(jvp.value).as_slice(), [1.0, 3.0].as_slice());
    assert_abs_diff_eq!(
        into_f32_vec(jvp.tangents.get(&d.id).unwrap().clone()).as_slice(),
        [0.2, 0.4].as_slice(),
        epsilon = 1e-6
    );

    // Vjp
    let vjp = d.evaluate_ord1(&points).unwrap();
    assert_abs_diff_eq!(into_f32_vec(vjp.value).as_slice(), [1.0, 3.0].as_slice());
    assert_abs_diff_eq!(
        into_f32_vec(vjp.cotangents.get(&l.id).unwrap().clone()).as_slice(),
        [1.0, 0.0, 1.0].as_slice(),
        epsilon = 1e-6
    );

    // Hvp
    let tangents = tangents
        .build()
        .into_iter()
        .map(|(id, frame)| (id, frame.squeeze_dim(2)))
        .collect();
    let hvp = d.evaluate_ord2(&points, &tangents).unwrap();
    assert_abs_diff_eq!(into_f32_vec(hvp.value).as_slice(), [1.0, 3.0].as_slice());
    assert_abs_diff_eq!(
        into_f32_vec(hvp.tangents.get(&d.id).unwrap().clone()).as_slice(),
        [0.2, 0.4].as_slice(),
        epsilon = 1e-6
    );
    assert_abs_diff_eq!(
        into_f32_vec(hvp.ord1_cotangents.get(&l.id).unwrap().clone()).as_slice(),
        [1.0, 0.0, 1.0].as_slice(),
        epsilon = 1e-6
    );
    assert_abs_diff_eq!(
        into_f32_vec(hvp.ord2_cotangents.get(&l.id).unwrap().clone()).as_slice(),
        [0.0, 0.0, 0.0].as_slice(),
        epsilon = 1e-6
    );
}
