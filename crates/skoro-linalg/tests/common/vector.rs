use approx::assert_abs_diff_eq;
use skoro_backend::{Backend, ElementConversion};
use skoro_model::{
    hash_map_builders::{points, tangent_frames},
    model::ModelClient,
    test_utils::{into_vec, tensor},
};
use skoro_operation::Input;
use skoro_riemann::euclidean::Euclidean;

use skoro_linalg::vector::{CrossProductBackend, CrossProductOps, LinalgVectorOps};

pub fn norm_squared_impl<B: Backend>() {
    let model = ModelClient::<B>::default();

    let v = model.add_parameter(Euclidean::<3>);
    let f = v.norm_squared().unwrap();

    // ForwardOnly
    let points = points().with_tensor(
        v.id(),
        Input::<B>::Float(tensor::<B, 2>(
            [[1.0f64, 2.0f64, 3.0f64]],
            &Default::default(),
        )),
    );

    let eval = f.evaluate(&points).unwrap();
    assert_abs_diff_eq!(
        into_vec(eval).as_slice(),
        into_vec(tensor::<B, 1>([14.0f64], &Default::default())).as_slice(),
        epsilon = f64::from_elem(1e-4f64),
    );

    // JVP
    let tangents = tangent_frames().with_tensor(
        v.id(),
        tensor::<B, 1>([0.1f64, 0.2f64, 0.3f64], &Default::default()).reshape([1, 3, 1]),
    );

    let jvp = f.evaluate_jvp(&points, &tangents).unwrap();
    assert_abs_diff_eq!(
        into_vec(jvp.value).as_slice(),
        into_vec(tensor::<B, 1>([14.0f64], &Default::default())).as_slice(),
        epsilon = f64::from_elem(1e-4f64),
    );
    assert_abs_diff_eq!(
        into_vec(jvp.tangents.get(&f.id()).unwrap().clone()).as_slice(),
        into_vec(tensor::<B, 1>([2.8f64], &Default::default())).as_slice(),
        epsilon = f64::from_elem(1e-4f64),
    );

    // VJP
    let vjp = f.evaluate_ord1(&points).unwrap();
    assert_abs_diff_eq!(
        into_vec(vjp.value).as_slice(),
        into_vec(tensor::<B, 1>([14.0f64], &Default::default())).as_slice(),
        epsilon = f64::from_elem(1e-4f64),
    );
    assert_abs_diff_eq!(
        into_vec(vjp.cotangents.get(&v.id()).unwrap().clone()).as_slice(),
        into_vec(tensor::<B, 1>(
            [2.0f64, 4.0f64, 6.0f64],
            &Default::default()
        ))
        .as_slice(),
        epsilon = f64::from_elem(1e-4f64),
    );

    // HVP
    let tangents = tangents
        .build()
        .into_iter()
        .map(|(id, frame)| (id, frame.squeeze_dim(2)))
        .collect();
    let hvp = f.evaluate_ord2(&points, &tangents).unwrap();
    assert_abs_diff_eq!(
        into_vec(hvp.value).as_slice(),
        into_vec(tensor::<B, 1>([14.0f64], &Default::default())).as_slice(),
        epsilon = f64::from_elem(1e-4f64),
    );
    assert_abs_diff_eq!(
        into_vec(hvp.tangents.get(&f.id()).unwrap().clone()).as_slice(),
        into_vec(tensor::<B, 1>([2.8f64], &Default::default())).as_slice(),
        epsilon = f64::from_elem(1e-4f64),
    );
    assert_abs_diff_eq!(
        into_vec(hvp.ord1_cotangents.get(&v.id()).unwrap().clone()).as_slice(),
        into_vec(tensor::<B, 1>(
            [2.0f64, 4.0f64, 6.0f64],
            &Default::default()
        ))
        .as_slice(),
        epsilon = f64::from_elem(1e-4f64),
    );
    assert_abs_diff_eq!(
        into_vec(hvp.ord2_cotangents.get(&v.id()).unwrap().clone()).as_slice(),
        into_vec(tensor::<B, 1>(
            [0.2f64, 0.4f64, 0.6f64],
            &Default::default()
        ))
        .as_slice(),
        epsilon = f64::from_elem(1e-4f64),
    );
}

pub fn norm_impl<B: Backend>() {
    let model = ModelClient::<B>::default();

    let v = model.add_parameter(Euclidean::<3>);
    let f = v.norm().unwrap();

    // ForwardOnly
    let points = points().with_tensor(
        v.id(),
        Input::<B>::Float(tensor::<B, 2>(
            [[1.0f64, 2.0f64, 3.0f64]],
            &Default::default(),
        )),
    );

    let sqrt14 = 14.0f64.sqrt();
    let eval = f.evaluate(&points).unwrap();
    assert_abs_diff_eq!(
        into_vec(eval).as_slice(),
        into_vec(tensor::<B, 1>([sqrt14], &Default::default())).as_slice(),
        epsilon = f64::from_elem(1e-4f64),
    );

    // JVP
    let tangents = tangent_frames().with_tensor(
        v.id(),
        tensor::<B, 1>([0.1f64, 0.2f64, 0.3f64], &Default::default()).reshape([1, 3, 1]),
    );

    let jvp = f.evaluate_jvp(&points, &tangents).unwrap();
    assert_abs_diff_eq!(
        into_vec(jvp.value).as_slice(),
        into_vec(tensor::<B, 1>([sqrt14], &Default::default())).as_slice(),
        epsilon = f64::from_elem(1e-4f64),
    );
    assert_abs_diff_eq!(
        into_vec(jvp.tangents.get(&f.id()).unwrap().clone()).as_slice(),
        into_vec(tensor::<B, 1>([1.4f64 / sqrt14], &Default::default())).as_slice(),
        epsilon = f64::from_elem(1e-4f64),
    );

    // VJP
    let vjp = f.evaluate_ord1(&points).unwrap();
    assert_abs_diff_eq!(
        into_vec(vjp.value).as_slice(),
        into_vec(tensor::<B, 1>([sqrt14], &Default::default())).as_slice(),
        epsilon = f64::from_elem(1e-4f64),
    );
    assert_abs_diff_eq!(
        into_vec(vjp.cotangents.get(&v.id()).unwrap().clone()).as_slice(),
        into_vec(tensor::<B, 1>(
            [1.0f64 / sqrt14, 2.0f64 / sqrt14, 3.0f64 / sqrt14],
            &Default::default()
        ))
        .as_slice(),
        epsilon = f64::from_elem(1e-4f64),
    );

    // HVP
    let tangents = tangents
        .build()
        .into_iter()
        .map(|(id, frame)| (id, frame.squeeze_dim(2)))
        .collect();
    let hvp = f.evaluate_ord2(&points, &tangents).unwrap();
    assert_abs_diff_eq!(
        into_vec(hvp.value).as_slice(),
        into_vec(tensor::<B, 1>([sqrt14], &Default::default())).as_slice(),
        epsilon = f64::from_elem(1e-4f64),
    );
    assert_abs_diff_eq!(
        into_vec(hvp.tangents.get(&f.id()).unwrap().clone()).as_slice(),
        into_vec(tensor::<B, 1>([1.4f64 / sqrt14], &Default::default())).as_slice(),
        epsilon = f64::from_elem(1e-4f64),
    );
    assert_abs_diff_eq!(
        into_vec(hvp.ord1_cotangents.get(&v.id()).unwrap().clone()).as_slice(),
        into_vec(tensor::<B, 1>(
            [1.0f64 / sqrt14, 2.0f64 / sqrt14, 3.0f64 / sqrt14],
            &Default::default()
        ))
        .as_slice(),
        epsilon = f64::from_elem(1e-4f64),
    );
    assert_abs_diff_eq!(
        into_vec(hvp.ord2_cotangents.get(&v.id()).unwrap().clone()).as_slice(),
        into_vec(tensor::<B, 1>(
            [0.0f64, 0.0f64, 0.0f64],
            &Default::default()
        ))
        .as_slice(),
        epsilon = f64::from_elem(1e-4f64),
    );
}

pub fn sum_impl<B: Backend>() {
    let model = ModelClient::<B>::default();

    let v = model.add_parameter(Euclidean::<3>);
    let s = v.sum().unwrap();

    // ForwardOnly
    let points = points().with_tensor(
        v.id(),
        tensor([[1.0f64, 2.0f64, 3.0f64]], &Default::default()),
    );

    let eval = s.evaluate(&points).unwrap();
    assert_abs_diff_eq!(
        into_vec(eval).as_slice(),
        into_vec(tensor::<B, 1>([6.0f64], &Default::default())).as_slice(),
        epsilon = f64::from_elem(1e-4f64)
    );

    // JVP
    let tangents = tangent_frames().with_tensor(
        v.id(),
        tensor::<B, 1>([0.1f64, 0.2f64, 0.3f64], &Default::default()).reshape([1, 3, 1]),
    );

    let jvp = s.evaluate_jvp(&points, &tangents).unwrap();
    assert_abs_diff_eq!(
        into_vec(jvp.value).as_slice(),
        into_vec(tensor::<B, 1>([6.0f64], &Default::default())).as_slice(),
        epsilon = f64::from_elem(1e-4f64),
    );
    assert_abs_diff_eq!(
        into_vec(jvp.tangents.get(&s.id()).unwrap().clone()).as_slice(),
        into_vec(tensor::<B, 1>([0.6f64], &Default::default())).as_slice(),
        epsilon = f64::from_elem(1e-4f64),
    );

    // VJP
    let vjp = s.evaluate_ord1(&points).unwrap();
    assert_abs_diff_eq!(
        into_vec(vjp.value).as_slice(),
        into_vec(tensor::<B, 1>([6.0f64], &Default::default())).as_slice(),
        epsilon = f64::from_elem(1e-4f64),
    );
    assert_abs_diff_eq!(
        into_vec(vjp.cotangents.get(&v.id()).unwrap().clone()).as_slice(),
        into_vec(tensor::<B, 1>(
            [1.0f64, 1.0f64, 1.0f64],
            &Default::default()
        ))
        .as_slice(),
        epsilon = f64::from_elem(1e-4f64),
    );

    // HVP
    let tangents = tangents
        .build()
        .into_iter()
        .map(|(id, frame)| (id, frame.squeeze_dim(2)))
        .collect();
    let hvp = s.evaluate_ord2(&points, &tangents).unwrap();
    assert_abs_diff_eq!(
        into_vec(hvp.value).as_slice(),
        into_vec(tensor::<B, 1>([6.0f64], &Default::default())).as_slice(),
        epsilon = f64::from_elem(1e-4f64),
    );
    assert_abs_diff_eq!(
        into_vec(hvp.tangents.get(&s.id()).unwrap().clone()).as_slice(),
        into_vec(tensor::<B, 1>([0.6f64], &Default::default())).as_slice(),
        epsilon = f64::from_elem(1e-4f64),
    );
    assert_abs_diff_eq!(
        into_vec(hvp.ord1_cotangents.get(&v.id()).unwrap().clone()).as_slice(),
        into_vec(tensor::<B, 1>(
            [1.0f64, 1.0f64, 1.0f64],
            &Default::default()
        ))
        .as_slice(),
        epsilon = f64::from_elem(1e-4f64),
    );
    assert_abs_diff_eq!(
        into_vec(hvp.ord2_cotangents.get(&v.id()).unwrap().clone()).as_slice(),
        into_vec(tensor::<B, 1>(
            [0.0f64, 0.0f64, 0.0f64],
            &Default::default()
        ))
        .as_slice(),
        epsilon = f64::from_elem(1e-4f64),
    );
}

pub fn cross_product_impl<B: Backend + CrossProductBackend>() {
    let model = ModelClient::<B>::default();

    let a = model.add_parameter(Euclidean::<3>);
    let b = model.add_parameter(Euclidean::<3>);
    let c = a.cross_product(&b).unwrap();

    let points = points()
        .with_tensor(
            a.id(),
            Input::<B>::Float(tensor::<B, 2>(
                [[1.0f64, 0.0f64, 0.0f64]],
                &Default::default(),
            )),
        )
        .with_tensor(
            b.id(),
            Input::<B>::Float(tensor::<B, 2>(
                [[0.0f64, 1.0f64, 0.0f64]],
                &Default::default(),
            )),
        );

    // e_x x e_y = e_z
    let eval = c.evaluate(&points).unwrap();
    assert_abs_diff_eq!(
        into_vec(eval).as_slice(),
        into_vec(tensor::<B, 1>(
            [0.0f64, 0.0f64, 1.0f64],
            &Default::default()
        ))
        .as_slice(),
        epsilon = f64::from_elem(1e-4f64),
    );
}
