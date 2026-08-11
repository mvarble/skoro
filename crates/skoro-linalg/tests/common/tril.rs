use approx::assert_abs_diff_eq;
use skoro_backend::{Backend, ElementConversion};
use skoro_model::{
    hash_map_builders::{points, tangent_frames},
    model::ModelClient,
    q::Q,
    test_utils::{into_vec, tensor},
};
use skoro_operation::Input;
use skoro_riemann::euclidean::Euclidean;

use skoro_linalg::tril::{Tril, TrilOps};

pub fn vector_product_impl<B: Backend>()
where
    Q<B, Tril>: TrilOps<B>,
    B::FloatElem: approx::AbsDiffEq<B::FloatElem, Epsilon = B::FloatElem> + Copy,
{
    let model = ModelClient::<B>::default();

    let l = model.add_parameter(Tril(2));
    let v = model.add_parameter(Euclidean::<2>);
    let lv = l.vector_product(&v).unwrap();

    // ForwardOnly
    let points = points()
        .with_tensor(
            l.id(),
            Input::<B>::Float(tensor([[1.0, 2.0, 3.0]], &Default::default())),
        )
        .with_tensor(
            v.id(),
            Input::<B>::Float(tensor([[4.0, 5.0]], &Default::default())),
        );

    let eval = lv.evaluate(&points).unwrap();
    assert_abs_diff_eq!(
        into_vec(eval).as_slice(),
        into_vec(tensor::<B, 1>([4.0, 23.0], &Default::default())).as_slice(),
        epsilon = B::FloatElem::from_elem(1e-6),
    );

    // JVP
    let tangents = tangent_frames()
        .with_tensor(
            l.id(),
            tensor::<B, 1>([0.1, 0.2, 0.3], &Default::default()).reshape([1, 3, 1]),
        )
        .with_tensor(
            v.id(),
            tensor::<B, 1>([0.4, 0.5], &Default::default()).reshape([1, 2, 1]),
        );

    let jvp = lv.evaluate_jvp(&points, &tangents).unwrap();
    assert_abs_diff_eq!(
        into_vec(jvp.value).as_slice(),
        into_vec(tensor::<B, 1>([4.0, 23.0], &Default::default())).as_slice(),
        epsilon = B::FloatElem::from_elem(1e-6),
    );
    assert_abs_diff_eq!(
        into_vec(jvp.tangents.get(&lv.id()).unwrap().clone()).as_slice(),
        into_vec(tensor::<B, 1>([0.8, 4.6], &Default::default())).as_slice(),
        epsilon = B::FloatElem::from_elem(1e-6),
    );

    // VJP
    let vjp = lv.evaluate_ord1(&points).unwrap();
    assert_abs_diff_eq!(
        into_vec(vjp.value).as_slice(),
        into_vec(tensor::<B, 1>([4.0, 23.0], &Default::default())).as_slice(),
        epsilon = B::FloatElem::from_elem(1e-6),
    );
    assert_abs_diff_eq!(
        into_vec(vjp.cotangents.get(&l.id()).unwrap().clone()).as_slice(),
        into_vec(tensor::<B, 1>([4.0, 4.0, 5.0], &Default::default())).as_slice(),
        epsilon = B::FloatElem::from_elem(1e-6),
    );
    assert_abs_diff_eq!(
        into_vec(vjp.cotangents.get(&v.id()).unwrap().clone()).as_slice(),
        into_vec(tensor::<B, 1>([3.0, 3.0], &Default::default())).as_slice(),
        epsilon = B::FloatElem::from_elem(1e-6),
    );

    // HVP
    let tangents = tangents
        .build()
        .into_iter()
        .map(|(id, frame)| (id, frame.squeeze_dim(2)))
        .collect();
    let hvp = lv.evaluate_ord2(&points, &tangents).unwrap();
    assert_abs_diff_eq!(
        into_vec(hvp.value).as_slice(),
        into_vec(tensor::<B, 1>([4.0, 23.0], &Default::default())).as_slice(),
        epsilon = B::FloatElem::from_elem(1e-6),
    );
    assert_abs_diff_eq!(
        into_vec(hvp.tangents.get(&lv.id()).unwrap().clone()).as_slice(),
        into_vec(tensor::<B, 1>([0.8, 4.6], &Default::default())).as_slice(),
        epsilon = B::FloatElem::from_elem(1e-6),
    );
    assert_abs_diff_eq!(
        into_vec(hvp.ord1_cotangents.get(&l.id()).unwrap().clone()).as_slice(),
        into_vec(tensor::<B, 1>([4.0, 4.0, 5.0], &Default::default())).as_slice(),
        epsilon = B::FloatElem::from_elem(1e-6),
    );
    assert_abs_diff_eq!(
        into_vec(hvp.ord1_cotangents.get(&v.id()).unwrap().clone()).as_slice(),
        into_vec(tensor::<B, 1>([3.0, 3.0], &Default::default())).as_slice(),
        epsilon = B::FloatElem::from_elem(1e-6),
    );

    // hd_L = pack(omega * dv^T) = pack([[1.0, 1.0]]^T * [[0.4, 0.5]]) = pack([[0.4, 0.5], [0.4, 0.5]])
    // L11=0.4, L21=0.4, L22=0.5 => [0.4, 0.4, 0.5]
    assert_abs_diff_eq!(
        into_vec(hvp.ord2_cotangents.get(&l.id()).unwrap().clone()).as_slice(),
        into_vec(tensor::<B, 1>([0.4, 0.4, 0.5], &Default::default())).as_slice(),
        epsilon = B::FloatElem::from_elem(1e-6),
    );
    // hd_v = dL^T * omega = [[0.1, 0.2], [0.0, 0.3]] * [1.0, 1.0]^T = [0.3, 0.3]
    assert_abs_diff_eq!(
        into_vec(hvp.ord2_cotangents.get(&v.id()).unwrap().clone()).as_slice(),
        into_vec(tensor::<B, 1>([0.3, 0.3], &Default::default())).as_slice(),
        epsilon = B::FloatElem::from_elem(1e-6),
    );
}

pub fn invert_impl<B: Backend>()
where
    Q<B, Tril>: TrilOps<B>,
    B::FloatElem: approx::AbsDiffEq<B::FloatElem, Epsilon = B::FloatElem> + Copy,
{
    let model = ModelClient::<B>::default();

    let l = model.add_parameter(Tril(2));
    let linv = l.invert().unwrap();

    // ForwardOnly
    let points = points().with_tensor(
        l.id(),
        Input::<B>::Float(tensor([[1.0, 2.0, 3.0]], &Default::default())),
    );
    let eval = linv.evaluate(&points).unwrap();
    assert_abs_diff_eq!(
        into_vec(eval).as_slice(),
        into_vec(tensor::<B, 1>(
            [1.0, -2.0 / 3.0, 1.0 / 3.0],
            &Default::default()
        ))
        .as_slice(),
        epsilon = B::FloatElem::from_elem(1e-6),
    );

    // JVP
    let tangents = tangent_frames().with_tensor(
        l.id(),
        tensor::<B, 1>([0.2, 0.1, 0.4], &Default::default()).reshape([1, 3, 1]),
    );
    let jvp = linv.evaluate_jvp(&points, &tangents).unwrap();
    assert_abs_diff_eq!(
        into_vec(jvp.value).as_slice(),
        into_vec(tensor::<B, 1>(
            [1.0, -2.0 / 3.0, 1.0 / 3.0],
            &Default::default()
        ))
        .as_slice(),
        epsilon = B::FloatElem::from_elem(1e-6),
    );
    assert_abs_diff_eq!(
        into_vec(jvp.tangents.get(&linv.id()).unwrap().clone()).as_slice(),
        into_vec(tensor::<B, 1>(
            [-0.2, 17.0 / 90.0, -2.0 / 45.0],
            &Default::default()
        ))
        .as_slice(),
        epsilon = B::FloatElem::from_elem(1e-6),
    );

    // VJP
    let vjp = linv.evaluate_ord1(&points).unwrap();
    assert_abs_diff_eq!(
        into_vec(vjp.value).as_slice(),
        into_vec(tensor::<B, 1>(
            [1.0, -2.0 / 3.0, 1.0 / 3.0],
            &Default::default()
        ))
        .as_slice(),
        epsilon = B::FloatElem::from_elem(1e-6),
    );
    assert_abs_diff_eq!(
        into_vec(vjp.cotangents.get(&l.id()).unwrap().clone()).as_slice(),
        into_vec(tensor::<B, 1>(
            [-1.0, 5.0 / 9.0, -1.0 / 9.0],
            &Default::default()
        ))
        .as_slice(),
        epsilon = B::FloatElem::from_elem(1e-6),
    );

    // HVP
    let tangents = tangents
        .build()
        .into_iter()
        .map(|(id, frame)| (id, frame.squeeze_dim(2)))
        .collect();
    let hvp = linv.evaluate_ord2(&points, &tangents).unwrap();
    assert_abs_diff_eq!(
        into_vec(hvp.value).as_slice(),
        into_vec(tensor::<B, 1>(
            [1.0, -2.0 / 3.0, 1.0 / 3.0],
            &Default::default()
        ))
        .as_slice(),
        epsilon = B::FloatElem::from_elem(1e-6),
    );
    assert_abs_diff_eq!(
        into_vec(hvp.tangents.get(&linv.id()).unwrap().clone()).as_slice(),
        into_vec(tensor::<B, 1>(
            [-0.2, 17.0 / 90.0, -2.0 / 45.0],
            &Default::default()
        ))
        .as_slice(),
        epsilon = B::FloatElem::from_elem(1e-6),
    );
    assert_abs_diff_eq!(
        into_vec(hvp.ord1_cotangents.get(&l.id()).unwrap().clone()).as_slice(),
        into_vec(tensor::<B, 1>(
            [-1.0, 5.0 / 9.0, -1.0 / 9.0],
            &Default::default()
        ))
        .as_slice(),
        epsilon = B::FloatElem::from_elem(1e-6),
    );
    assert_abs_diff_eq!(
        into_vec(hvp.ord2_cotangents.get(&l.id()).unwrap().clone()).as_slice(),
        into_vec(tensor::<B, 1>(
            [-0.4, 41.0 / 135.0, -4.0 / 135.0],
            &Default::default()
        ))
        .as_slice(),
        epsilon = B::FloatElem::from_elem(1e-6),
    );
}

pub fn diag_impl<B: Backend>()
where
    Q<B, Tril>: TrilOps<B>,
    B::FloatElem: approx::AbsDiffEq<B::FloatElem, Epsilon = B::FloatElem> + Copy,
{
    let model = ModelClient::<B>::default();

    let l = model.add_parameter(Tril(2));
    let d = l.diag().unwrap();

    // ForwardOnly
    let points = points().with_tensor(
        l.id(),
        Input::<B>::Float(tensor([[1.0, 2.0, 3.0]], &Default::default())),
    );
    let eval = d.evaluate(&points).unwrap();
    assert_abs_diff_eq!(
        into_vec(eval).as_slice(),
        into_vec(tensor::<B, 1>([1.0, 3.0], &Default::default())).as_slice(),
        epsilon = B::FloatElem::from_elem(1e-6),
    );

    // Jvp
    let tangents = tangent_frames().with_tensor(
        l.id(),
        tensor::<B, 1>([0.2, 0.1, 0.4], &Default::default()).reshape([1, 3, 1]),
    );
    let jvp = d.evaluate_jvp(&points, &tangents).unwrap();
    assert_abs_diff_eq!(
        into_vec(jvp.value).as_slice(),
        into_vec(tensor::<B, 1>([1.0, 3.0], &Default::default())).as_slice(),
        epsilon = B::FloatElem::from_elem(1e-6),
    );
    assert_abs_diff_eq!(
        into_vec(jvp.tangents.get(&d.id()).unwrap().clone()).as_slice(),
        into_vec(tensor::<B, 1>([0.2, 0.4], &Default::default())).as_slice(),
        epsilon = B::FloatElem::from_elem(1e-6),
    );

    // Vjp
    let vjp = d.evaluate_ord1(&points).unwrap();
    assert_abs_diff_eq!(
        into_vec(vjp.value).as_slice(),
        into_vec(tensor::<B, 1>([1.0, 3.0], &Default::default())).as_slice(),
        epsilon = B::FloatElem::from_elem(1e-6),
    );
    assert_abs_diff_eq!(
        into_vec(vjp.cotangents.get(&l.id()).unwrap().clone()).as_slice(),
        into_vec(tensor::<B, 1>([1.0, 0.0, 1.0], &Default::default())).as_slice(),
        epsilon = B::FloatElem::from_elem(1e-6),
    );

    // Hvp
    let tangents = tangents
        .build()
        .into_iter()
        .map(|(id, frame)| (id, frame.squeeze_dim(2)))
        .collect();
    let hvp = d.evaluate_ord2(&points, &tangents).unwrap();
    assert_abs_diff_eq!(
        into_vec(hvp.value).as_slice(),
        into_vec(tensor::<B, 1>([1.0, 3.0], &Default::default())).as_slice(),
        epsilon = B::FloatElem::from_elem(1e-6),
    );
    assert_abs_diff_eq!(
        into_vec(hvp.tangents.get(&d.id()).unwrap().clone()).as_slice(),
        into_vec(tensor::<B, 1>([0.2, 0.4], &Default::default())).as_slice(),
        epsilon = B::FloatElem::from_elem(1e-6),
    );
    assert_abs_diff_eq!(
        into_vec(hvp.ord1_cotangents.get(&l.id()).unwrap().clone()).as_slice(),
        into_vec(tensor::<B, 1>([1.0, 0.0, 1.0], &Default::default())).as_slice(),
        epsilon = B::FloatElem::from_elem(1e-6),
    );
    assert_abs_diff_eq!(
        into_vec(hvp.ord2_cotangents.get(&l.id()).unwrap().clone()).as_slice(),
        into_vec(tensor::<B, 1>([0.0, 0.0, 0.0], &Default::default())).as_slice(),
        epsilon = B::FloatElem::from_elem(1e-6),
    );
}
