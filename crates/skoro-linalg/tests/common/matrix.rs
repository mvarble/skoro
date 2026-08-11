use skoro_backend::{Backend, ElementConversion, Tensor, TensorData};
use skoro_model::{
    hash_map_builders::points,
    model::ModelClient,
    test_utils::{into_ndarray, test_hvp_numerically, test_jvp_numerically, test_vjp_numerically},
};
use skoro_operation::Input;
use skoro_riemann::matrix::{EuclideanMatrix36, EuclideanMatrix62};

use skoro_linalg::matrix::MatmulOps;

pub fn left_broadcast_matmul_impl<B: Backend>()
where
    B::FloatElem: approx::AbsDiffEq<B::FloatElem, Epsilon = B::FloatElem> + Copy,
{
    use skoro_linalg::matrix::left_broadcast_matmul;
    // [2, 4, 2]
    let a = Tensor::<B, 3>::from_floats(
        [
            [[1.0, 2.0], [3.0, 4.0], [-8.0, 10.0], [16.0, 0.1]],
            [[-1.0, -1.0], [-1.0, 0.0], [0.0, -1.0], [0.0, 0.0]],
        ],
        &Default::default(),
    );
    // [2, 2, 2, 3]
    let b = Tensor::<B, 4>::from_floats(
        [
            [
                [[5.0, 6.0, -1.0], [-3.0, 10.0, 0.1]],
                [[-0.5, -0.3, 0.2], [1.0, 2.0, 3.0]],
            ],
            [
                [[10.0, 0.0, 5.0], [-0.45, 0.123, 0.8]],
                [[-1.0, -2.0, -3.0], [1.0, 1.0, 1.0]],
            ],
        ],
        &Default::default(),
    );
    // [2, 4, 2, 3]
    let c = left_broadcast_matmul::<B>(a.clone(), b.clone());
    assert_eq!(c.dims(), [2, 4, 2, 3]);

    for i in 0..3 {
        let mats0: Tensor<B, 3> = c.clone().slice_dim(3, i).squeeze_dim(3);
        let mats0 = into_ndarray(mats0);
        let mats0_actual = a.clone().matmul(b.clone().slice_dim(3, i).squeeze_dim(3));
        let mats0_actual = into_ndarray(mats0_actual);
        approx::assert_abs_diff_eq!(mats0, mats0_actual);
    }
}

pub fn right_broadcast_matmul_impl<B: Backend>()
where
    B::FloatElem: approx::AbsDiffEq<B::FloatElem, Epsilon = B::FloatElem> + Copy,
{
    use skoro_linalg::matrix::right_broadcast_matmul;
    // [2, 2, 4, 2]
    let a = Tensor::<B, 4>::from_floats(
        [
            [
                [[1.0, 2.0], [3.0, 4.0], [-8.0, 10.0], [16.0, 0.1]],
                [[-1.0, -1.0], [-1.0, 0.0], [0.0, -1.0], [0.0, 0.0]],
            ],
            [
                [[-1.0, -2.0], [-3.0, -4.0], [8.0, -10.0], [-16.0, -0.1]],
                [[1.0, 1.0], [1.0, 0.0], [0.0, 1.0], [0.0, 0.0]],
            ],
        ],
        &Default::default(),
    );
    // [2, 4, 3]
    let b = Tensor::<B, 3>::from_floats(
        [
            [
                [5.0, 6.0, -1.0],
                [-0.5, -0.3, 0.2],
                [3.0, -10.0, -0.1],
                [0.5, 0.3, -0.2],
            ],
            [
                [10.0, 0.0, 5.0],
                [-1.0, -2.0, -3.0],
                [-0.5, -0.3, 0.2],
                [-0.45, 0.123, 0.8],
            ],
        ],
        &Default::default(),
    );
    // [2, 2, 3, 2]
    let c = right_broadcast_matmul(a.clone(), b.clone());
    assert_eq!(c.dims(), [2, 2, 3, 2]);

    for i in 0..2 {
        let mats0: Tensor<B, 3> = c.clone().slice_dim(3, i).squeeze_dim(3);
        let mats0 = into_ndarray(mats0);
        let mats0_actual = a.clone().slice_dim(3, i).squeeze_dim(3).matmul(b.clone());
        let mats0_actual = into_ndarray(mats0_actual);
        approx::assert_abs_diff_eq!(mats0, mats0_actual);
    }
}

pub fn matmul_impl<B: Backend>(delta: B::FloatElem, epsilon: B::FloatElem)
where
    B::FloatElem: approx::AbsDiffEq<B::FloatElem, Epsilon = B::FloatElem> + Copy,
{
    let model = ModelClient::<B>::default();
    let a = model.add_parameter(EuclideanMatrix36 {});
    let b = model.add_parameter(EuclideanMatrix62 {});
    let c = a.matmul(&b).unwrap();

    let batches = 4;

    let alen = batches * 3 * 6;
    let mut aval = Vec::with_capacity(alen);
    for i in 0..alen {
        let x: f32 = i as f32 / alen as f32;
        aval.push(x * x + x - 2.0f32);
    }
    let aval =
        Tensor::<B, 2>::from_data(TensorData::new(aval, [batches, 3 * 6]), &Default::default());

    let blen = batches * 6 * 2;
    let mut bval = Vec::with_capacity(blen);
    for i in 0..blen {
        let x: f32 = i as f32 / blen as f32;
        bval.push(2.0 * x - 1.0 + x.sin());
    }
    let bval =
        Tensor::<B, 2>::from_data(TensorData::new(bval, [batches, 6 * 2]), &Default::default());

    let p = points()
        .with_quantity(&a, Input::<B>::Float(aval.clone()))
        .with_quantity(&b, Input::<B>::Float(bval.clone()))
        .build();

    // Because this works, we expect the error is occurring in the derivative calculations or the
    // test harness itself
    let cval = c.evaluate(&p).unwrap();
    let cval = into_ndarray(cval);
    let cval_actual = aval
        .clone()
        .reshape([batches, 3, 6])
        .matmul(bval.clone().reshape([batches, 6, 2]))
        .reshape([batches, 6]);
    let cval_actual = into_ndarray(cval_actual);
    approx::assert_abs_diff_eq!(cval, cval_actual, epsilon = B::FloatElem::from_elem(1e-6));

    test_jvp_numerically(&a, &c, p.clone(), delta, epsilon);
    test_vjp_numerically(&a, &c, p.clone(), delta, epsilon);
    test_hvp_numerically(&a, &c, p.clone(), delta, epsilon);
    test_jvp_numerically(&b, &c, p.clone(), delta, epsilon);
    test_vjp_numerically(&b, &c, p.clone(), delta, epsilon);
    test_hvp_numerically(&b, &c, p.clone(), delta, epsilon);
}
