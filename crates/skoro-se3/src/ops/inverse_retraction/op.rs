use num_traits::Float;
use skoro_backend::{
    CotangentFrame, CubeBackend, CubeFloat, CubeInt, CubeRuntime, NdArrayBackend, NdArrayFloat,
    NdArrayInt, NdArrayTensor, Point, SharedArray, TangentVector, Tensor,
};
use skoro_linalg::vector::cross_product::cube::cross_product as cube_cross;
use skoro_linalg::vector::cross_product::cube::left_broadcast_cross_product as cube_left_broadcast_cross;
use skoro_linalg::vector::cross_product::ndarray::cross_product as ndarray_cross;
use skoro_linalg::vector::cross_product::ndarray::left_broadcast_cross_product as ndarray_left_broadcast_cross;
use skoro_operation::{
    Input,
    diff::{DiffModeState, DiffOperation, DiffOperationForward},
    sum_opt_tensors,
};
use skoro_so3::ops::log::{cube, ndarray};

use super::SE3InverseRetraction;

impl<R: CubeRuntime, F: CubeFloat, I: CubeInt> DiffOperation<CubeBackend<R, F, I>>
    for SE3InverseRetraction
{
    type VjpState = (
        Option<Point<CubeBackend<R, F, I>>>,
        Option<Point<CubeBackend<R, F, I>>>,
        Option<Tensor<CubeBackend<R, F, I>, 3>>,
        bool,
    );
    type HvpState = [Option<TangentVector<CubeBackend<R, F, I>>>; 2];

    fn forward(
        &self,
        inputs: Vec<Input<CubeBackend<R, F, I>>>,
        diff_state: DiffModeState<CubeBackend<R, F, I>>,
    ) -> DiffOperationForward<CubeBackend<R, F, I>, Self::VjpState, Self::HvpState> {
        let inputs: Vec<Point<CubeBackend<R, F, I>>> =
            inputs.into_iter().map(Input::into_float).collect();
        let [rp0, rp1] = inputs.try_into().unwrap();
        let b = rp0.dims()[0];
        let r0 = rp0.clone().slice_dim(1, 0..9).reshape([b, 3, 3]);
        let p0 = rp0.slice_dim(1, 9..12);
        let r1 = rp1.clone().slice_dim(1, 0..9).reshape([b, 3, 3]);
        let p1 = rp1.slice_dim(1, 9..12);
        let r0inv_dp = r0
            .clone()
            .transpose()
            .matmul((p1.clone() - p0.clone()).unsqueeze_dim(2))
            .squeeze_dim(2);
        let dr = r0.clone().transpose().matmul(r1.clone());
        let omega = cube::log(dr.clone().reshape([b, 9]));
        match diff_state {
            DiffModeState::ForwardOnly => {
                DiffOperationForward::forward_only(Point::cat(vec![omega, r0inv_dp], 1))
            }
            DiffModeState::Jvp(tvecs) => {
                let [drp0, drp1] = tvecs.try_into().unwrap();
                DiffOperationForward::jvp(
                    Point::cat(vec![omega.clone(), r0inv_dp.clone()], 1),
                    sum_opt_tensors([
                        drp0.map(|drp0| {
                            let omega0 = drp0.clone().slice_dim(1, 0..3);
                            let v0 = drp0.clone().slice_dim(1, 3..6);
                            let drot = -cube::log_jvp(
                                omega.clone(),
                                dr.clone().transpose().matmul(omega0.clone()),
                            );
                            let dpsn = cube_left_broadcast_cross(r0inv_dp.clone(), omega0) - v0;
                            Tensor::cat(vec![drot, dpsn], 1)
                        }),
                        drp1.map(|drp1| {
                            let omega1 = drp1.clone().slice_dim(1, 0..3);
                            let v1 = drp1.clone().slice_dim(1, 3..6);
                            let drot = cube::log_jvp(omega, omega1);
                            let dpsn = dr.matmul(v1);
                            Tensor::cat(vec![drot, dpsn], 1)
                        }),
                    ]),
                )
            }
            DiffModeState::Vjp(diffs) => {
                let [diff0, diff1] = diffs.try_into().unwrap();
                DiffOperationForward::vjp(
                    Point::cat(vec![omega.clone(), r0inv_dp.clone()], 1),
                    (
                        if diff0 || diff1 { Some(omega) } else { None },
                        if diff0 { Some(r0inv_dp) } else { None },
                        if diff0 || diff1 { Some(dr) } else { None },
                        diff1,
                    ),
                )
            }
            DiffModeState::Hvp(tvecs) => {
                let [drp0, drp1] = tvecs.try_into().unwrap();
                let diff0 = drp0.is_some();
                let diff1 = drp1.is_some();
                DiffOperationForward::hvp(
                    Point::cat(vec![omega.clone(), r0inv_dp.clone()], 1),
                    sum_opt_tensors([
                        drp0.as_ref().map(|drp0| {
                            let omega0 = drp0.clone().slice_dim(1, 0..3);
                            let v0 = drp0.clone().slice_dim(1, 3..6);
                            let drot = -cube::log_jvp(
                                omega.clone(),
                                dr.clone()
                                    .transpose()
                                    .matmul(omega0.clone().unsqueeze_dim(2)),
                            )
                            .squeeze_dim(2);
                            let dpsn = cube_cross(r0inv_dp.clone(), omega0) - v0;
                            Tensor::cat(vec![drot, dpsn], 1)
                        }),
                        drp1.as_ref().map(|drp1| {
                            let omega1 = drp1.clone().slice_dim(1, 0..3);
                            let v1 = drp1.clone().slice_dim(1, 3..6);
                            let drot = cube::log_jvp(omega.clone(), omega1.unsqueeze_dim(2))
                                .squeeze_dim(2);
                            let dpsn = dr.clone().matmul(v1.unsqueeze_dim(2)).squeeze_dim(2);
                            Tensor::cat(vec![drot, dpsn], 1)
                        }),
                    ]),
                    (
                        if diff0 || diff1 { Some(omega) } else { None },
                        if diff0 { Some(r0inv_dp) } else { None },
                        if diff0 || diff1 { Some(dr) } else { None },
                        diff1,
                    ),
                    [drp0, drp1],
                )
            }
        }
    }

    fn vjp(
        &self,
        vjp_state: &Self::VjpState,
        alpha: CotangentFrame<CubeBackend<R, F, I>>,
    ) -> Vec<Option<CotangentFrame<CubeBackend<R, F, I>>>> {
        match vjp_state {
            (Some(omega), Some(r0inv_dp), Some(dr), true) => {
                let alphar = alpha.clone().slice_dim(1, 0..3);
                let alphap = alpha.slice_dim(1, 3..6);
                let drot0 = -dr
                    .clone()
                    .matmul(cube::log_vjp(omega.clone(), alphar.clone()))
                    - cube_left_broadcast_cross(r0inv_dp.clone(), alphap.clone());
                let dpsn0 = -alphap.clone();
                let drot1 = cube::log_vjp(omega.clone(), alphar);
                let dpsn1 = dr.clone().transpose().matmul(alphap);
                vec![
                    Some(Tensor::cat(vec![drot0, dpsn0], 1)),
                    Some(Tensor::cat(vec![drot1, dpsn1], 1)),
                ]
            }
            (Some(omega), Some(r0inv_dp), Some(dr), false) => {
                let alphar = alpha.clone().slice_dim(1, 0..3);
                let alphap = alpha.slice_dim(1, 3..6);
                let drot0 = -dr
                    .clone()
                    .matmul(cube::log_vjp(omega.clone(), alphar.clone()))
                    - cube_left_broadcast_cross(r0inv_dp.clone(), alphap.clone());
                let dpsn0 = -alphap.clone();
                vec![Some(Tensor::cat(vec![drot0, dpsn0], 1)), None]
            }
            (Some(omega), None, Some(dr), true) => {
                let alphar = alpha.clone().slice_dim(1, 0..3);
                let alphap = alpha.slice_dim(1, 3..6);
                let drot1 = cube::log_vjp(omega.clone(), alphar);
                let dpsn1 = dr.clone().transpose().matmul(alphap);
                vec![None, Some(Tensor::cat(vec![drot1, dpsn1], 1))]
            }
            (None, None, None, false) => {
                vec![None, None]
            }
            _ => unreachable!(),
        }
    }

    fn hvp(
        &self,
        vjp_state: &Self::VjpState,
        hvp_state: &Self::HvpState,
        alpha: CotangentFrame<CubeBackend<R, F, I>>,
    ) -> Vec<Option<CotangentFrame<CubeBackend<R, F, I>>>> {
        match (vjp_state, hvp_state) {
            ((Some(omega), Some(r0inv_dp), Some(dr), true), [Some(drp0), Some(drp1)]) => {
                let alphar = alpha.clone().slice_dim(1, 0..3);
                let alphap = alpha.slice_dim(1, 3..6);

                let omega0 = drp0.clone().slice_dim(1, 0..3);
                let v0 = drp0.clone().slice_dim(1, 3..6);

                let omega1 = drp1.clone().slice_dim(1, 0..3);
                let v1 = drp1.clone().slice_dim(1, 3..6);

                let vjp = cube::log_vjp(omega.clone(), alphar.clone());
                let a0 = cube_left_broadcast_cross(omega0.clone(), dr.clone().matmul(vjp.clone()));
                let a1 = dr
                    .clone()
                    .matmul(cube_left_broadcast_cross(omega1.clone(), vjp));
                let neg_dromega0_plus_omega1 = -dr
                    .clone()
                    .transpose()
                    .matmul(omega0.clone().unsqueeze_dim(2))
                    .squeeze_dim(2)
                    + omega1.clone();
                let log_hvp = cube::log_hvp(
                    omega.clone(),
                    neg_dromega0_plus_omega1.clone(),
                    alphar.clone(),
                );

                let local_so3_hvp = a0 - a1 - dr.clone().matmul(log_hvp.clone());

                let b0 = -cube_left_broadcast_cross(
                    cube_cross(r0inv_dp.clone(), omega0.clone()),
                    alphap.clone(),
                );
                let b1 = -cube_left_broadcast_cross(
                    dr.clone().matmul(v1.unsqueeze_dim(2)).squeeze_dim(2) - v0,
                    alphap.clone(),
                );

                let hvp_r0 = local_so3_hvp + b0 + b1;
                let hvp_p0 = alphar.zeros_like();
                let hvp_r1 = log_hvp;
                let hvp_p1 =
                    -cube_left_broadcast_cross(neg_dromega0_plus_omega1, dr.clone().matmul(alphap));

                vec![
                    Some(Tensor::cat(vec![hvp_r0, hvp_p0], 1)),
                    Some(Tensor::cat(vec![hvp_r1, hvp_p1], 1)),
                ]
            }
            ((Some(omega), Some(r0inv_dp), Some(dr), false), [Some(drp0), None]) => {
                let alphar = alpha.clone().slice_dim(1, 0..3);
                let alphap = alpha.slice_dim(1, 3..6);

                let omega0 = drp0.clone().slice_dim(1, 0..3);
                let v0 = drp0.clone().slice_dim(1, 3..6);

                let vjp = cube::log_vjp(omega.clone(), alphar.clone());
                let a0 = cube_left_broadcast_cross(omega0.clone(), dr.clone().matmul(vjp.clone()));
                let neg_dromega0 = -dr
                    .clone()
                    .transpose()
                    .matmul(omega0.clone().unsqueeze_dim(2))
                    .squeeze_dim(2);
                let log_hvp = cube::log_hvp(omega.clone(), neg_dromega0, alphar.clone());

                let local_so3_hvp = a0 - dr.clone().matmul(log_hvp.clone());

                let b0 = -cube_left_broadcast_cross(
                    cube_cross(r0inv_dp.clone(), omega0.clone()),
                    alphap.clone(),
                );
                let b1 = cube_left_broadcast_cross(v0, alphap.clone());

                let hvp_r0 = local_so3_hvp + b0 + b1;
                let hvp_p0 = alphar.zeros_like();

                vec![Some(Tensor::cat(vec![hvp_r0, hvp_p0], 1)), None]
            }
            ((Some(omega), None, Some(dr), true), [None, Some(drp1)]) => {
                let alphar = alpha.clone().slice_dim(1, 0..3);
                let alphap = alpha.slice_dim(1, 3..6);

                let omega1 = drp1.clone().slice_dim(1, 0..3);

                let log_hvp = cube::log_hvp(omega.clone(), omega1.clone(), alphar.clone());

                let hvp_r1 = log_hvp;
                let hvp_p1 = -cube_left_broadcast_cross(omega1, dr.clone().matmul(alphap));

                vec![None, Some(Tensor::cat(vec![hvp_r1, hvp_p1], 1))]
            }
            ((None, None, None, false), [None, None]) => {
                vec![None, None]
            }
            _ => unreachable!(),
        }
    }
}

impl<F, I> DiffOperation<NdArrayBackend<F, I>> for SE3InverseRetraction
where
    F: NdArrayFloat + Float,
    I: NdArrayInt,
    NdArrayTensor: From<SharedArray<F>> + From<SharedArray<I>>,
{
    type VjpState = (
        Option<Point<NdArrayBackend<F, I>>>,
        Option<Point<NdArrayBackend<F, I>>>,
        Option<Tensor<NdArrayBackend<F, I>, 3>>,
        bool,
    );
    type HvpState = [Option<TangentVector<NdArrayBackend<F, I>>>; 2];

    fn forward(
        &self,
        inputs: Vec<Input<NdArrayBackend<F, I>>>,
        diff_state: DiffModeState<NdArrayBackend<F, I>>,
    ) -> DiffOperationForward<NdArrayBackend<F, I>, Self::VjpState, Self::HvpState> {
        let inputs: Vec<Point<NdArrayBackend<F, I>>> =
            inputs.into_iter().map(Input::into_float).collect();
        let [rp0, rp1] = inputs.try_into().unwrap();
        let b = rp0.dims()[0];
        let r0 = rp0.clone().slice_dim(1, 0..9).reshape([b, 3, 3]);
        let p0 = rp0.slice_dim(1, 9..12);
        let r1 = rp1.clone().slice_dim(1, 0..9).reshape([b, 3, 3]);
        let p1 = rp1.slice_dim(1, 9..12);
        let r0inv_dp = r0
            .clone()
            .transpose()
            .matmul((p1.clone() - p0.clone()).unsqueeze_dim(2))
            .squeeze_dim(2);
        let dr = r0.clone().transpose().matmul(r1.clone());
        let omega = ndarray::log(dr.clone().reshape([b, 9]));
        match diff_state {
            DiffModeState::ForwardOnly => {
                DiffOperationForward::forward_only(Point::cat(vec![omega, r0inv_dp], 1))
            }
            DiffModeState::Jvp(tvecs) => {
                let [drp0, drp1] = tvecs.try_into().unwrap();
                DiffOperationForward::jvp(
                    Point::cat(vec![omega.clone(), r0inv_dp.clone()], 1),
                    sum_opt_tensors([
                        drp0.map(|drp0| {
                            let omega0 = drp0.clone().slice_dim(1, 0..3);
                            let v0 = drp0.clone().slice_dim(1, 3..6);
                            let drot = -ndarray::log_jvp(
                                omega.clone(),
                                dr.clone().transpose().matmul(omega0.clone()),
                            );
                            let dpsn = ndarray_left_broadcast_cross(r0inv_dp.clone(), omega0) - v0;
                            Tensor::cat(vec![drot, dpsn], 1)
                        }),
                        drp1.map(|drp1| {
                            let omega1 = drp1.clone().slice_dim(1, 0..3);
                            let v1 = drp1.clone().slice_dim(1, 3..6);
                            let drot = ndarray::log_jvp(omega, omega1);
                            let dpsn = dr.matmul(v1);
                            Tensor::cat(vec![drot, dpsn], 1)
                        }),
                    ]),
                )
            }
            DiffModeState::Vjp(diffs) => {
                let [diff0, diff1] = diffs.try_into().unwrap();
                DiffOperationForward::vjp(
                    Point::cat(vec![omega.clone(), r0inv_dp.clone()], 1),
                    (
                        if diff0 || diff1 { Some(omega) } else { None },
                        if diff0 { Some(r0inv_dp) } else { None },
                        if diff0 || diff1 { Some(dr) } else { None },
                        diff1,
                    ),
                )
            }
            DiffModeState::Hvp(tvecs) => {
                let [drp0, drp1] = tvecs.try_into().unwrap();
                let diff0 = drp0.is_some();
                let diff1 = drp1.is_some();
                DiffOperationForward::hvp(
                    Point::cat(vec![omega.clone(), r0inv_dp.clone()], 1),
                    sum_opt_tensors([
                        drp0.as_ref().map(|drp0| {
                            let omega0 = drp0.clone().slice_dim(1, 0..3);
                            let v0 = drp0.clone().slice_dim(1, 3..6);
                            let drot = -ndarray::log_jvp(
                                omega.clone(),
                                dr.clone()
                                    .transpose()
                                    .matmul(omega0.clone().unsqueeze_dim(2)),
                            )
                            .squeeze_dim(2);
                            let dpsn = ndarray_cross(r0inv_dp.clone(), omega0) - v0;
                            Tensor::cat(vec![drot, dpsn], 1)
                        }),
                        drp1.as_ref().map(|drp1| {
                            let omega1 = drp1.clone().slice_dim(1, 0..3);
                            let v1 = drp1.clone().slice_dim(1, 3..6);
                            let drot = ndarray::log_jvp(omega.clone(), omega1.unsqueeze_dim(2))
                                .squeeze_dim(2);
                            let dpsn = dr.clone().matmul(v1.unsqueeze_dim(2)).squeeze_dim(2);
                            Tensor::cat(vec![drot, dpsn], 1)
                        }),
                    ]),
                    (
                        if diff0 || diff1 { Some(omega) } else { None },
                        if diff0 { Some(r0inv_dp) } else { None },
                        if diff0 || diff1 { Some(dr) } else { None },
                        diff1,
                    ),
                    [drp0, drp1],
                )
            }
        }
    }

    fn vjp(
        &self,
        vjp_state: &Self::VjpState,
        alpha: CotangentFrame<NdArrayBackend<F, I>>,
    ) -> Vec<Option<CotangentFrame<NdArrayBackend<F, I>>>> {
        match vjp_state {
            (Some(omega), Some(r0inv_dp), Some(dr), true) => {
                let alphar = alpha.clone().slice_dim(1, 0..3);
                let alphap = alpha.slice_dim(1, 3..6);
                let drot0 = -dr
                    .clone()
                    .matmul(ndarray::log_vjp(omega.clone(), alphar.clone()))
                    - ndarray_left_broadcast_cross(r0inv_dp.clone(), alphap.clone());
                let dpsn0 = -alphap.clone();
                let drot1 = ndarray::log_vjp(omega.clone(), alphar);
                let dpsn1 = dr.clone().transpose().matmul(alphap);
                vec![
                    Some(Tensor::cat(vec![drot0, dpsn0], 1)),
                    Some(Tensor::cat(vec![drot1, dpsn1], 1)),
                ]
            }
            (Some(omega), Some(r0inv_dp), Some(dr), false) => {
                let alphar = alpha.clone().slice_dim(1, 0..3);
                let alphap = alpha.slice_dim(1, 3..6);
                let drot0 = -dr
                    .clone()
                    .matmul(ndarray::log_vjp(omega.clone(), alphar.clone()))
                    - ndarray_left_broadcast_cross(r0inv_dp.clone(), alphap.clone());
                let dpsn0 = -alphap.clone();
                vec![Some(Tensor::cat(vec![drot0, dpsn0], 1)), None]
            }
            (Some(omega), None, Some(dr), true) => {
                let alphar = alpha.clone().slice_dim(1, 0..3);
                let alphap = alpha.slice_dim(1, 3..6);
                let drot1 = ndarray::log_vjp(omega.clone(), alphar);
                let dpsn1 = dr.clone().transpose().matmul(alphap);
                vec![None, Some(Tensor::cat(vec![drot1, dpsn1], 1))]
            }
            (None, None, None, false) => {
                vec![None, None]
            }
            _ => unreachable!(),
        }
    }

    fn hvp(
        &self,
        vjp_state: &Self::VjpState,
        hvp_state: &Self::HvpState,
        alpha: CotangentFrame<NdArrayBackend<F, I>>,
    ) -> Vec<Option<CotangentFrame<NdArrayBackend<F, I>>>> {
        match (vjp_state, hvp_state) {
            ((Some(omega), Some(r0inv_dp), Some(dr), true), [Some(drp0), Some(drp1)]) => {
                let alphar = alpha.clone().slice_dim(1, 0..3);
                let alphap = alpha.slice_dim(1, 3..6);

                let omega0 = drp0.clone().slice_dim(1, 0..3);
                let v0 = drp0.clone().slice_dim(1, 3..6);

                let omega1 = drp1.clone().slice_dim(1, 0..3);
                let v1 = drp1.clone().slice_dim(1, 3..6);

                let vjp = ndarray::log_vjp(omega.clone(), alphar.clone());
                let a0 =
                    ndarray_left_broadcast_cross(omega0.clone(), dr.clone().matmul(vjp.clone()));
                let a1 = dr
                    .clone()
                    .matmul(ndarray_left_broadcast_cross(omega1.clone(), vjp));
                let neg_dromega0_plus_omega1 = -dr
                    .clone()
                    .transpose()
                    .matmul(omega0.clone().unsqueeze_dim(2))
                    .squeeze_dim(2)
                    + omega1.clone();
                let log_hvp = ndarray::log_hvp(
                    omega.clone(),
                    neg_dromega0_plus_omega1.clone(),
                    alphar.clone(),
                );

                let local_so3_hvp = a0 - a1 - dr.clone().matmul(log_hvp.clone());

                let b0 = -ndarray_left_broadcast_cross(
                    ndarray_cross(r0inv_dp.clone(), omega0.clone()),
                    alphap.clone(),
                );
                let b1 = -ndarray_left_broadcast_cross(
                    dr.clone().matmul(v1.unsqueeze_dim(2)).squeeze_dim(2) - v0,
                    alphap.clone(),
                );

                let hvp_r0 = local_so3_hvp + b0 + b1;
                let hvp_p0 = alphar.zeros_like();
                let hvp_r1 = log_hvp;
                let hvp_p1 = -ndarray_left_broadcast_cross(
                    neg_dromega0_plus_omega1,
                    dr.clone().matmul(alphap),
                );

                vec![
                    Some(Tensor::cat(vec![hvp_r0, hvp_p0], 1)),
                    Some(Tensor::cat(vec![hvp_r1, hvp_p1], 1)),
                ]
            }
            ((Some(omega), Some(r0inv_dp), Some(dr), false), [Some(drp0), None]) => {
                let alphar = alpha.clone().slice_dim(1, 0..3);
                let alphap = alpha.slice_dim(1, 3..6);

                let omega0 = drp0.clone().slice_dim(1, 0..3);
                let v0 = drp0.clone().slice_dim(1, 3..6);

                let vjp = ndarray::log_vjp(omega.clone(), alphar.clone());
                let a0 =
                    ndarray_left_broadcast_cross(omega0.clone(), dr.clone().matmul(vjp.clone()));
                let neg_dromega0 = -dr
                    .clone()
                    .transpose()
                    .matmul(omega0.clone().unsqueeze_dim(2))
                    .squeeze_dim(2);
                let log_hvp = ndarray::log_hvp(omega.clone(), neg_dromega0, alphar.clone());

                let local_so3_hvp = a0 - dr.clone().matmul(log_hvp.clone());

                let b0 = -ndarray_left_broadcast_cross(
                    ndarray_cross(r0inv_dp.clone(), omega0.clone()),
                    alphap.clone(),
                );
                let b1 = ndarray_left_broadcast_cross(v0, alphap.clone());

                let hvp_r0 = local_so3_hvp + b0 + b1;
                let hvp_p0 = alphar.zeros_like();

                vec![Some(Tensor::cat(vec![hvp_r0, hvp_p0], 1)), None]
            }
            ((Some(omega), None, Some(dr), true), [None, Some(drp1)]) => {
                let alphar = alpha.clone().slice_dim(1, 0..3);
                let alphap = alpha.slice_dim(1, 3..6);

                let omega1 = drp1.clone().slice_dim(1, 0..3);

                let log_hvp = ndarray::log_hvp(omega.clone(), omega1.clone(), alphar.clone());

                let hvp_r1 = log_hvp;
                let hvp_p1 = -ndarray_left_broadcast_cross(omega1, dr.clone().matmul(alphap));

                vec![None, Some(Tensor::cat(vec![hvp_r1, hvp_p1], 1))]
            }
            ((None, None, None, false), [None, None]) => {
                vec![None, None]
            }
            _ => unreachable!(),
        }
    }
}
