use skoro_backend::{
    CotangentFrame, CubeBackend, CubeFloat, CubeInt, CubeRuntime, NdArrayBackend, NdArrayFloat,
    NdArrayInt, NdArrayTensor, Point, SharedArray, TangentVector, Tensor,
};
use skoro_linalg::vector::cross_product::cube::left_broadcast_cross_product as cube_cross;
use skoro_linalg::vector::cross_product::ndarray::left_broadcast_cross_product as ndarray_cross;
use skoro_operation::{
    Input,
    diff::{DiffModeState, DiffOperation, DiffOperationForward},
    sum_opt_tensors,
};

use crate::ops::log::{cube, ndarray};

use super::SO3InverseRetraction;

impl<R: CubeRuntime, F: CubeFloat, I: CubeInt> DiffOperation<CubeBackend<R, F, I>>
    for SO3InverseRetraction
{
    type VjpState = (
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
        let [u, v] = inputs.try_into().unwrap();
        let b = u.dims()[0];
        match diff_state {
            DiffModeState::ForwardOnly => DiffOperationForward::forward_only(cube::log(
                u.reshape([b, 3, 3])
                    .transpose()
                    .matmul(v.reshape([b, 3, 3]))
                    .reshape([b, 9]),
            )),
            DiffModeState::Jvp(tvecs) => {
                let [du, dv] = tvecs.try_into().unwrap();
                let dr = u
                    .reshape([b, 3, 3])
                    .transpose()
                    .matmul(v.reshape([b, 3, 3]));
                let omega = cube::log(dr.clone().reshape([b, 9]));
                DiffOperationForward::jvp(
                    omega.clone(),
                    sum_opt_tensors([
                        du.map(|du| -cube::log_jvp(omega.clone(), dr.transpose().matmul(du))),
                        dv.map(|dv| cube::log_jvp(omega, dv)),
                    ]),
                )
            }
            DiffModeState::Vjp(diffs) => {
                let [udiff, vdiff] = diffs.try_into().unwrap();
                let dr = u
                    .reshape([b, 3, 3])
                    .transpose()
                    .matmul(v.reshape([b, 3, 3]));
                let omega = cube::log(dr.clone().reshape([b, 9]));
                DiffOperationForward::vjp(
                    omega.clone(),
                    (
                        if udiff || vdiff { Some(omega) } else { None },
                        if udiff { Some(dr) } else { None },
                        vdiff,
                    ),
                )
            }
            DiffModeState::Hvp(tvecs) => {
                let [du, dv] = tvecs.try_into().unwrap();
                let dr = u
                    .reshape([b, 3, 3])
                    .transpose()
                    .matmul(v.reshape([b, 3, 3]));
                let omega = cube::log(dr.clone().reshape([b, 9]));
                let udiff = du.is_some();
                let vdiff = dv.is_some();
                DiffOperationForward::hvp(
                    omega.clone(),
                    sum_opt_tensors([
                        du.as_ref().map(|du| {
                            -cube::log_jvp(
                                omega.clone(),
                                dr.clone().transpose().matmul(du.clone().unsqueeze_dim(2)),
                            )
                            .squeeze_dim(2)
                        }),
                        dv.as_ref().map(|dv| {
                            cube::log_jvp(omega.clone(), dv.clone().unsqueeze_dim(2)).squeeze_dim(2)
                        }),
                    ]),
                    (
                        if udiff || vdiff { Some(omega) } else { None },
                        if udiff { Some(dr.clone()) } else { None },
                        vdiff,
                    ),
                    [du, dv],
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
            (Some(omega), Some(dr), true) => vec![
                Some(
                    -dr.clone()
                        .matmul(cube::log_vjp(omega.clone(), alpha.clone())),
                ),
                Some(cube::log_vjp(omega.clone(), alpha)),
            ],
            (Some(omega), Some(dr), false) => vec![
                Some(-dr.clone().matmul(cube::log_vjp(omega.clone(), alpha))),
                None,
            ],
            (Some(omega), None, true) => vec![None, Some(cube::log_vjp(omega.clone(), alpha))],
            (None, None, false) => vec![None, None],
            _ => unreachable!(),
        }
    }

    fn hvp(
        &self,
        vjp_state: &Self::VjpState,
        hvp_state: &Self::HvpState,
        alpha: CotangentFrame<CubeBackend<R, F, I>>,
    ) -> Vec<Option<CotangentFrame<CubeBackend<R, F, I>>>> {
        // TODO: fix implementation
        match (vjp_state, hvp_state) {
            ((Some(omega), Some(dr), true), [Some(du), Some(dv)]) => {
                let vjp = cube::log_vjp(omega.clone(), alpha.clone());
                let a0 = cube_cross(du.clone(), dr.clone().matmul(vjp.clone()));
                let a1 = dr.clone().matmul(cube_cross(dv.clone(), vjp));
                let hvp = cube::log_hvp(
                    omega.clone(),
                    -dr.clone()
                        .transpose()
                        .matmul(du.clone().unsqueeze_dim(2))
                        .squeeze_dim(2)
                        + dv.clone(),
                    alpha,
                );
                vec![Some(a0 - a1 - dr.clone().matmul(hvp.clone())), Some(hvp)]
            }
            ((Some(omega), Some(dr), false), [Some(du), None]) => {
                let a0 = cube_cross(
                    du.clone(),
                    dr.clone()
                        .matmul(cube::log_vjp(omega.clone(), alpha.clone())),
                );
                let hvp = cube::log_hvp(
                    omega.clone(),
                    -dr.clone()
                        .transpose()
                        .matmul(du.clone().unsqueeze_dim(2))
                        .squeeze_dim(2),
                    alpha,
                );
                vec![Some(a0 - dr.clone().matmul(hvp)), None]
            }
            ((Some(omega), None, true), [None, Some(dv)]) => {
                vec![None, Some(cube::log_hvp(omega.clone(), dv.clone(), alpha))]
            }
            ((None, None, false), [None, None]) => vec![None, None],
            _ => unreachable!(),
        }
    }
}

impl<F, I> DiffOperation<NdArrayBackend<F, I>> for SO3InverseRetraction
where
    F: NdArrayFloat + num_traits::Float,
    I: NdArrayInt,
    NdArrayTensor: From<SharedArray<F>> + From<SharedArray<I>>,
{
    type VjpState = (
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
        let [u, v] = inputs.try_into().unwrap();
        let b = u.dims()[0];
        match diff_state {
            DiffModeState::ForwardOnly => DiffOperationForward::forward_only(ndarray::log(
                u.reshape([b, 3, 3])
                    .transpose()
                    .matmul(v.reshape([b, 3, 3]))
                    .reshape([b, 9]),
            )),
            DiffModeState::Jvp(tvecs) => {
                let [du, dv] = tvecs.try_into().unwrap();
                let dr = u
                    .reshape([b, 3, 3])
                    .transpose()
                    .matmul(v.reshape([b, 3, 3]));
                let omega = ndarray::log(dr.clone().reshape([b, 9]));
                DiffOperationForward::jvp(
                    omega.clone(),
                    sum_opt_tensors([
                        du.map(|du| -ndarray::log_jvp(omega.clone(), dr.transpose().matmul(du))),
                        dv.map(|dv| ndarray::log_jvp(omega, dv)),
                    ]),
                )
            }
            DiffModeState::Vjp(diffs) => {
                let [udiff, vdiff] = diffs.try_into().unwrap();
                let dr = u
                    .reshape([b, 3, 3])
                    .transpose()
                    .matmul(v.reshape([b, 3, 3]));
                let omega = ndarray::log(dr.clone().reshape([b, 9]));
                DiffOperationForward::vjp(
                    omega.clone(),
                    (
                        if udiff || vdiff { Some(omega) } else { None },
                        if udiff { Some(dr) } else { None },
                        vdiff,
                    ),
                )
            }
            DiffModeState::Hvp(tvecs) => {
                let [du, dv] = tvecs.try_into().unwrap();
                let dr = u
                    .reshape([b, 3, 3])
                    .transpose()
                    .matmul(v.reshape([b, 3, 3]));
                let omega = ndarray::log(dr.clone().reshape([b, 9]));
                let udiff = du.is_some();
                let vdiff = dv.is_some();
                DiffOperationForward::hvp(
                    omega.clone(),
                    sum_opt_tensors([
                        du.as_ref().map(|du| {
                            -ndarray::log_jvp(
                                omega.clone(),
                                dr.clone().transpose().matmul(du.clone().unsqueeze_dim(2)),
                            )
                            .squeeze_dim(2)
                        }),
                        dv.as_ref().map(|dv| {
                            ndarray::log_jvp(omega.clone(), dv.clone().unsqueeze_dim(2))
                                .squeeze_dim(2)
                        }),
                    ]),
                    (
                        if udiff || vdiff { Some(omega) } else { None },
                        if udiff { Some(dr.clone()) } else { None },
                        vdiff,
                    ),
                    [du, dv],
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
            (Some(omega), Some(dr), true) => vec![
                Some(
                    -dr.clone()
                        .matmul(ndarray::log_vjp(omega.clone(), alpha.clone())),
                ),
                Some(ndarray::log_vjp(omega.clone(), alpha)),
            ],
            (Some(omega), Some(dr), false) => vec![
                Some(-dr.clone().matmul(ndarray::log_vjp(omega.clone(), alpha))),
                None,
            ],
            (Some(omega), None, true) => vec![None, Some(ndarray::log_vjp(omega.clone(), alpha))],
            (None, None, false) => vec![None, None],
            _ => unreachable!(),
        }
    }

    fn hvp(
        &self,
        vjp_state: &Self::VjpState,
        hvp_state: &Self::HvpState,
        alpha: CotangentFrame<NdArrayBackend<F, I>>,
    ) -> Vec<Option<CotangentFrame<NdArrayBackend<F, I>>>> {
        // TODO: fix implementation
        match (vjp_state, hvp_state) {
            ((Some(omega), Some(dr), true), [Some(du), Some(dv)]) => {
                let vjp = ndarray::log_vjp(omega.clone(), alpha.clone());
                let a0 = ndarray_cross(du.clone(), dr.clone().matmul(vjp.clone()));
                let a1 = dr.clone().matmul(ndarray_cross(dv.clone(), vjp));
                let hvp = ndarray::log_hvp(
                    omega.clone(),
                    -dr.clone()
                        .transpose()
                        .matmul(du.clone().unsqueeze_dim(2))
                        .squeeze_dim(2)
                        + dv.clone(),
                    alpha,
                );
                vec![Some(a0 - a1 - dr.clone().matmul(hvp.clone())), Some(hvp)]
            }
            ((Some(omega), Some(dr), false), [Some(du), None]) => {
                let a0 = ndarray_cross(
                    du.clone(),
                    dr.clone()
                        .matmul(ndarray::log_vjp(omega.clone(), alpha.clone())),
                );
                let hvp = ndarray::log_hvp(
                    omega.clone(),
                    -dr.clone()
                        .transpose()
                        .matmul(du.clone().unsqueeze_dim(2))
                        .squeeze_dim(2),
                    alpha,
                );
                vec![Some(a0 - dr.clone().matmul(hvp)), None]
            }
            ((Some(omega), None, true), [None, Some(dv)]) => {
                vec![
                    None,
                    Some(ndarray::log_hvp(omega.clone(), dv.clone(), alpha)),
                ]
            }
            ((None, None, false), [None, None]) => vec![None, None],
            _ => unreachable!(),
        }
    }
}
