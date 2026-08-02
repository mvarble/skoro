use num_traits::Float;

use skoro_backend::{
    CotangentFrame, CubeBackend, CubeFloat, CubeInt, CubeRuntime, NdArrayBackend, NdArrayFloat,
    NdArrayInt, NdArrayTensor, Point, SharedArray, TangentVector, Tensor,
};
use skoro_linalg::{
    vector::cross_product::cube::left_broadcast_cross_product as cube_cross,
    vector::cross_product::ndarray::left_broadcast_cross_product as ndarray_cross,
};
use skoro_operation::{
    Input,
    diff::{DiffModeState, DiffOperation, DiffOperationForward},
    sum_opt_tensors,
};
use skoro_so3::ops::exp::{cube, ndarray};

use super::SE3Retraction;

impl<R: CubeRuntime, F: CubeFloat, I: CubeInt> DiffOperation<CubeBackend<R, F, I>>
    for SE3Retraction
{
    type VjpState = (
        Option<Tensor<CubeBackend<R, F, I>, 3>>,
        Option<Point<CubeBackend<R, F, I>>>,
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
        let [rp, omegav] = inputs.try_into().unwrap();
        let b = rp.dims()[0];
        let r = rp.clone().slice_dim(1, 0..9);
        let p = rp.slice_dim(1, 9..12);
        let omega = omegav.clone().slice_dim(1, 0..3);
        let v = omegav.slice_dim(1, 3..6);
        let p_plus_rv = p + r
            .clone()
            .reshape([b, 3, 3])
            .matmul(v.unsqueeze_dim(2))
            .squeeze_dim(2);
        let rmat = r.reshape([b, 3, 3]);
        let expomega = cube::exp(omega.clone()).reshape([b, 3, 3]);
        match diff_state {
            DiffModeState::ForwardOnly => {
                let rexpomega = rmat.matmul(expomega).reshape([b, 9]);
                DiffOperationForward::forward_only(Point::cat(vec![rexpomega, p_plus_rv], 1))
            }
            DiffModeState::Jvp(tvecs) => {
                let [drp, domegav] = tvecs.try_into().unwrap();
                let dr = drp.as_ref().map(|drp| drp.clone().slice_dim(1, 0..3));
                let dp = drp.map(|drp| drp.slice_dim(1, 3..6));
                let domega = domegav
                    .as_ref()
                    .map(|domegav| domegav.clone().slice_dim(1, 0..3));
                let dv = domegav.map(|domegav| domegav.slice_dim(1, 3..6));

                let rexpomega = rmat.matmul(expomega.clone()).reshape([b, 9]);
                DiffOperationForward::jvp(
                    Point::cat(vec![rexpomega, p_plus_rv], 1),
                    sum_opt_tensors([
                        dr.map(|dr| expomega.clone().transpose().matmul(dr)),
                        dp.map(|dp| expomega.clone().matmul(dp)),
                        domega.map(|domega| cube::exp_jvp(omega, domega)),
                        dv.map(|dv| expomega.matmul(dv)),
                    ]),
                )
            }
            DiffModeState::Vjp(diffs) => {
                let [rp_diff, omegav_diff] = diffs.try_into().unwrap();
                let rexpomega = rmat.matmul(expomega.clone()).reshape([b, 9]);
                DiffOperationForward::vjp(
                    Point::cat(vec![rexpomega, p_plus_rv], 1),
                    (
                        if rp_diff || omegav_diff {
                            Some(expomega)
                        } else {
                            None
                        },
                        if omegav_diff { Some(omega) } else { None },
                        rp_diff,
                    ),
                )
            }
            DiffModeState::Hvp(tvecs) => {
                let [drp, domegav] = tvecs.try_into().unwrap();

                let rp_diff = drp.is_some();
                let omegav_diff = domegav.is_some();

                let dr = drp.as_ref().map(|drp| drp.clone().slice_dim(1, 0..3));
                let dp = drp.map(|drp| drp.slice_dim(1, 3..6));
                let domega = domegav
                    .as_ref()
                    .map(|domegav| domegav.clone().slice_dim(1, 0..3));
                let dv = domegav.map(|domegav| domegav.slice_dim(1, 3..6));

                let expomega = cube::exp(omega.clone()).reshape([b, 3, 3]);
                let dexp_domega = domega.as_ref().map(|domega| {
                    cube::exp_jvp(omega.clone(), domega.clone().unsqueeze_dim(2)).squeeze_dim(2)
                });
                let rexpomega = rmat.matmul(expomega.clone()).reshape([b, 9]);
                DiffOperationForward::hvp(
                    Point::cat(vec![rexpomega, p_plus_rv], 1),
                    sum_opt_tensors([
                        dr.map(|dr| {
                            expomega
                                .clone()
                                .transpose()
                                .matmul(dr.unsqueeze_dim(2))
                                .squeeze_dim(2)
                        }),
                        dp.map(|dp| expomega.clone().matmul(dp.unsqueeze_dim(2)).squeeze_dim(2)),
                        dexp_domega.clone(),
                        dv.map(|dv| expomega.clone().matmul(dv.unsqueeze_dim(2)).squeeze_dim(2)),
                    ]),
                    (
                        if rp_diff || omegav_diff {
                            Some(expomega)
                        } else {
                            None
                        },
                        if omegav_diff { Some(omega) } else { None },
                        rp_diff,
                    ),
                    [
                        if rp_diff || omegav_diff {
                            dexp_domega
                        } else {
                            None
                        },
                        if omegav_diff { domega } else { None },
                    ],
                )
            }
        }
    }

    fn vjp(
        &self,
        vjp_state: &Self::VjpState,
        alpharp: CotangentFrame<CubeBackend<R, F, I>>,
    ) -> Vec<Option<CotangentFrame<CubeBackend<R, F, I>>>> {
        let alphar = alpharp.clone().slice_dim(1, 0..3);
        let alphap = alpharp.slice_dim(1, 3..6);
        match vjp_state {
            (Some(expomega), Some(omega), true) => {
                let expomega_alphar = expomega.clone().matmul(alphar.clone());
                let expomega_alphap = expomega.clone().matmul(alphap);
                let expvjp_alphar = cube::exp_vjp(omega.clone(), alphar);
                vec![
                    Some(CotangentFrame::cat(
                        vec![expomega_alphar, expomega_alphap.clone()],
                        1,
                    )),
                    Some(CotangentFrame::cat(vec![expvjp_alphar, expomega_alphap], 1)),
                ]
            }
            (Some(expomega), None, true) => {
                let expomega_alphar = expomega.clone().matmul(alphar.clone());
                let expomega_alphap = expomega.clone().matmul(alphap);
                vec![
                    Some(CotangentFrame::cat(
                        vec![expomega_alphar, expomega_alphap.clone()],
                        1,
                    )),
                    None,
                ]
            }
            (Some(expomega), Some(omega), false) => {
                let expomega_alphap = expomega.clone().matmul(alphap);
                let expvjp_alphar = cube::exp_vjp(omega.clone(), alphar);
                vec![
                    None,
                    Some(CotangentFrame::cat(vec![expvjp_alphar, expomega_alphap], 1)),
                ]
            }
            (None, None, false) => vec![None, None],
            _ => unreachable!(),
        }
    }

    fn hvp(
        &self,
        (expomega, omega, rpdiff): &Self::VjpState,
        [dexp_domega, domega]: &Self::HvpState,
        alpharp: CotangentFrame<CubeBackend<R, F, I>>,
    ) -> Vec<Option<CotangentFrame<CubeBackend<R, F, I>>>> {
        let alphar = alpharp.clone().slice_dim(1, 0..3);
        let alphap = alpharp.clone().slice_dim(1, 3..6);

        match (
            *rpdiff,
            expomega.as_ref(),
            omega.as_ref(),
            dexp_domega.as_ref(),
            domega.as_ref(),
        ) {
            (true, Some(expomega), Some(omega), Some(dexp_domega), Some(domega)) => {
                let alphar0 = expomega
                    .clone()
                    .matmul(cube_cross(dexp_domega.clone(), alphar.clone()));
                let alphar1 = cube::exp_hvp(omega.clone(), domega.clone(), alphar);
                let alphap = expomega
                    .clone()
                    .matmul(cube_cross(dexp_domega.clone(), alphap.clone()));

                vec![
                    Some(CotangentFrame::cat(vec![alphar0, alphap.clone()], 1)),
                    Some(CotangentFrame::cat(vec![alphar1, alphap], 1)),
                ]
            }
            (true, Some(_), None, Some(_), None) => {
                vec![Some(alpharp.zeros_like()), None]
            }
            (false, Some(expomega), Some(omega), Some(dexp_domega), Some(domega)) => {
                let alphar1 = cube::exp_hvp(omega.clone(), domega.clone(), alphar);
                let alphap = expomega
                    .clone()
                    .matmul(cube_cross(dexp_domega.clone(), alphap.clone()));

                vec![None, Some(CotangentFrame::cat(vec![alphar1, alphap], 1))]
            }
            (false, None, None, None, None) => vec![None, None],
            _ => unreachable!(),
        }
    }
}

impl<F, I> DiffOperation<NdArrayBackend<F, I>> for SE3Retraction
where
    F: NdArrayFloat + Float,
    I: NdArrayInt,
    NdArrayTensor: From<SharedArray<F>> + From<SharedArray<I>>,
{
    type VjpState = (
        Option<Tensor<NdArrayBackend<F, I>, 3>>,
        Option<Point<NdArrayBackend<F, I>>>,
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
        let [rp, omegav] = inputs.try_into().unwrap();
        let b = rp.dims()[0];
        let r = rp.clone().slice_dim(1, 0..9);
        let p = rp.slice_dim(1, 9..12);
        let omega = omegav.clone().slice_dim(1, 0..3);
        let v = omegav.slice_dim(1, 3..6);
        let p_plus_rv = p + r
            .clone()
            .reshape([b, 3, 3])
            .matmul(v.unsqueeze_dim(2))
            .squeeze_dim(2);
        let rmat = r.reshape([b, 3, 3]);
        let expomega = ndarray::exp(omega.clone()).reshape([b, 3, 3]);
        match diff_state {
            DiffModeState::ForwardOnly => {
                let rexpomega = rmat.matmul(expomega).reshape([b, 9]);
                DiffOperationForward::forward_only(Point::cat(vec![rexpomega, p_plus_rv], 1))
            }
            DiffModeState::Jvp(tvecs) => {
                let [drp, domegav] = tvecs.try_into().unwrap();
                let dr = drp.as_ref().map(|drp| drp.clone().slice_dim(1, 0..3));
                let dp = drp.map(|drp| drp.slice_dim(1, 3..6));
                let domega = domegav
                    .as_ref()
                    .map(|domegav| domegav.clone().slice_dim(1, 0..3));
                let dv = domegav.map(|domegav| domegav.slice_dim(1, 3..6));

                let rexpomega = rmat.matmul(expomega.clone()).reshape([b, 9]);
                DiffOperationForward::jvp(
                    Point::cat(vec![rexpomega, p_plus_rv], 1),
                    sum_opt_tensors([
                        dr.map(|dr| expomega.clone().transpose().matmul(dr)),
                        dp.map(|dp| expomega.clone().matmul(dp)),
                        domega.map(|domega| ndarray::exp_jvp(omega, domega)),
                        dv.map(|dv| expomega.matmul(dv)),
                    ]),
                )
            }
            DiffModeState::Vjp(diffs) => {
                let [rp_diff, omegav_diff] = diffs.try_into().unwrap();
                let rexpomega = rmat.matmul(expomega.clone()).reshape([b, 9]);
                DiffOperationForward::vjp(
                    Point::cat(vec![rexpomega, p_plus_rv], 1),
                    (
                        if rp_diff || omegav_diff {
                            Some(expomega)
                        } else {
                            None
                        },
                        if omegav_diff { Some(omega) } else { None },
                        rp_diff,
                    ),
                )
            }
            DiffModeState::Hvp(tvecs) => {
                let [drp, domegav] = tvecs.try_into().unwrap();

                let rp_diff = drp.is_some();
                let omegav_diff = domegav.is_some();

                let dr = drp.as_ref().map(|drp| drp.clone().slice_dim(1, 0..3));
                let dp = drp.map(|drp| drp.slice_dim(1, 3..6));
                let domega = domegav
                    .as_ref()
                    .map(|domegav| domegav.clone().slice_dim(1, 0..3));
                let dv = domegav.map(|domegav| domegav.slice_dim(1, 3..6));

                let expomega = ndarray::exp(omega.clone()).reshape([b, 3, 3]);
                let dexp_domega = domega.as_ref().map(|domega| {
                    ndarray::exp_jvp(omega.clone(), domega.clone().unsqueeze_dim(2)).squeeze_dim(2)
                });
                let rexpomega = rmat.matmul(expomega.clone()).reshape([b, 9]);
                DiffOperationForward::hvp(
                    Point::cat(vec![rexpomega, p_plus_rv], 1),
                    sum_opt_tensors([
                        dr.map(|dr| {
                            expomega
                                .clone()
                                .transpose()
                                .matmul(dr.unsqueeze_dim(2))
                                .squeeze_dim(2)
                        }),
                        dp.map(|dp| expomega.clone().matmul(dp.unsqueeze_dim(2)).squeeze_dim(2)),
                        dexp_domega.clone(),
                        dv.map(|dv| expomega.clone().matmul(dv.unsqueeze_dim(2)).squeeze_dim(2)),
                    ]),
                    (
                        if rp_diff || omegav_diff {
                            Some(expomega)
                        } else {
                            None
                        },
                        if omegav_diff { Some(omega) } else { None },
                        rp_diff,
                    ),
                    [
                        if rp_diff || omegav_diff {
                            dexp_domega
                        } else {
                            None
                        },
                        if omegav_diff { domega } else { None },
                    ],
                )
            }
        }
    }

    fn vjp(
        &self,
        vjp_state: &Self::VjpState,
        alpharp: CotangentFrame<NdArrayBackend<F, I>>,
    ) -> Vec<Option<CotangentFrame<NdArrayBackend<F, I>>>> {
        let alphar = alpharp.clone().slice_dim(1, 0..3);
        let alphap = alpharp.slice_dim(1, 3..6);
        match vjp_state {
            (Some(expomega), Some(omega), true) => {
                let expomega_alphar = expomega.clone().matmul(alphar.clone());
                let expomega_alphap = expomega.clone().matmul(alphap);
                let expvjp_alphar = ndarray::exp_vjp(omega.clone(), alphar);
                vec![
                    Some(CotangentFrame::cat(
                        vec![expomega_alphar, expomega_alphap.clone()],
                        1,
                    )),
                    Some(CotangentFrame::cat(vec![expvjp_alphar, expomega_alphap], 1)),
                ]
            }
            (Some(expomega), None, true) => {
                let expomega_alphar = expomega.clone().matmul(alphar.clone());
                let expomega_alphap = expomega.clone().matmul(alphap);
                vec![
                    Some(CotangentFrame::cat(
                        vec![expomega_alphar, expomega_alphap.clone()],
                        1,
                    )),
                    None,
                ]
            }
            (Some(expomega), Some(omega), false) => {
                let expomega_alphap = expomega.clone().matmul(alphap);
                let expvjp_alphar = ndarray::exp_vjp(omega.clone(), alphar);
                vec![
                    None,
                    Some(CotangentFrame::cat(vec![expvjp_alphar, expomega_alphap], 1)),
                ]
            }
            (None, None, false) => vec![None, None],
            _ => unreachable!(),
        }
    }

    fn hvp(
        &self,
        (expomega, omega, rpdiff): &Self::VjpState,
        [dexp_domega, domega]: &Self::HvpState,
        alpharp: CotangentFrame<NdArrayBackend<F, I>>,
    ) -> Vec<Option<CotangentFrame<NdArrayBackend<F, I>>>> {
        let alphar = alpharp.clone().slice_dim(1, 0..3);
        let alphap = alpharp.clone().slice_dim(1, 3..6);

        match (
            *rpdiff,
            expomega.as_ref(),
            omega.as_ref(),
            dexp_domega.as_ref(),
            domega.as_ref(),
        ) {
            (true, Some(expomega), Some(omega), Some(dexp_domega), Some(domega)) => {
                let alphar0 = expomega
                    .clone()
                    .matmul(ndarray_cross(dexp_domega.clone(), alphar.clone()));
                let alphar1 = ndarray::exp_hvp(omega.clone(), domega.clone(), alphar);
                let alphap = expomega
                    .clone()
                    .matmul(ndarray_cross(dexp_domega.clone(), alphap.clone()));

                vec![
                    Some(CotangentFrame::cat(vec![alphar0, alphap.clone()], 1)),
                    Some(CotangentFrame::cat(vec![alphar1, alphap], 1)),
                ]
            }
            (true, Some(_), None, Some(_), None) => {
                vec![Some(alpharp.zeros_like()), None]
            }
            (false, Some(expomega), Some(omega), Some(dexp_domega), Some(domega)) => {
                let alphar1 = ndarray::exp_hvp(omega.clone(), domega.clone(), alphar);
                let alphap = expomega
                    .clone()
                    .matmul(ndarray_cross(dexp_domega.clone(), alphap.clone()));

                vec![None, Some(CotangentFrame::cat(vec![alphar1, alphap], 1))]
            }
            (false, None, None, None, None) => vec![None, None],
            _ => unreachable!(),
        }
    }
}
