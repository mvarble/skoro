use skoro_backend::{
    CotangentFrame, CubeBackend, CubeFloat, CubeInt, CubeRuntime, Flex, Point, TangentVector,
    Tensor,
};
use skoro_linalg::{
    vector::cross_product::cube::left_broadcast_cross_product as cube_cross,
    vector::cross_product::flex::left_broadcast_cross_product as host_cross,
};
use skoro_operation::{
    Input,
    diff::{DiffModeState, DiffOperation, DiffOperationForward},
    sum_opt_tensors,
};

use super::SO3Retraction;
use crate::ops::exp::{cube, flex};

impl<R: CubeRuntime, F: CubeFloat, I: CubeInt> DiffOperation<CubeBackend<R, F, I>>
    for SO3Retraction
{
    type VjpState = (
        Option<Tensor<CubeBackend<R, F, I>, 3>>,
        Option<Point<CubeBackend<R, F, I>>>,
    );
    type HvpState = [Option<TangentVector<CubeBackend<R, F, I>>>; 2];

    fn forward(
        &self,
        inputs: Vec<Input<CubeBackend<R, F, I>>>,
        diff_state: DiffModeState<CubeBackend<R, F, I>>,
    ) -> DiffOperationForward<CubeBackend<R, F, I>, Self::VjpState, Self::HvpState> {
        let inputs: Vec<Point<CubeBackend<R, F, I>>> =
            inputs.into_iter().map(Input::unwrap_float).collect();
        let [r, omega] = inputs.try_into().unwrap();
        let b = r.dims()[0];
        let rmat = r.reshape([b, 3, 3]);
        let expomega = cube::exp(omega.clone()).reshape([b, 3, 3]);
        match diff_state {
            DiffModeState::ForwardOnly => {
                DiffOperationForward::forward_only(rmat.matmul(expomega).reshape([b, 9]))
            }
            DiffModeState::Jvp(tvecs) => {
                let [dr, domega] = tvecs.try_into().unwrap();
                DiffOperationForward::jvp(
                    rmat.matmul(expomega.clone()).reshape([b, 9]),
                    sum_opt_tensors([
                        dr.map(|dr| expomega.transpose().matmul(dr)),
                        domega.map(|domega| cube::exp_jvp(omega, domega)),
                    ]),
                )
            }
            DiffModeState::Vjp(diffs) => {
                let [r_diff, omega_diff] = diffs.try_into().unwrap();
                DiffOperationForward::vjp(
                    rmat.matmul(expomega.clone()).reshape([b, 9]),
                    (
                        if r_diff { Some(expomega) } else { None },
                        if omega_diff { Some(omega) } else { None },
                    ),
                )
            }
            DiffModeState::Hvp {
                diffs,
                tangents: tvecs,
            } => {
                let [dr, domega] = tvecs.try_into().unwrap();
                let [r_diff, omega_diff] = diffs.try_into().unwrap();
                let expomega = cube::exp(omega.clone()).reshape([b, 3, 3]);
                let dexp_domega = domega.as_ref().map(|domega| {
                    cube::exp_jvp(omega.clone(), domega.clone().unsqueeze_dim(2)).squeeze_dim(2)
                });
                DiffOperationForward::hvp(
                    rmat.matmul(expomega.clone()).reshape([b, 9]),
                    sum_opt_tensors([
                        dr.map(|dr| {
                            expomega
                                .clone()
                                .transpose()
                                .matmul(dr.unsqueeze_dim(2))
                                .squeeze_dim(2)
                        }),
                        dexp_domega.clone(),
                    ]),
                    // `diffs` (not the tangents) says whether an input traces back to a free
                    // variable, so it is what gates the state `vjp` short-circuits on. A
                    // differentiable input may still arrive with no seeded tangent.
                    (r_diff.then_some(expomega), omega_diff.then_some(omega)),
                    [dexp_domega, domega],
                )
            }
        }
    }

    fn vjp(
        &self,
        (expomega, omega): &Self::VjpState,
        alpha: CotangentFrame<CubeBackend<R, F, I>>,
    ) -> Vec<Option<CotangentFrame<CubeBackend<R, F, I>>>> {
        match (expomega.as_ref(), omega.as_ref()) {
            (Some(expomega), Some(omega)) => vec![
                Some(expomega.clone().matmul(alpha.clone())),
                Some(cube::exp_vjp(omega.clone(), alpha)),
            ],
            (Some(expomega), None) => vec![Some(expomega.clone().matmul(alpha)), None],
            (None, Some(omega)) => vec![None, Some(cube::exp_vjp(omega.clone(), alpha))],
            (None, None) => vec![None, None],
        }
    }

    fn hvp(
        &self,
        (expomega, omega): &Self::VjpState,
        [dexp_domega, domega]: &Self::HvpState,
        alpha: CotangentFrame<CubeBackend<R, F, I>>,
    ) -> Vec<Option<CotangentFrame<CubeBackend<R, F, I>>>> {
        let alpha0 = match (expomega, dexp_domega) {
            (Some(expomega), Some(dexp_domega)) => Some(
                expomega
                    .clone()
                    .matmul(cube_cross(dexp_domega.clone(), alpha.clone())),
            ),
            (Some(_), None) => Some(alpha.zeros_like()),
            _ => None,
        };
        let alpha1 = omega
            .as_ref()
            .zip(domega.as_ref())
            .map(|(omega, domega)| cube::exp_hvp(omega.clone(), domega.clone(), alpha));
        vec![alpha0, alpha1]
    }
}

impl DiffOperation<Flex> for SO3Retraction {
    type VjpState = (Option<Tensor<Flex, 3>>, Option<Point<Flex>>);
    type HvpState = [Option<TangentVector<Flex>>; 2];

    fn forward(
        &self,
        inputs: Vec<Input<Flex>>,
        diff_state: DiffModeState<Flex>,
    ) -> DiffOperationForward<Flex, Self::VjpState, Self::HvpState> {
        let inputs: Vec<Point<Flex>> = inputs.into_iter().map(Input::unwrap_float).collect();
        let [r, omega] = inputs.try_into().unwrap();
        let b = r.dims()[0];
        let rmat = r.reshape([b, 3, 3]);
        let expomega = flex::exp(omega.clone()).reshape([b, 3, 3]);
        match diff_state {
            DiffModeState::ForwardOnly => {
                DiffOperationForward::forward_only(rmat.matmul(expomega).reshape([b, 9]))
            }
            DiffModeState::Jvp(tvecs) => {
                let [dr, domega] = tvecs.try_into().unwrap();
                DiffOperationForward::jvp(
                    rmat.matmul(expomega.clone()).reshape([b, 9]),
                    sum_opt_tensors([
                        dr.map(|dr| expomega.transpose().matmul(dr)),
                        domega.map(|domega| flex::exp_jvp(omega, domega)),
                    ]),
                )
            }
            DiffModeState::Vjp(diffs) => {
                let [r_diff, omega_diff] = diffs.try_into().unwrap();
                DiffOperationForward::vjp(
                    rmat.matmul(expomega.clone()).reshape([b, 9]),
                    (
                        if r_diff { Some(expomega) } else { None },
                        if omega_diff { Some(omega) } else { None },
                    ),
                )
            }
            DiffModeState::Hvp {
                diffs,
                tangents: tvecs,
            } => {
                let [dr, domega] = tvecs.try_into().unwrap();
                let [r_diff, omega_diff] = diffs.try_into().unwrap();
                let expomega = flex::exp(omega.clone()).reshape([b, 3, 3]);
                let dexp_domega = domega.as_ref().map(|domega| {
                    flex::exp_jvp(omega.clone(), domega.clone().unsqueeze_dim(2)).squeeze_dim(2)
                });
                DiffOperationForward::hvp(
                    rmat.matmul(expomega.clone()).reshape([b, 9]),
                    sum_opt_tensors([
                        dr.map(|dr| {
                            expomega
                                .clone()
                                .transpose()
                                .matmul(dr.unsqueeze_dim(2))
                                .squeeze_dim(2)
                        }),
                        dexp_domega.clone(),
                    ]),
                    // `diffs` (not the tangents) says whether an input traces back to a free
                    // variable, so it is what gates the state `vjp` short-circuits on. A
                    // differentiable input may still arrive with no seeded tangent.
                    (r_diff.then_some(expomega), omega_diff.then_some(omega)),
                    [dexp_domega, domega],
                )
            }
        }
    }

    fn vjp(
        &self,
        (expomega, omega): &Self::VjpState,
        alpha: CotangentFrame<Flex>,
    ) -> Vec<Option<CotangentFrame<Flex>>> {
        match (expomega.as_ref(), omega.as_ref()) {
            (Some(expomega), Some(omega)) => vec![
                Some(expomega.clone().matmul(alpha.clone())),
                Some(flex::exp_vjp(omega.clone(), alpha)),
            ],
            (Some(expomega), None) => vec![Some(expomega.clone().matmul(alpha)), None],
            (None, Some(omega)) => vec![None, Some(flex::exp_vjp(omega.clone(), alpha))],
            (None, None) => vec![None, None],
        }
    }

    fn hvp(
        &self,
        (expomega, omega): &Self::VjpState,
        [dexp_domega, domega]: &Self::HvpState,
        alpha: CotangentFrame<Flex>,
    ) -> Vec<Option<CotangentFrame<Flex>>> {
        let alpha0 = match (expomega, dexp_domega) {
            (Some(expomega), Some(dexp_domega)) => Some(
                expomega
                    .clone()
                    .matmul(host_cross(dexp_domega.clone(), alpha.clone())),
            ),
            (Some(_), None) => Some(alpha.zeros_like()),
            _ => None,
        };
        let alpha1 = omega
            .as_ref()
            .zip(domega.as_ref())
            .map(|(omega, domega)| flex::exp_hvp(omega.clone(), domega.clone(), alpha));
        vec![alpha0, alpha1]
    }
}
