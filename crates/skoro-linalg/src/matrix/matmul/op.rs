use skoro_backend::{Backend, CotangentFrame, Point, Tensor};
use skoro_operation::{
    Input,
    diff::{DiffModeState, DiffOperation, DiffOperationForward},
    sum_opt_tensors,
};

use super::Matmul;
use crate::matrix::{left_broadcast_matmul, right_broadcast_matmul};

impl<B: Backend> DiffOperation<B> for Matmul {
    type VjpState = [Option<Tensor<B, 3>>; 2];
    type HvpState = Option<[Tensor<B, 3>; 2]>;

    fn forward(
        &self,
        inputs: Vec<Input<B>>,
        diff_state: DiffModeState<B>,
    ) -> DiffOperationForward<B, Self::VjpState, Self::HvpState> {
        let inputs: Vec<Point<B>> = inputs.into_iter().map(Input::into_float).collect();
        let [u, v] = inputs.try_into().unwrap();
        let [b, _] = u.dims();
        let Matmul(m, n, k) = *self;
        let u = u.reshape([b, m, n]);
        let v = v.reshape([b, n, k]);
        let outshape = [b, m * k];
        match diff_state {
            DiffModeState::ForwardOnly => {
                DiffOperationForward::forward_only(u.matmul(v).reshape(outshape))
            }
            DiffModeState::Jvp(tvecs) => {
                let [du, dv] = tvecs.try_into().unwrap();
                DiffOperationForward::jvp(
                    u.clone().matmul(v.clone()).reshape(outshape),
                    sum_opt_tensors([
                        du.map(|du| {
                            let [_, _, f] = du.dims();
                            let outshape = [b, m * k, f];
                            right_broadcast_matmul(du.reshape([b, m, n, f]), v).reshape(outshape)
                        }),
                        dv.map(|dv| {
                            let [_, _, f] = dv.dims();
                            let outshape = [b, m * k, f];
                            left_broadcast_matmul(u, dv.reshape([b, n, k, f])).reshape(outshape)
                        }),
                    ]),
                )
            }
            DiffModeState::Vjp(diffs) => {
                let [udiff, vdiff] = diffs.try_into().unwrap();
                match (udiff, vdiff) {
                    (true, true) => DiffOperationForward::vjp(
                        u.clone().matmul(v.clone()).reshape(outshape),
                        [Some(v), Some(u)],
                    ),
                    (true, false) => DiffOperationForward::vjp(
                        u.matmul(v.clone()).reshape(outshape),
                        [Some(v), None],
                    ),
                    (false, true) => DiffOperationForward::vjp(
                        u.clone().matmul(v).reshape(outshape),
                        [None, Some(u)],
                    ),
                    (false, false) => {
                        DiffOperationForward::vjp(u.matmul(v).reshape(outshape), [None, None])
                    }
                }
            }
            DiffModeState::Hvp(tvecs) => {
                let [du, dv] = tvecs.try_into().unwrap();
                match (du, dv) {
                    (Some(du), Some(dv)) => {
                        let du = du.reshape([b, m, n]);
                        let dv = dv.reshape([b, n, k]);
                        DiffOperationForward::hvp(
                            u.clone().matmul(v.clone()).reshape(outshape),
                            Some(
                                (du.clone().matmul(v.clone()) + u.clone().matmul(dv.clone()))
                                    .reshape(outshape),
                            ),
                            [Some(v), Some(u)],
                            Some([du, dv]),
                        )
                    }
                    (Some(du), None) => {
                        let du = du.reshape([b, m, n]);
                        DiffOperationForward::hvp(
                            u.matmul(v.clone()).reshape(outshape),
                            Some(du.matmul(v.clone()).reshape(outshape)),
                            [Some(v), None],
                            None,
                        )
                    }
                    (None, Some(dv)) => {
                        let dv = dv.reshape([b, n, k]);
                        DiffOperationForward::hvp(
                            u.clone().matmul(v).reshape(outshape),
                            Some(u.clone().matmul(dv).reshape(outshape)),
                            [None, Some(u)],
                            None,
                        )
                    }
                    (None, None) => DiffOperationForward::hvp(
                        u.matmul(v).reshape(outshape),
                        None,
                        [None, None],
                        None,
                    ),
                }
            }
        }
    }

    fn vjp(
        &self,
        vjp_state: &Self::VjpState,
        omega: CotangentFrame<B>,
    ) -> Vec<Option<CotangentFrame<B>>> {
        let [b, _, f] = omega.dims();
        let Matmul(m, n, k) = *self;
        match vjp_state {
            [Some(v), Some(u)] => {
                let omega = omega.reshape([b, m, k, f]);
                let lshape = [b, m * n, f];
                let rshape = [b, n * k, f];
                vec![
                    Some(
                        right_broadcast_matmul(omega.clone(), v.clone().transpose())
                            .reshape(lshape),
                    ),
                    Some(left_broadcast_matmul(u.clone().transpose(), omega).reshape(rshape)),
                ]
            }
            [Some(v), None] => {
                let omega = omega.reshape([b, m, k, f]);
                let lshape = [b, m * n, f];
                vec![
                    Some(right_broadcast_matmul(omega, v.clone().transpose()).reshape(lshape)),
                    None,
                ]
            }
            [None, Some(u)] => {
                let omega = omega.reshape([b, m, k, f]);
                let rshape = [b, n * k, f];
                vec![
                    None,
                    Some(left_broadcast_matmul(u.clone().transpose(), omega).reshape(rshape)),
                ]
            }
            [None, None] => vec![None, None],
        }
    }

    fn hvp(
        &self,
        vjp_state: &Self::VjpState,
        hvp_state: &Self::HvpState,
        omega: CotangentFrame<B>,
    ) -> Vec<Option<CotangentFrame<B>>> {
        let [b, _, f] = omega.dims();
        let Matmul(m, n, k) = *self;
        match (vjp_state, hvp_state) {
            ([Some(_), Some(_)], Some([du, dv])) => {
                let omega = omega.reshape([b, m, k, f]);
                let lshape = [b, m * n, f];
                let rshape = [b, n * k, f];
                vec![
                    Some(
                        right_broadcast_matmul(omega.clone(), dv.clone().transpose())
                            .reshape(lshape),
                    ),
                    Some(left_broadcast_matmul(du.clone().transpose(), omega).reshape(rshape)),
                ]
            }
            ([Some(_), None], _) => {
                vec![
                    Some(Tensor::<B, 3>::zeros([b, m * n, f], &omega.device())),
                    None,
                ]
            }
            ([None, Some(_)], _) => {
                vec![
                    None,
                    Some(Tensor::<B, 3>::zeros([b, n * k, f], &omega.device())),
                ]
            }
            ([None, None], _) => {
                vec![None, None]
            }
            _ => unreachable!(),
        }
    }
}
