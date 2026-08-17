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
    type HvpState = [Option<Tensor<B, 3>>; 2];

    fn forward(
        &self,
        inputs: Vec<Input<B>>,
        diff_state: DiffModeState<B>,
    ) -> DiffOperationForward<B, Self::VjpState, Self::HvpState> {
        let inputs: Vec<Point<B>> = inputs.into_iter().map(Input::unwrap_float).collect();
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
            DiffModeState::Hvp {
                diffs,
                tangents: tvecs,
            } => {
                let [du, dv] = tvecs.try_into().unwrap();
                let [udiff, vdiff] = diffs.try_into().unwrap();
                let du = du.map(|du| du.reshape([b, m, n]));
                let dv = dv.map(|dv| dv.reshape([b, n, k]));
                let tangent = sum_opt_tensors([
                    du.as_ref()
                        .map(|du| du.clone().matmul(v.clone()).reshape(outshape)),
                    dv.as_ref()
                        .map(|dv| u.clone().matmul(dv.clone()).reshape(outshape)),
                ]);
                DiffOperationForward::hvp(
                    u.clone().matmul(v.clone()).reshape(outshape),
                    tangent,
                    [udiff.then(|| v.clone()), vdiff.then(|| u.clone())],
                    [du, dv],
                )
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
        // `∂(uv)/∂u = ω vᵀ` moves with `dv`, and `∂/∂v = uᵀ ω` moves with `du`, so each output
        // depends on the *other* input's tangent. Both are also gated on the corresponding
        // `VjpState` slot: an input rooted only in constants has a zero second-order cotangent
        // just as it has a zero first-order one.
        let [udiff, vdiff] = vjp_state;
        let [du, dv] = hvp_state;
        let omega = omega.reshape([b, m, k, f]);
        vec![
            udiff.as_ref().and(dv.as_ref()).map(|dv| {
                right_broadcast_matmul(omega.clone(), dv.clone().transpose()).reshape([b, m * n, f])
            }),
            vdiff.as_ref().and(du.as_ref()).map(|du| {
                left_broadcast_matmul(du.clone().transpose(), omega.clone()).reshape([b, n * k, f])
            }),
        ]
    }
}
