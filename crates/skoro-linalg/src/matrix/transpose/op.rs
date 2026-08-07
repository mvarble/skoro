use skoro_backend::{Backend, CotangentFrame, Point};
use skoro_operation::{
    Input,
    diff::{DiffModeState, DiffOperation, DiffOperationForward},
};

use super::Transpose;

impl<B: Backend> DiffOperation<B> for Transpose {
    type VjpState = bool;
    type HvpState = ();

    fn forward(
        &self,
        inputs: Vec<Input<B>>,
        diff_state: DiffModeState<B>,
    ) -> DiffOperationForward<B, Self::VjpState, Self::HvpState> {
        let inputs: Vec<Point<B>> = inputs.into_iter().map(Input::into_float).collect();
        let [u] = inputs.try_into().unwrap();
        let [b, _] = u.dims();
        let Transpose(m, n) = *self;
        let u = u.reshape([b, m, n]);
        let outshape = [b, n * m];
        match diff_state {
            DiffModeState::ForwardOnly => {
                DiffOperationForward::forward_only(u.transpose().reshape(outshape))
            }
            DiffModeState::Jvp(tvecs) => {
                let [du] = tvecs.try_into().unwrap();
                DiffOperationForward::jvp(
                    u.clone().transpose().reshape(outshape),
                    du.map(|du| {
                        let f = du.dims()[2];
                        du.swap_dims(1, 2)
                            .reshape([b, f, m, n])
                            .transpose()
                            .reshape([b, f, m * n])
                            .swap_dims(1, 2)
                    }),
                )
            }
            DiffModeState::Vjp(diffs) => {
                let [udiff] = diffs.try_into().unwrap();
                DiffOperationForward::vjp(u.transpose().reshape(outshape), udiff)
            }
            DiffModeState::Hvp {
                diffs,
                tangents: tvecs,
            } => {
                let [du] = tvecs.try_into().unwrap();
                let udiff = diffs[0];
                DiffOperationForward::hvp(
                    u.transpose().reshape(outshape),
                    du.map(|du| du.reshape([b, m, n]).transpose().reshape(outshape)),
                    udiff,
                    (),
                )
            }
        }
    }

    fn vjp(
        &self,
        vjp_state: &Self::VjpState,
        omega: CotangentFrame<B>,
    ) -> Vec<Option<CotangentFrame<B>>> {
        if *vjp_state {
            let [b, _, f] = omega.dims();
            let Transpose(m, n) = *self;
            vec![Some(
                omega
                    .swap_dims(1, 2)
                    .reshape([b, f, n, m])
                    .transpose()
                    .reshape([b, f, m * n])
                    .swap_dims(1, 2),
            )]
        } else {
            vec![None]
        }
    }

    fn hvp(
        &self,
        vjp_state: &Self::VjpState,
        _: &Self::HvpState,
        omega: CotangentFrame<B>,
    ) -> Vec<Option<CotangentFrame<B>>> {
        if *vjp_state {
            vec![Some(omega.zeros_like())]
        } else {
            vec![None]
        }
    }
}
