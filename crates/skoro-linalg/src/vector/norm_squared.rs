use burn::tensor::linalg::l2_norm;

use skoro_backend::{Backend, CotangentFrame, Point, TangentVector};
use skoro_operation::{
    Input,
    diff::{DiffModeState, DiffOperation, DiffOperationForward},
};

#[derive(Debug, Clone, Copy)]
pub struct NormSquared;

impl<B: Backend> DiffOperation<B> for NormSquared {
    type VjpState = Option<Point<B>>;
    type HvpState = Option<TangentVector<B>>;

    fn forward(
        &self,
        inputs: Vec<Input<B>>,
        diff_state: DiffModeState<B>,
    ) -> DiffOperationForward<B, Self::VjpState, Self::HvpState> {
        let inputs: Vec<Point<B>> = inputs.into_iter().map(Input::unwrap_float).collect();
        let [x] = inputs.try_into().unwrap();
        let f = l2_norm(x.clone(), 1).square();
        match diff_state {
            DiffModeState::ForwardOnly => DiffOperationForward::forward_only(f),
            DiffModeState::Jvp(tvecs) => {
                let [dx] = tvecs.try_into().unwrap();
                DiffOperationForward::jvp(
                    f,
                    dx.map(|dx| 2.0 * (x.unsqueeze_dim(2) * dx).sum_dim(1)),
                )
            }
            DiffModeState::Vjp(diffs) => {
                let [diff] = diffs.try_into().unwrap();
                DiffOperationForward::vjp(f, if diff { Some(x) } else { None })
            }
            DiffModeState::Hvp {
                diffs,
                tangents: tvecs,
            } => {
                let [dx] = tvecs.try_into().unwrap();
                let vjp_state = if diffs[0] { Some(x.clone()) } else { None };
                DiffOperationForward::hvp(
                    f,
                    dx.as_ref().map(|dx| 2.0 * (x * dx.clone()).sum_dim(1)),
                    vjp_state,
                    dx,
                )
            }
        }
    }

    fn vjp(
        &self,
        state: &Self::VjpState,
        omega: CotangentFrame<B>,
    ) -> Vec<Option<CotangentFrame<B>>> {
        if let Some(x) = state {
            vec![Some(2.0 * omega * x.clone().unsqueeze_dim(2))]
        } else {
            vec![None]
        }
    }

    fn hvp(
        &self,
        _: &Self::VjpState,
        dx: &Self::HvpState,
        omega: CotangentFrame<B>,
    ) -> Vec<Option<CotangentFrame<B>>> {
        if let Some(dx) = dx {
            vec![Some(2.0 * omega * dx.clone().unsqueeze_dim(2))]
        } else {
            vec![None]
        }
    }
}
