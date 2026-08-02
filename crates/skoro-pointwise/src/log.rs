use skoro_backend::{Backend, CotangentFrame, Point, TangentVector};
use skoro_operation::{
    Input,
    diff::{DiffModeState, DiffOperation, DiffOperationForward},
};

#[derive(Debug, Clone, Copy)]
pub struct PointwiseLog;

impl<B: Backend> DiffOperation<B> for PointwiseLog {
    type VjpState = Option<Point<B>>;
    type HvpState = Option<TangentVector<B>>;

    fn forward(
        &self,
        inputs: Vec<Input<B>>,
        diff_state: DiffModeState<B>,
    ) -> DiffOperationForward<B, Self::VjpState, Self::HvpState> {
        let inputs: Vec<Point<B>> = inputs.into_iter().map(Input::into_float).collect();
        let [x] = inputs.try_into().unwrap();
        match diff_state {
            DiffModeState::ForwardOnly => DiffOperationForward::forward_only(x.log()),
            DiffModeState::Jvp(tvecs) => {
                let [dx] = tvecs.try_into().unwrap();
                DiffOperationForward::jvp(x.clone().log(), dx.map(|dx| dx / x.unsqueeze_dim(2)))
            }
            DiffModeState::Vjp(diffs) => {
                let [diff] = diffs.try_into().unwrap();
                if diff {
                    DiffOperationForward::vjp(x.clone().log(), Some(x))
                } else {
                    DiffOperationForward::vjp(x.log(), None)
                }
            }
            DiffModeState::Hvp(tvecs) => {
                let [dx] = tvecs.try_into().unwrap();
                if let Some(dx) = dx {
                    DiffOperationForward::hvp(
                        x.clone().log(),
                        Some(dx.clone() / x.clone()),
                        Some(x),
                        Some(dx),
                    )
                } else {
                    DiffOperationForward::hvp(x.log(), None, None, None)
                }
            }
        }
    }

    fn vjp(
        &self,
        state: &Self::VjpState,
        omega: CotangentFrame<B>,
    ) -> Vec<Option<CotangentFrame<B>>> {
        if let Some(x) = state {
            vec![Some(omega / x.clone().unsqueeze_dim(2))]
        } else {
            vec![None]
        }
    }

    fn hvp(
        &self,
        vjp_state: &Self::VjpState,
        hvp_state: &Self::HvpState,
        omega: CotangentFrame<B>,
    ) -> Vec<Option<CotangentFrame<B>>> {
        if let (Some(x), Some(dx)) = (vjp_state, hvp_state) {
            vec![Some(
                -omega * (dx.clone() / x.clone().square()).unsqueeze_dim(2),
            )]
        } else {
            vec![None]
        }
    }
}
