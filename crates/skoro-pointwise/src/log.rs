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
        let inputs: Vec<Point<B>> = inputs.into_iter().map(Input::unwrap_float).collect();
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
            DiffModeState::Hvp {
                diffs,
                tangents: tvecs,
            } => {
                let [dx] = tvecs.try_into().unwrap();
                let tangent = dx.as_ref().map(|dx| dx.clone() / x.clone());
                let vjp_state = diffs[0].then(|| x.clone());
                DiffOperationForward::hvp(x.log(), tangent, vjp_state, dx)
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
