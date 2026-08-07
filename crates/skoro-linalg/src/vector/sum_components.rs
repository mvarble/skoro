use skoro_backend::{Backend, CotangentFrame, Point};
use skoro_operation::{
    Input,
    diff::{DiffModeState, DiffOperation, DiffOperationForward},
};

#[derive(Debug, Clone, Copy)]
pub struct SumComponents;

impl<B: Backend> DiffOperation<B> for SumComponents {
    type VjpState = Option<usize>;
    type HvpState = ();

    fn forward(
        &self,
        inputs: Vec<Input<B>>,
        diff_state: DiffModeState<B>,
    ) -> DiffOperationForward<B, Self::VjpState, Self::HvpState> {
        let inputs: Vec<Point<B>> = inputs.into_iter().map(Input::into_float).collect();
        let [x] = inputs.try_into().unwrap();
        match diff_state {
            DiffModeState::ForwardOnly => DiffOperationForward::forward_only(x.sum_dim(1)),
            DiffModeState::Jvp(tvecs) => {
                let [dx] = tvecs.try_into().unwrap();
                DiffOperationForward::jvp(x.sum_dim(1), dx.map(|dx| dx.sum_dim(1)))
            }
            DiffModeState::Vjp(diffs) => {
                let [diff] = diffs.try_into().unwrap();
                let dim = if diff { Some(x.dims()[1]) } else { None };
                DiffOperationForward::vjp(x.sum_dim(1), dim)
            }
            DiffModeState::Hvp {
                diffs,
                tangents: tvecs,
            } => {
                let [dx] = tvecs.try_into().unwrap();
                let dim = if diffs[0] { Some(x.dims()[1]) } else { None };
                DiffOperationForward::hvp(x.sum_dim(1), dx.map(|dx| dx.sum_dim(1)), dim, ())
            }
        }
    }

    fn vjp(
        &self,
        dim: &Self::VjpState,
        omega: CotangentFrame<B>,
    ) -> Vec<Option<CotangentFrame<B>>> {
        vec![dim.as_ref().map(|dim| omega.repeat_dim(1, *dim))]
    }

    fn hvp(
        &self,
        dim: &Self::VjpState,
        _: &Self::HvpState,
        omega: CotangentFrame<B>,
    ) -> Vec<Option<CotangentFrame<B>>> {
        vec![
            dim.as_ref()
                .map(|dim| omega.zeros_like().repeat_dim(1, *dim)),
        ]
    }
}
