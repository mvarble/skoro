use skoro_backend::{Backend, CotangentFrame, Point};
use skoro_operation::{
    Input,
    diff::{DiffModeState, DiffOperation, DiffOperationForward},
};

#[derive(Debug, Clone, Copy)]
pub struct PointwiseAbs;

impl<B: Backend> DiffOperation<B> for PointwiseAbs {
    type VjpState = Option<Point<B>>;
    type HvpState = ();

    fn forward(
        &self,
        inputs: Vec<Input<B>>,
        diff_state: DiffModeState<B>,
    ) -> DiffOperationForward<B, Self::VjpState, Self::HvpState> {
        let inputs: Vec<Point<B>> = inputs.into_iter().map(Input::into_float).collect();
        let [x] = inputs.try_into().unwrap();
        match diff_state {
            DiffModeState::ForwardOnly => DiffOperationForward::forward_only(x.abs()),
            DiffModeState::Jvp(tvecs) => {
                let [dx] = tvecs.try_into().unwrap();
                DiffOperationForward::jvp(
                    x.clone().abs(),
                    dx.map(|dx| {
                        dx.full_like(-1.0)
                            .mask_fill(x.unsqueeze_dim(2).greater_equal_elem(0.0), 1.0)
                    }),
                )
            }
            DiffModeState::Vjp(diffs) => {
                let [diff] = diffs.try_into().unwrap();
                if diff {
                    DiffOperationForward::vjp(x.clone().abs(), Some(x))
                } else {
                    DiffOperationForward::vjp(x.abs(), None)
                }
            }
            DiffModeState::Hvp(tvecs) => {
                let [dx] = tvecs.try_into().unwrap();
                if let Some(dx) = dx {
                    DiffOperationForward::hvp(
                        x.clone().abs(),
                        Some(
                            dx.full_like(-1.0)
                                .mask_fill(x.clone().greater_equal_elem(0.0), 1.0),
                        ),
                        Some(x),
                        (),
                    )
                } else {
                    DiffOperationForward::hvp(x.abs(), None, None, ())
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
            let mask = x.clone().unsqueeze_dim(2).greater_equal_elem(0.0);
            let neg_omega = -omega.clone();
            vec![Some(neg_omega.mask_where(mask, omega))]
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
        if vjp_state.is_some() {
            vec![Some(omega.zeros_like())]
        } else {
            vec![None]
        }
    }
}
