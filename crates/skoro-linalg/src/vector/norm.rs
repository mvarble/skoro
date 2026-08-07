use burn::tensor::linalg::l2_norm;

use skoro_backend::{Backend, CotangentFrame, Point, TangentVector};
use skoro_operation::{
    Input,
    diff::{DiffModeState, DiffOperation, DiffOperationForward},
};

#[derive(Debug, Clone, Copy)]
pub struct Norm;

impl<B: Backend> DiffOperation<B> for Norm {
    type VjpState = Option<[Point<B>; 2]>;
    type HvpState = Option<TangentVector<B>>;

    fn forward(
        &self,
        inputs: Vec<Input<B>>,
        diff_state: DiffModeState<B>,
    ) -> DiffOperationForward<B, Self::VjpState, Self::HvpState> {
        let inputs: Vec<Point<B>> = inputs.into_iter().map(Input::into_float).collect();
        let [x] = inputs.try_into().unwrap();
        let f = l2_norm(x.clone(), 1);
        match diff_state {
            DiffModeState::ForwardOnly => DiffOperationForward::forward_only(f),
            DiffModeState::Jvp(tvecs) => {
                let [dx] = tvecs.try_into().unwrap();
                DiffOperationForward::jvp(
                    f.clone(),
                    dx.map(|dx| ((x / f).unsqueeze_dim(2) * dx).sum_dim(1)),
                )
            }
            DiffModeState::Vjp(diffs) => {
                let [diff] = diffs.try_into().unwrap();
                DiffOperationForward::vjp(f.clone(), if diff { Some([x, f]) } else { None })
            }
            DiffModeState::Hvp {
                diffs,
                tangents: tvecs,
            } => {
                let [dx] = tvecs.try_into().unwrap();
                DiffOperationForward::hvp(
                    f.clone(),
                    dx.as_ref()
                        .map(|dx| ((x.clone() / f.clone()) * dx.clone()).sum_dim(1)),
                    if diffs[0] { Some([x, f]) } else { None },
                    dx,
                )
            }
        }
    }

    fn vjp(
        &self,
        vjp_state: &Self::VjpState,
        omega: CotangentFrame<B>,
    ) -> Vec<Option<CotangentFrame<B>>> {
        if let Some([x, f]) = vjp_state {
            vec![Some(omega * (x.clone() / f.clone()).unsqueeze_dim(2))]
        } else {
            vec![None]
        }
    }

    fn hvp(
        &self,
        vjp_state: &Self::VjpState,
        dx: &Self::HvpState,
        omega: CotangentFrame<B>,
    ) -> Vec<Option<CotangentFrame<B>>> {
        if let Some(([x, f], dx)) = vjp_state.as_ref().zip(dx.as_ref()) {
            let dx = dx.clone().unsqueeze_dim(2);
            let x = x.clone().unsqueeze_dim(2);
            let f = f.clone().unsqueeze_dim(2);
            vec![Some(
                (omega / f.clone())
                    * (dx.clone() - (x.clone() * x.transpose().matmul(dx)) / (f.clone() * f)),
            )]
        } else {
            vec![None]
        }
    }
}
