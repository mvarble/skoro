use skoro_backend::{Backend, CotangentFrame, Point, ndarray_simd_workaround};
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
        let inputs: Vec<Point<B>> = inputs.into_iter().map(Input::unwrap_float).collect();
        let [x] = inputs.try_into().unwrap();
        // `abs` goes through `ndarray_simd_workaround`: the input is often a `select` result
        // (a `Tril` diagonal, say) that is still held elsewhere, which is the case burn-ndarray's
        // SIMD kernel gets wrong.
        match diff_state {
            DiffModeState::ForwardOnly => {
                DiffOperationForward::forward_only(ndarray_simd_workaround::abs(x))
            }
            DiffModeState::Jvp(tvecs) => {
                let [dx] = tvecs.try_into().unwrap();
                DiffOperationForward::jvp(
                    ndarray_simd_workaround::abs(x.clone()),
                    dx.map(|dx| {
                        dx.full_like(-1.0)
                            .mask_fill(x.unsqueeze_dim(2).greater_equal_elem(0.0), 1.0)
                    }),
                )
            }
            DiffModeState::Vjp(diffs) => {
                let [diff] = diffs.try_into().unwrap();
                if diff {
                    DiffOperationForward::vjp(ndarray_simd_workaround::abs(x.clone()), Some(x))
                } else {
                    DiffOperationForward::vjp(ndarray_simd_workaround::abs(x), None)
                }
            }
            DiffModeState::Hvp {
                diffs,
                tangents: tvecs,
            } => {
                let [dx] = tvecs.try_into().unwrap();
                let tangent = dx.map(|dx| {
                    dx.full_like(-1.0)
                        .mask_fill(x.clone().greater_equal_elem(0.0), 1.0)
                });
                let vjp_state = diffs[0].then(|| x.clone());
                DiffOperationForward::hvp(ndarray_simd_workaround::abs(x), tangent, vjp_state, ())
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
