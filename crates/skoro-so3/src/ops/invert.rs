use skoro_backend::{Backend, CotangentFrame, Point, TangentVector};
use skoro_core::error::SkoroResult;
use skoro_graph::quantity::ContinuousQuantity;
use skoro_model::{map::register_map1, q::Q};
use skoro_operation::{
    Input,
    diff::{DiffModeState, DiffOperation, DiffOperationForward},
};

use super::body::{apply, apply_transpose, cross, matrix, repeat};
use crate::quantity::SO3;

pub trait SO3InvertOps<B: Backend> {
    fn invert(&self) -> SkoroResult<Self>
    where
        Self: Sized;
}

impl<B: Backend> SO3InvertOps<B> for Q<B, SO3>
where
    SO3: ContinuousQuantity<B>,
{
    fn invert(&self) -> SkoroResult<Self> {
        register_map1(SO3, SO3Invert, self)
    }
}

/// `y = r⁻¹ = rᵀ`, differentiated in the body-frame chart.
///
/// `(r Exp(a))ᵀ = Exp(−a) rᵀ = rᵀ Exp(−r a)`, so read at `y`:
///
/// ```text
/// JVP  c = −r a
/// VJP  ∂r = −rᵀ α
/// HVP  ∂r → a × (rᵀ α)
/// ```
///
/// This is also what `transpose` of a rotation means. It is the only such operation: a plain matrix
/// transpose would differentiate the 9-number storage, not the rotation.
#[derive(Debug, Clone, Copy, Default)]
pub struct SO3Invert;

impl<B: Backend> DiffOperation<B> for SO3Invert {
    /// `r`, if it is differentiable.
    type VjpState = Option<Point<B>>;
    /// The tangent seeded at `r`, if one was.
    type HvpState = Option<TangentVector<B>>;

    fn forward(
        &self,
        inputs: Vec<Input<B>>,
        diff_state: DiffModeState<B>,
    ) -> DiffOperationForward<B, Self::VjpState, Self::HvpState> {
        let [r]: [Point<B>; 1] = inputs
            .into_iter()
            .map(Input::unwrap_float)
            .collect::<Vec<_>>()
            .try_into()
            .unwrap();
        let [batch, _] = r.dims();
        let output = matrix(&r).swap_dims(1, 2).reshape([batch, 9]);
        match diff_state {
            DiffModeState::ForwardOnly => DiffOperationForward::forward_only(output),
            DiffModeState::Jvp(frames) => {
                let [a] = frames.try_into().unwrap();
                let tangent = a.map(|a| -apply(&r, a));
                DiffOperationForward::jvp(output, tangent)
            }
            DiffModeState::Vjp(diffs) => DiffOperationForward::vjp(output, diffs[0].then_some(r)),
            DiffModeState::Hvp { diffs, tangents } => {
                let [a] = tangents.try_into().unwrap();
                let tangent = a
                    .clone()
                    .map(|a| (-apply(&r, a.unsqueeze_dim(2))).squeeze_dim(2));
                DiffOperationForward::hvp(output, tangent, diffs[0].then_some(r), a)
            }
        }
    }

    fn vjp(
        &self,
        state: &Self::VjpState,
        alpha: CotangentFrame<B>,
    ) -> Vec<Option<CotangentFrame<B>>> {
        vec![state.as_ref().map(|r| -apply_transpose(r, alpha))]
    }

    fn hvp(
        &self,
        vjp_state: &Self::VjpState,
        hvp_state: &Self::HvpState,
        alpha: CotangentFrame<B>,
    ) -> Vec<Option<CotangentFrame<B>>> {
        let f = alpha.dims()[2];
        vec![
            vjp_state
                .as_ref()
                .zip(hvp_state.as_ref())
                .map(|(r, a)| cross(repeat(a, f), apply_transpose(r, alpha))),
        ]
    }
}
