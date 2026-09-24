use skoro_backend::{Backend, CotangentFrame, Point, TangentVector};
use skoro_core::error::SkoroResult;
use skoro_graph::quantity::ContinuousQuantity;
use skoro_model::{map::register_map2, q::Q};
use skoro_operation::{
    Input,
    diff::{DiffModeState, DiffOperation, DiffOperationForward},
};

use super::body::{apply, apply_transpose, cross, matrix, repeat, sum};
use crate::quantity::SO3;

#[inline(always)]
pub fn compose<B>(r0: &Q<B, SO3>, r1: &Q<B, SO3>) -> SkoroResult<Q<B, SO3>>
where
    B: Backend,
    SO3: ContinuousQuantity<B>,
{
    r0.compose(r1)
}

pub trait SO3ComposeOps<B: Backend>
where
    SO3: ContinuousQuantity<B>,
{
    fn compose(&self, other: &Q<B, SO3>) -> SkoroResult<Q<B, SO3>>;
}

impl<B: Backend> SO3ComposeOps<B> for Q<B, SO3>
where
    SO3: ContinuousQuantity<B>,
{
    fn compose(&self, other: &Q<B, SO3>) -> SkoroResult<Q<B, SO3>> {
        register_map2(SO3, SO3Compose, (self, other))
    }
}

/// `y = r₀ r₁`, differentiated in the body-frame chart of every rotation.
///
/// With perturbations `r₀ Exp(a₀)` and `r₁ Exp(a₁)`,
/// `r₀ Exp(a₀) r₁ Exp(a₁) = r₀ r₁ Exp(r₁ᵀa₀) Exp(a₁)`, so read at `y`:
///
/// ```text
/// JVP  c = r₁ᵀ a₀ + a₁
/// VJP  ∂r₀ = r₁ α          ∂r₁ = α
/// HVP  ∂r₀ → r₁ (a₁ × α)   ∂r₁ → 0
/// ```
///
/// The HVP is the derivative of the VJP along the seeded tangent with `α` held fixed; only `r₁`
/// appears in it, so only a tangent seeded at `r₁` contributes.
#[derive(Debug, Clone, Copy, Default)]
pub struct SO3Compose;

/// `r₁` if `r₀` is differentiable, and whether `r₁` is.
type ComposeVjp<B> = (Option<Point<B>>, bool);

impl<B: Backend> DiffOperation<B> for SO3Compose {
    type VjpState = ComposeVjp<B>;
    /// The tangent seeded at `r₁`, if one was.
    type HvpState = Option<TangentVector<B>>;

    fn forward(
        &self,
        inputs: Vec<Input<B>>,
        diff_state: DiffModeState<B>,
    ) -> DiffOperationForward<B, Self::VjpState, Self::HvpState> {
        let [r0, r1]: [Point<B>; 2] = inputs
            .into_iter()
            .map(Input::unwrap_float)
            .collect::<Vec<_>>()
            .try_into()
            .unwrap();
        let [batch, _] = r0.dims();
        let output = matrix(&r0).matmul(matrix(&r1)).reshape([batch, 9]);
        match diff_state {
            DiffModeState::ForwardOnly => DiffOperationForward::forward_only(output),
            DiffModeState::Jvp(frames) => {
                let [a0, a1] = frames.try_into().unwrap();
                let tangent = sum(a0.map(|a0| apply_transpose(&r1, a0)), a1);
                DiffOperationForward::jvp(output, tangent)
            }
            DiffModeState::Vjp(diffs) => {
                let [d0, d1] = diffs.try_into().unwrap();
                DiffOperationForward::vjp(output, (d0.then_some(r1), d1))
            }
            DiffModeState::Hvp { diffs, tangents } => {
                let [d0, d1] = diffs.try_into().unwrap();
                let [a0, a1] = tangents.try_into().unwrap();
                let frame = |a: TangentVector<B>| a.unsqueeze_dim::<3>(2);
                let tangent = sum(
                    a0.map(|a0| apply_transpose(&r1, frame(a0))),
                    a1.clone().map(frame),
                )
                .map(|t| t.squeeze_dim(2));
                DiffOperationForward::hvp(output, tangent, (d0.then_some(r1), d1), a1)
            }
        }
    }

    fn vjp(
        &self,
        state: &Self::VjpState,
        alpha: CotangentFrame<B>,
    ) -> Vec<Option<CotangentFrame<B>>> {
        let (r1, d1) = state;
        vec![
            r1.as_ref().map(|r1| apply(r1, alpha.clone())),
            d1.then_some(alpha),
        ]
    }

    fn hvp(
        &self,
        vjp_state: &Self::VjpState,
        hvp_state: &Self::HvpState,
        alpha: CotangentFrame<B>,
    ) -> Vec<Option<CotangentFrame<B>>> {
        let frames = alpha.dims()[2];
        let d_r0 = vjp_state
            .0
            .as_ref()
            .zip(hvp_state.as_ref())
            .map(|(r1, a1)| apply(r1, cross(repeat(a1, frames), alpha)));
        vec![d_r0, None]
    }
}
