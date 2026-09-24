use skoro_backend::{Backend, CotangentFrame, Point, TangentFrame, TangentVector};
use skoro_core::error::SkoroResult;
use skoro_graph::quantity::ContinuousQuantity;
use skoro_model::{map::register_map1, q::Q};
use skoro_operation::{
    Input,
    diff::{DiffModeState, DiffOperation, DiffOperationForward},
};
use skoro_so3::ops::body::{apply, apply_transpose, cross, matrix, repeat};

use super::pose::{join_frame, join_pose, one_column, split_frame, split_pose, split_vector};
use crate::quantity::SE3;

pub trait SE3InvertOps<B: Backend>
where
    SE3: ContinuousQuantity<B>,
{
    /// `self⁻¹`.
    fn invert(&self) -> SkoroResult<Q<B, SE3>>;
}

impl<B: Backend> SE3InvertOps<B> for Q<B, SE3>
where
    SE3: ContinuousQuantity<B>,
{
    fn invert(&self) -> SkoroResult<Q<B, SE3>> {
        register_map1(SE3, SE3Invert, self)
    }
}

/// `(r, p)⁻¹ = (rᵀ, −rᵀ p)`, differentiated in the body-frame chart.
///
/// With `q = rᵀ p`, perturbing by `(a, b)` and reading at the output:
///
/// ```text
/// JVP  c = −r a
///      d = r (a × q − b)
/// VJP  ∂x = (−w_c + q × w_d,  −w_d)                       w = rᵀ α
/// HVP  ∂x → (a × w_c + (b − a × q) × w_d − q × (a × w_d),  a × w_d)
/// ```
#[derive(Debug, Clone, Copy, Default)]
pub struct SE3Invert;

impl<B: Backend> DiffOperation<B> for SE3Invert {
    /// `(r, q)`, if the pose is differentiable.
    type VjpState = Option<(Point<B>, Point<B>)>;
    /// The tangent seeded at the pose, if one was.
    type HvpState = Option<TangentVector<B>>;

    fn forward(
        &self,
        inputs: Vec<Input<B>>,
        diff_state: DiffModeState<B>,
    ) -> DiffOperationForward<B, Self::VjpState, Self::HvpState> {
        let [x]: [Point<B>; 1] = inputs
            .into_iter()
            .map(Input::unwrap_float)
            .collect::<Vec<_>>()
            .try_into()
            .unwrap();
        let (r, p) = split_pose(x);
        let [batch, _] = r.dims();
        let q = apply_transpose(&r, one_column(p)).squeeze_dim(2);
        let output = join_pose(matrix(&r).swap_dims(1, 2).reshape([batch, 9]), -q.clone());
        let push = |t: TangentFrame<B>| {
            let f = t.dims()[2];
            let (a, b) = split_frame(t);
            let c = -apply(&r, a.clone());
            let d = apply(&r, cross(a, repeat(&q, f)) - b);
            join_frame(Some(c), Some(d)).unwrap()
        };
        match diff_state {
            DiffModeState::ForwardOnly => DiffOperationForward::forward_only(output),
            DiffModeState::Jvp(frames) => {
                let [t] = frames.try_into().unwrap();
                DiffOperationForward::jvp(output, t.map(push))
            }
            DiffModeState::Vjp(diffs) => {
                DiffOperationForward::vjp(output, diffs[0].then_some((r, q)))
            }
            DiffModeState::Hvp { diffs, tangents } => {
                let [t] = tangents.try_into().unwrap();
                let tangent = t.clone().map(|t| push(one_column(t)).squeeze_dim(2));
                let vjp_state = diffs[0].then_some((r, q));
                DiffOperationForward::hvp(output, tangent, vjp_state, t)
            }
        }
    }

    fn vjp(
        &self,
        state: &Self::VjpState,
        alpha: CotangentFrame<B>,
    ) -> Vec<Option<CotangentFrame<B>>> {
        let d_x = state.as_ref().map(|(r, q)| {
            let f = alpha.dims()[2];
            let (alpha_c, alpha_d) = split_frame(alpha);
            let (w_c, w_d) = (apply_transpose(r, alpha_c), apply_transpose(r, alpha_d));
            let d_a = -w_c + cross(repeat(q, f), w_d.clone());
            join_frame(Some(d_a), Some(-w_d)).unwrap()
        });
        vec![d_x]
    }

    fn hvp(
        &self,
        vjp_state: &Self::VjpState,
        hvp_state: &Self::HvpState,
        alpha: CotangentFrame<B>,
    ) -> Vec<Option<CotangentFrame<B>>> {
        let d_x = vjp_state
            .as_ref()
            .zip(hvp_state.as_ref())
            .map(|((r, q), t)| {
                let f = alpha.dims()[2];
                let (a, b) = split_vector(t.clone());
                let (a, b, q) = (repeat(&a, f), repeat(&b, f), repeat(q, f));
                let (alpha_c, alpha_d) = split_frame(alpha);
                let (w_c, w_d) = (apply_transpose(r, alpha_c), apply_transpose(r, alpha_d));
                let q_dot = b - cross(a.clone(), q.clone());
                let a_x_wd = cross(a.clone(), w_d.clone());
                let d_a = cross(a, w_c) + cross(q_dot, w_d) - cross(q, a_x_wd.clone());
                join_frame(Some(d_a), Some(a_x_wd)).unwrap()
            });
        vec![d_x]
    }
}
