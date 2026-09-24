use skoro_backend::{Backend, CotangentFrame, Point, TangentFrame, TangentVector};
use skoro_core::error::SkoroResult;
use skoro_graph::quantity::ContinuousQuantity;
use skoro_model::{map::register_map2, q::Q};
use skoro_operation::{
    Input,
    diff::{DiffModeState, DiffOperation, DiffOperationForward},
};
use skoro_so3::ops::body::{apply, apply_transpose, cross, matrix, repeat, sum};

use super::pose::{join_frame, join_pose, one_column, split_frame, split_pose, split_vector};
use crate::quantity::SE3;

pub trait SE3ComposeOps<B: Backend>
where
    SE3: ContinuousQuantity<B>,
{
    /// `self ∘ other`: the pose `other` expressed in `self`'s frame.
    fn compose(&self, other: &Q<B, SE3>) -> SkoroResult<Q<B, SE3>>;
}

impl<B: Backend> SE3ComposeOps<B> for Q<B, SE3>
where
    SE3: ContinuousQuantity<B>,
{
    fn compose(&self, other: &Q<B, SE3>) -> SkoroResult<Q<B, SE3>> {
        register_map2(SE3, SE3Compose, (self, other))
    }
}

/// `(r₀, p₀) ∘ (r₁, p₁) = (r₀ r₁, p₀ + r₀ p₁)`, differentiated in the body-frame chart of every
/// pose.
///
/// Perturbing `x₀` by `(a₀, b₀)` and `x₁` by `(a₁, b₁)` and reading the result in the chart at the
/// output:
///
/// ```text
/// JVP  c = r₁ᵀ a₀ + a₁
///      d = r₁ᵀ (b₀ + a₀ × p₁) + b₁
/// VJP  ∂x₀ = (r₁ α_c + p₁ × (r₁ α_d),  r₁ α_d)
///      ∂x₁ = (α_c, α_d)
/// HVP  ∂x₀ → (r₁ (a₁ × α_c) + r₁ (b₁ × α_d) + p₁ × (r₁ (a₁ × α_d)),  r₁ (a₁ × α_d))
///      ∂x₁ → 0
/// ```
///
/// The VJP reads only `x₁`, so only a tangent seeded at `x₁` reaches the HVP.
#[derive(Debug, Clone, Copy, Default)]
pub struct SE3Compose;

/// `(r₁, p₁)`, and which of the two poses is differentiable.
#[derive(Debug, Clone)]
pub struct ComposeVjp<B: Backend> {
    r1: Point<B>,
    p1: Point<B>,
    d0: bool,
    d1: bool,
}

impl<B: Backend> DiffOperation<B> for SE3Compose {
    type VjpState = Option<ComposeVjp<B>>;
    /// The tangent seeded at `x₁`, if one was.
    type HvpState = Option<TangentVector<B>>;

    fn forward(
        &self,
        inputs: Vec<Input<B>>,
        diff_state: DiffModeState<B>,
    ) -> DiffOperationForward<B, Self::VjpState, Self::HvpState> {
        let [x0, x1]: [Point<B>; 2] = inputs
            .into_iter()
            .map(Input::unwrap_float)
            .collect::<Vec<_>>()
            .try_into()
            .unwrap();
        let (r0, p0) = split_pose(x0);
        let (r1, p1) = split_pose(x1);
        let [batch, _] = r0.dims();
        let r = matrix(&r0).matmul(matrix(&r1)).reshape([batch, 9]);
        let p = p0 + apply(&r0, one_column(p1.clone())).squeeze_dim(2);
        let output = join_pose(r, p);
        let push = |t0: Option<TangentFrame<B>>, t1: Option<TangentFrame<B>>| {
            let from_x0 = t0.map(|t0| {
                let f = t0.dims()[2];
                let (a0, b0) = split_frame(t0);
                let c = apply_transpose(&r1, a0.clone());
                let d = apply_transpose(&r1, b0 + cross(a0, repeat(&p1, f)));
                join_frame(Some(c), Some(d)).unwrap()
            });
            sum(from_x0, t1)
        };
        let state = |d0: bool, d1: bool| {
            (d0 || d1).then(|| ComposeVjp {
                r1: r1.clone(),
                p1: p1.clone(),
                d0,
                d1,
            })
        };
        match diff_state {
            DiffModeState::ForwardOnly => DiffOperationForward::forward_only(output),
            DiffModeState::Jvp(frames) => {
                let [t0, t1] = frames.try_into().unwrap();
                let tangent = push(t0, t1);
                DiffOperationForward::jvp(output, tangent)
            }
            DiffModeState::Vjp(diffs) => {
                let vjp_state = state(diffs[0], diffs[1]);
                DiffOperationForward::vjp(output, vjp_state)
            }
            DiffModeState::Hvp { diffs, tangents } => {
                let [t0, t1] = tangents.try_into().unwrap();
                let tangent =
                    push(t0.map(one_column), t1.clone().map(one_column)).map(|t| t.squeeze_dim(2));
                let vjp_state = state(diffs[0], diffs[1]);
                DiffOperationForward::hvp(output, tangent, vjp_state, t1)
            }
        }
    }

    fn vjp(
        &self,
        state: &Self::VjpState,
        alpha: CotangentFrame<B>,
    ) -> Vec<Option<CotangentFrame<B>>> {
        let Some(state) = state else {
            return vec![None, None];
        };
        let f = alpha.dims()[2];
        let d_x0 = state.d0.then(|| {
            let (alpha_c, alpha_d) = split_frame(alpha.clone());
            let turned_d = apply(&state.r1, alpha_d);
            let d_a = apply(&state.r1, alpha_c) + cross(repeat(&state.p1, f), turned_d.clone());
            join_frame(Some(d_a), Some(turned_d)).unwrap()
        });
        let d_x1 = state.d1.then_some(alpha);
        vec![d_x0, d_x1]
    }

    fn hvp(
        &self,
        vjp_state: &Self::VjpState,
        hvp_state: &Self::HvpState,
        alpha: CotangentFrame<B>,
    ) -> Vec<Option<CotangentFrame<B>>> {
        let Some(state) = vjp_state else {
            return vec![None, None];
        };
        let f = alpha.dims()[2];
        let d_x0 = state
            .d0
            .then(|| {
                hvp_state.as_ref().map(|t1| {
                    let (a1, b1) = split_vector(t1.clone());
                    let (alpha_c, alpha_d) = split_frame(alpha);
                    let (a1, b1) = (repeat(&a1, f), repeat(&b1, f));
                    let turned_d = apply(&state.r1, cross(a1.clone(), alpha_d.clone()));
                    let d_a = apply(&state.r1, cross(a1, alpha_c))
                        + apply(&state.r1, cross(b1, alpha_d))
                        + cross(repeat(&state.p1, f), turned_d.clone());
                    join_frame(Some(d_a), Some(turned_d)).unwrap()
                })
            })
            .flatten();
        vec![d_x0, None]
    }
}
