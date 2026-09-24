use skoro_backend::{Backend, CotangentFrame, Point, TangentFrame, TangentVector};
use skoro_core::error::SkoroResult;
use skoro_model::{map::register_map2, q::Q};
use skoro_operation::{
    Input,
    diff::{DiffModeState, DiffOperation, DiffOperationForward},
};
use skoro_so3::ops::{
    RotationVector,
    body::{apply, apply_transpose, cross, fold, repeat, spread, stack, sum, tile, unstack},
};

use super::pose::{join_frame, one_column, split_frame, split_pose, split_vector};
use crate::quantity::SE3;

pub trait SE3TransformOps<B: Backend> {
    /// `r u + p`: a point (or each column of a `3 × n` matrix) expressed in this pose's frame,
    /// carried out to the frame the pose is expressed in.
    fn transform<T: RotationVector<B>>(&self, other: &Q<B, T>) -> SkoroResult<Q<B, T>>;
}

impl<B: Backend> SE3TransformOps<B> for Q<B, SE3>
where
    SE3: skoro_graph::quantity::ContinuousQuantity<B>,
{
    fn transform<T: RotationVector<B>>(&self, other: &Q<B, T>) -> SkoroResult<Q<B, T>> {
        let otype = other.get_quantity_type();
        let columns = otype.columns()?;
        register_map2(otype, SE3Transform(columns), (self, other))
    }
}

/// What the backward passes of a pose acting on `U ∈ ℝ^{3×n}` read.
#[derive(Debug, Clone)]
pub struct TransformVjp<B: Backend> {
    r: Point<B>,
    /// `U`.
    columns: Point<B>,
    d_pose: bool,
    d_operand: bool,
}

/// The tangents seeded at the pose and at the operand, each if one was.
type TransformHvp<B> = (Option<TangentVector<B>>, Option<TangentVector<B>>);

fn inputs<B: Backend>(inputs: Vec<Input<B>>) -> (Point<B>, Point<B>, Point<B>) {
    let [x, u]: [Point<B>; 2] = inputs
        .into_iter()
        .map(Input::unwrap_float)
        .collect::<Vec<_>>()
        .try_into()
        .unwrap();
    let (r, p) = split_pose(x);
    (r, p, u)
}

/// `Y = r U + p` column by column, differentiated in the body-frame chart of the pose.
///
/// Perturbing the pose by `(a, b)` and `U` by `B`, with `w = rᵀ α` and a pose's cotangent summed
/// over the columns `c`:
///
/// ```text
/// JVP  Ẏ = r (a × U + b + B)
/// VJP  ∂x = (Σ_c u_c × w_c,  Σ_c w_c)                        ∂U = w
/// HVP  ∂x → (Σ_c (B_c × w_c − u_c × (a × w_c)),  −Σ_c a × w_c)
///      ∂U → −a × w
/// ```
///
/// The VJP never reads `p`, so the translation's own tangent `b` reaches no HVP.
#[derive(Debug, Clone, Copy)]
pub struct SE3Transform(pub usize);

impl<B: Backend> DiffOperation<B> for SE3Transform {
    type VjpState = Option<TransformVjp<B>>;
    type HvpState = TransformHvp<B>;

    fn forward(
        &self,
        inputs_: Vec<Input<B>>,
        diff_state: DiffModeState<B>,
    ) -> DiffOperationForward<B, Self::VjpState, Self::HvpState> {
        let n = self.0;
        let (r, p, u) = inputs(inputs_);
        let [batch, _] = r.dims();
        let y = (apply(&r, spread(&u, n, 1)) + repeat(&p, n)).reshape([batch, 3 * n]);
        let push = |t: Option<TangentFrame<B>>, bu: Option<TangentFrame<B>>| {
            let from_pose = t.map(|t| {
                let f = t.dims()[2];
                let (a, b) = split_frame(t);
                cross(tile(a, n), spread(&u, n, f)) + tile(b, n)
            });
            let from_operand = bu.map(|bu| unstack(bu, n));
            sum(from_pose, from_operand).map(|t| stack(apply(&r, t), n))
        };
        let state = |d_pose: bool, d_operand: bool| {
            (d_pose || d_operand).then(|| TransformVjp {
                r: r.clone(),
                columns: u.clone(),
                d_pose,
                d_operand,
            })
        };
        match diff_state {
            DiffModeState::ForwardOnly => DiffOperationForward::forward_only(y),
            DiffModeState::Jvp(frames) => {
                let [t, bu] = frames.try_into().unwrap();
                let tangent = push(t, bu);
                DiffOperationForward::jvp(y, tangent)
            }
            DiffModeState::Vjp(diffs) => {
                let vjp_state = state(diffs[0], diffs[1]);
                DiffOperationForward::vjp(y, vjp_state)
            }
            DiffModeState::Hvp { diffs, tangents } => {
                let [t, bu] = tangents.try_into().unwrap();
                let tangent = push(t.clone().map(one_column), bu.clone().map(one_column))
                    .map(|t| t.squeeze_dim(2));
                let vjp_state = state(diffs[0], diffs[1]);
                DiffOperationForward::hvp(y, tangent, vjp_state, (t, bu))
            }
        }
    }

    fn vjp(
        &self,
        state: &Self::VjpState,
        alpha: CotangentFrame<B>,
    ) -> Vec<Option<CotangentFrame<B>>> {
        let n = self.0;
        let Some(state) = state else {
            return vec![None, None];
        };
        let f = alpha.dims()[2];
        let w = apply_transpose(&state.r, unstack(alpha, n));
        let d_x = state.d_pose.then(|| {
            let d_a = fold(cross(spread(&state.columns, n, f), w.clone()), n);
            join_frame(Some(d_a), Some(fold(w.clone(), n))).unwrap()
        });
        let d_u = state.d_operand.then(|| stack(w, n));
        vec![d_x, d_u]
    }

    fn hvp(
        &self,
        vjp_state: &Self::VjpState,
        hvp_state: &Self::HvpState,
        alpha: CotangentFrame<B>,
    ) -> Vec<Option<CotangentFrame<B>>> {
        let n = self.0;
        let Some(state) = vjp_state else {
            return vec![None, None];
        };
        let (t, bu) = hvp_state;
        let f = alpha.dims()[2];
        let w = apply_transpose(&state.r, unstack(alpha, n));
        let a = t
            .as_ref()
            .map(|t| repeat(&split_vector(t.clone()).0, n * f));
        let d_x = state.d_pose.then(|| {
            let from_operand = bu
                .as_ref()
                .map(|bu| fold(cross(spread(bu, n, f), w.clone()), n));
            let from_pose = a.as_ref().map(|a| {
                -fold(
                    cross(spread(&state.columns, n, f), cross(a.clone(), w.clone())),
                    n,
                )
            });
            let d_b = a.as_ref().map(|a| -fold(cross(a.clone(), w.clone()), n));
            join_frame(sum(from_operand, from_pose), d_b)
        });
        let d_u = state
            .d_operand
            .then(|| a.as_ref().map(|a| stack(-cross(a.clone(), w.clone()), n)));
        vec![d_x.flatten(), d_u.flatten()]
    }
}
