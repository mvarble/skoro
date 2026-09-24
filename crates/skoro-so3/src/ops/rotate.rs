use skoro_backend::{Backend, CotangentFrame, Point, TangentVector};
use skoro_core::error::{SkoroError, SkoroResult};
use skoro_graph::quantity::ContinuousQuantity;
use skoro_model::{map::register_map2, q::Q};
use skoro_operation::{
    Input,
    diff::{DiffModeState, DiffOperation, DiffOperationForward},
};
use skoro_riemann::{
    euclidean::{Euclidean, EuclideanD},
    matrix::{EuclideanMatrix, EuclideanMatrixD},
};

use super::body::{apply, apply_transpose, cross, fold, repeat, spread, stack, sum, tile, unstack};
use crate::quantity::SO3;

/// Rotate a vector (or frame of vectors) by an `SO(3)` rotation.
///
/// Returns [`SkoroError::InvalidRotationFrame`] if the operand's leading
/// dimension is not 3.
#[inline(always)]
pub fn rotate<B, T>(r: &Q<B, SO3>, v: &Q<B, T>) -> SkoroResult<Q<B, T>>
where
    B: Backend,
    SO3: ContinuousQuantity<B>,
    T: RotationVector<B>,
{
    r.rotate(v)
}

/// Apply the inverse rotation to a vector (or frame of vectors).
///
/// Returns [`SkoroError::InvalidRotationFrame`] if the operand's leading
/// dimension is not 3.
#[inline(always)]
pub fn unrotate<B, T>(r: &Q<B, SO3>, v: &Q<B, T>) -> SkoroResult<Q<B, T>>
where
    B: Backend,
    SO3: ContinuousQuantity<B>,
    T: RotationVector<B>,
{
    r.unrotate(v)
}

pub trait SO3RotateOps<B: Backend> {
    fn rotate<T>(&self, other: &Q<B, T>) -> SkoroResult<Q<B, T>>
    where
        T: RotationVector<B>;

    fn unrotate<T>(&self, other: &Q<B, T>) -> SkoroResult<Q<B, T>>
    where
        T: RotationVector<B>;
}

impl<B> SO3RotateOps<B> for Q<B, SO3>
where
    B: Backend,
    SO3: ContinuousQuantity<B>,
{
    fn rotate<T>(&self, other: &Q<B, T>) -> SkoroResult<Q<B, T>>
    where
        T: RotationVector<B>,
    {
        let otype = other.get_quantity_type();
        let columns = otype.columns()?;
        register_map2(otype, SO3Rotate(columns), (self, other))
    }

    fn unrotate<T>(&self, other: &Q<B, T>) -> SkoroResult<Q<B, T>>
    where
        T: RotationVector<B>,
    {
        let otype = other.get_quantity_type();
        let columns = otype.columns()?;
        register_map2(otype, SO3Unrotate(columns), (self, other))
    }
}

pub trait RotationVector<B: Backend>: ContinuousQuantity<B> {
    /// The number of 3-vectors this quantity holds: 1 for a vector, `n` for a `3 × n` matrix.
    ///
    /// Returns [`SkoroError::InvalidRotationFrame`] for anything that is not 3 rows.
    fn columns(&self) -> SkoroResult<usize>;
}

impl<B: Backend> RotationVector<B> for EuclideanD {
    fn columns(&self) -> SkoroResult<usize> {
        if self.0 != 3 {
            return Err(SkoroError::InvalidRotationFrame { rows: self.0 });
        }
        Ok(1)
    }
}

impl<B: Backend> RotationVector<B> for Euclidean<3> {
    #[inline(always)]
    fn columns(&self) -> SkoroResult<usize> {
        Ok(1)
    }
}

impl<B: Backend> RotationVector<B> for EuclideanMatrixD {
    fn columns(&self) -> SkoroResult<usize> {
        if self.0 != 3 {
            return Err(SkoroError::InvalidRotationFrame { rows: self.0 });
        }
        Ok(self.1)
    }
}

impl<B: Backend, const N: usize> RotationVector<B> for EuclideanMatrix<3, N> {
    #[inline(always)]
    fn columns(&self) -> SkoroResult<usize> {
        Ok(N)
    }
}

/// What the backward passes of a rotation of `V ∈ ℝ^{3×n}` read: the rotation, `V` or its image,
/// and which inputs are differentiable. Absent when neither is.
#[derive(Debug, Clone)]
pub struct RotationVjp<B: Backend> {
    rotation: Point<B>,
    /// `V` for [`SO3Rotate`], `U = rᵀV` for [`SO3Unrotate`].
    columns: Point<B>,
    d_rotation: bool,
    d_operand: bool,
}

/// The tangents seeded at the rotation and at the operand, each if one was.
type RotationHvp<B> = (Option<TangentVector<B>>, Option<TangentVector<B>>);

/// A tangent vector `[batch, d]` as a one-column frame.
fn one_column<B: Backend>(v: TangentVector<B>) -> skoro_backend::TangentFrame<B> {
    v.unsqueeze_dim(2)
}

/// `Y = r V` for `V ∈ ℝ^{3×n}` (a vector when `n = 1`), differentiated in the body-frame chart of
/// `r`.
///
/// With `r Exp(a)` and `V + B`, `Y` moves by `r (a × V + B)` column by column, so with
/// `w_c = rᵀ α_c` and a rotation's cotangent summed over the columns `c`:
///
/// ```text
/// JVP  Ẏ = r (a × V + B)
/// VJP  ∂r = Σ_c v_c × w_c                  ∂V = rᵀ α
/// HVP  ∂r → Σ_c (b_c × w_c − v_c × (a × w_c))
///      ∂V → −a × w
/// ```
#[derive(Debug, Clone, Copy)]
pub struct SO3Rotate(pub usize);

impl<B: Backend> DiffOperation<B> for SO3Rotate {
    type VjpState = Option<RotationVjp<B>>;
    type HvpState = RotationHvp<B>;

    fn forward(
        &self,
        inputs: Vec<Input<B>>,
        diff_state: DiffModeState<B>,
    ) -> DiffOperationForward<B, Self::VjpState, Self::HvpState> {
        let n = self.0;
        let [r, v]: [Point<B>; 2] = inputs
            .into_iter()
            .map(Input::unwrap_float)
            .collect::<Vec<_>>()
            .try_into()
            .unwrap();
        let [batch, _] = r.dims();
        let y = apply(&r, spread(&v, n, 1)).reshape([batch, 3 * n]);
        // `Ẏ` for a rotation frame `a` and an operand frame `b`, each optional.
        let push = |a: Option<skoro_backend::TangentFrame<B>>,
                    b: Option<skoro_backend::TangentFrame<B>>| {
            let f = a.as_ref().or(b.as_ref()).map(|t| t.dims()[2])?;
            let turned = a.map(|a| cross(tile(a, n), spread(&v, n, f)));
            let moved = b.map(|b| unstack(b, n));
            sum(turned, moved).map(|t| stack(apply(&r, t), n))
        };
        let state = |d_rotation: bool, d_operand: bool| {
            (d_rotation || d_operand).then(|| RotationVjp {
                rotation: r.clone(),
                columns: v.clone(),
                d_rotation,
                d_operand,
            })
        };
        match diff_state {
            DiffModeState::ForwardOnly => DiffOperationForward::forward_only(y),
            DiffModeState::Jvp(frames) => {
                let [a, b] = frames.try_into().unwrap();
                let tangent = push(a, b);
                DiffOperationForward::jvp(y, tangent)
            }
            DiffModeState::Vjp(diffs) => {
                let vjp_state = state(diffs[0], diffs[1]);
                DiffOperationForward::vjp(y, vjp_state)
            }
            DiffModeState::Hvp { diffs, tangents } => {
                let [a, b] = tangents.try_into().unwrap();
                let tangent = push(a.clone().map(one_column), b.clone().map(one_column))
                    .map(|t| t.squeeze_dim(2));
                let vjp_state = state(diffs[0], diffs[1]);
                DiffOperationForward::hvp(y, tangent, vjp_state, (a, b))
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
        let w = apply_transpose(&state.rotation, unstack(alpha, n));
        let d_r = state
            .d_rotation
            .then(|| fold(cross(spread(&state.columns, n, f), w.clone()), n));
        let d_v = state.d_operand.then(|| stack(w, n));
        vec![d_r, d_v]
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
        let (a, b) = hvp_state;
        let f = alpha.dims()[2];
        let w = apply_transpose(&state.rotation, unstack(alpha, n));
        let d_r = state.d_rotation.then(|| {
            let from_b = b
                .as_ref()
                .map(|b| fold(cross(spread(b, n, f), w.clone()), n));
            let from_a = a.as_ref().map(|a| {
                let turned = cross(repeat(a, n * f), w.clone());
                -fold(cross(spread(&state.columns, n, f), turned), n)
            });
            sum(from_b, from_a)
        });
        let d_v = state.d_operand.then(|| {
            a.as_ref()
                .map(|a| stack(-cross(repeat(a, n * f), w.clone()), n))
        });
        vec![d_r.flatten(), d_v.flatten()]
    }
}

/// `U = rᵀ V` for `V ∈ ℝ^{3×n}` (a vector when `n = 1`), differentiated in the body-frame chart of
/// `r`.
///
/// With `r Exp(a)` and `V + B`, `U` moves by `−a × U + rᵀB` column by column, so, summing a
/// rotation's cotangent over the columns `c`:
///
/// ```text
/// JVP  U̇ = −a × U + rᵀ B
/// VJP  ∂r = Σ_c α_c × u_c           ∂V = r α
/// HVP  ∂r → Σ_c α_c × (rᵀ b_c − a × u_c)
///      ∂V → r (a × α)
/// ```
#[derive(Debug, Clone, Copy)]
pub struct SO3Unrotate(pub usize);

impl<B: Backend> DiffOperation<B> for SO3Unrotate {
    type VjpState = Option<RotationVjp<B>>;
    type HvpState = RotationHvp<B>;

    fn forward(
        &self,
        inputs: Vec<Input<B>>,
        diff_state: DiffModeState<B>,
    ) -> DiffOperationForward<B, Self::VjpState, Self::HvpState> {
        let n = self.0;
        let [r, v]: [Point<B>; 2] = inputs
            .into_iter()
            .map(Input::unwrap_float)
            .collect::<Vec<_>>()
            .try_into()
            .unwrap();
        let [batch, _] = r.dims();
        let u = apply_transpose(&r, spread(&v, n, 1)).reshape([batch, 3 * n]);
        // `U̇` for a rotation frame `a` and an operand frame `b`, each optional.
        let push = |a: Option<skoro_backend::TangentFrame<B>>,
                    b: Option<skoro_backend::TangentFrame<B>>| {
            let f = a.as_ref().or(b.as_ref()).map(|t| t.dims()[2])?;
            let rotated = a.map(|a| -cross(tile(a, n), spread(&u, n, f)));
            let moved = b.map(|b| apply_transpose(&r, unstack(b, n)));
            sum(rotated, moved).map(|t| stack(t, n))
        };
        let state = |d_rotation: bool, d_operand: bool| {
            (d_rotation || d_operand).then(|| RotationVjp {
                rotation: r.clone(),
                columns: u.clone(),
                d_rotation,
                d_operand,
            })
        };
        match diff_state {
            DiffModeState::ForwardOnly => DiffOperationForward::forward_only(u),
            DiffModeState::Jvp(frames) => {
                let [a, b] = frames.try_into().unwrap();
                let tangent = push(a, b);
                DiffOperationForward::jvp(u, tangent)
            }
            DiffModeState::Vjp(diffs) => {
                let vjp_state = state(diffs[0], diffs[1]);
                DiffOperationForward::vjp(u, vjp_state)
            }
            DiffModeState::Hvp { diffs, tangents } => {
                let [a, b] = tangents.try_into().unwrap();
                let tangent = push(a.clone().map(one_column), b.clone().map(one_column))
                    .map(|t| t.squeeze_dim(2));
                let vjp_state = state(diffs[0], diffs[1]);
                DiffOperationForward::hvp(u, tangent, vjp_state, (a, b))
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
        let alpha = unstack(alpha, n);
        let d_r = state
            .d_rotation
            .then(|| fold(cross(alpha.clone(), spread(&state.columns, n, f)), n));
        let d_v = state
            .d_operand
            .then(|| stack(apply(&state.rotation, alpha), n));
        vec![d_r, d_v]
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
        let (a, b) = hvp_state;
        let f = alpha.dims()[2];
        let alpha = unstack(alpha, n);
        let d_r = state.d_rotation.then(|| {
            let from_b = b.as_ref().map(|b| {
                fold(
                    cross(
                        alpha.clone(),
                        apply_transpose(&state.rotation, spread(b, n, f)),
                    ),
                    n,
                )
            });
            let from_a = a.as_ref().map(|a| {
                let turned = cross(repeat(a, n * f), spread(&state.columns, n, f));
                -fold(cross(alpha.clone(), turned), n)
            });
            sum(from_b, from_a)
        });
        let d_v = state.d_operand.then(|| {
            a.as_ref().map(|a| {
                stack(
                    apply(&state.rotation, cross(repeat(a, n * f), alpha.clone())),
                    n,
                )
            })
        });
        vec![d_r.flatten(), d_v.flatten()]
    }
}
