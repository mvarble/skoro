use skoro_backend::{
    CotangentFrame, CubeBackend, CubeFloat, CubeInt, CubeRuntime, NdArrayBackend, NdArrayFloat,
    NdArrayInt, NdArrayTensor, Point, SharedArray, TangentVector, Tensor,
};
use skoro_linalg::vector::cross_product::cube::left_broadcast_cross_product as cube_cross;
use skoro_linalg::vector::cross_product::ndarray::left_broadcast_cross_product as ndarray_cross;
use skoro_operation::{
    Input,
    diff::{DiffModeState, DiffOperation, DiffOperationForward},
    sum_opt_tensors,
};

use crate::ops::log::{cube, ndarray};

use super::SO3InverseRetraction;

// The inverse of the `SO(3)` retraction `R(r, ω) = r Exp(ω)`:
// ```
// Rinv(r₀, r₁) = Log(r₀ᵀ r₁) = ω,   Δr = r₀ᵀ r₁
// ```
// Both inputs carry the body-frame chart `a ↦ r Exp(a)`; the output is Euclidean.
//
// **JVP.** With input tangents `a₀` and `a₁`, and `h = a₁ - Δrᵀ a₀`,
// ```
// dω = J⁻¹(ω) h
// ```
//
// **VJP.** Against a cotangent `α`, writing `vjp = J⁻ᵀ(ω) α`,
// ```
// ∂r₀ = -Δr vjp,   ∂r₁ = vjp
// ```
//
// **HVP.** Using `dΔr/dt = -[a₀]× Δr + Δr [a₁]×`,
// ```
// ∇∂r₀ = a₀ × (Δr vjp) - Δr (a₁ × vjp) - Δr ∇ₕ vjp
// ∇∂r₁ = ∇ₕ vjp
// ```
// where `∇ₕ vjp = log_hvp(ω, J⁻¹(ω) h, α)`. Note the `J⁻¹(ω)`: `log_hvp` differentiates along an
// **ω-space** velocity, whereas `h` is a group-side perturbation of `Δr` (cf. `SO3Log::forward`,
// which feeds it `log_jvp(ω, ·)`). The two coincide only as `ω → 0`, which is why a fixture with
// nearly-equal attitudes hides the difference.
impl<R: CubeRuntime, F: CubeFloat, I: CubeInt> DiffOperation<CubeBackend<R, F, I>>
    for SO3InverseRetraction
{
    /// `(ω, Δr)` for each input, present exactly when that input is differentiable.
    type VjpState = (
        Option<(Point<CubeBackend<R, F, I>>, Tensor<CubeBackend<R, F, I>, 3>)>,
        Option<(Point<CubeBackend<R, F, I>>, Tensor<CubeBackend<R, F, I>, 3>)>,
    );
    type HvpState = [Option<TangentVector<CubeBackend<R, F, I>>>; 2];

    fn forward(
        &self,
        inputs: Vec<Input<CubeBackend<R, F, I>>>,
        diff_state: DiffModeState<CubeBackend<R, F, I>>,
    ) -> DiffOperationForward<CubeBackend<R, F, I>, Self::VjpState, Self::HvpState> {
        let inputs: Vec<Point<CubeBackend<R, F, I>>> =
            inputs.into_iter().map(Input::into_float).collect();
        let [u, v] = inputs.try_into().unwrap();
        let b = u.dims()[0];
        let dr = u
            .reshape([b, 3, 3])
            .transpose()
            .matmul(v.reshape([b, 3, 3]));
        let omega = cube::log(dr.clone().reshape([b, 9]));
        match diff_state {
            DiffModeState::ForwardOnly => DiffOperationForward::forward_only(omega),
            DiffModeState::Jvp(tvecs) => {
                let [du, dv] = tvecs.try_into().unwrap();
                DiffOperationForward::jvp(
                    omega.clone(),
                    sum_opt_tensors([
                        du.map(|du| {
                            -cube::log_jvp(omega.clone(), dr.clone().transpose().matmul(du))
                        }),
                        dv.map(|dv| cube::log_jvp(omega.clone(), dv)),
                    ]),
                )
            }
            DiffModeState::Vjp(diffs) => {
                let [udiff, vdiff] = diffs.try_into().unwrap();
                DiffOperationForward::vjp(
                    omega.clone(),
                    (
                        udiff.then(|| (omega.clone(), dr.clone())),
                        vdiff.then_some((omega, dr)),
                    ),
                )
            }
            DiffModeState::Hvp {
                diffs,
                tangents: tvecs,
            } => {
                let [du, dv] = tvecs.try_into().unwrap();
                let [udiff, vdiff] = diffs.try_into().unwrap();
                DiffOperationForward::hvp(
                    omega.clone(),
                    sum_opt_tensors([
                        du.as_ref().map(|du| {
                            -cube::log_jvp(
                                omega.clone(),
                                dr.clone().transpose().matmul(du.clone().unsqueeze_dim(2)),
                            )
                            .squeeze_dim(2)
                        }),
                        dv.as_ref().map(|dv| {
                            cube::log_jvp(omega.clone(), dv.clone().unsqueeze_dim(2)).squeeze_dim(2)
                        }),
                    ]),
                    // `diffs` (not the tangents) says whether an input traces back to a free
                    // variable, so it is what gates the state `vjp` short-circuits on. A
                    // differentiable input may still arrive with no seeded tangent.
                    (
                        udiff.then(|| (omega.clone(), dr.clone())),
                        vdiff.then_some((omega, dr)),
                    ),
                    [du, dv],
                )
            }
        }
    }

    fn vjp(
        &self,
        (u_state, v_state): &Self::VjpState,
        alpha: CotangentFrame<CubeBackend<R, F, I>>,
    ) -> Vec<Option<CotangentFrame<CubeBackend<R, F, I>>>> {
        vec![
            u_state.as_ref().map(|(omega, dr)| {
                -dr.clone()
                    .matmul(cube::log_vjp(omega.clone(), alpha.clone()))
            }),
            v_state
                .as_ref()
                .map(|(omega, _)| cube::log_vjp(omega.clone(), alpha.clone())),
        ]
    }

    fn hvp(
        &self,
        (u_state, v_state): &Self::VjpState,
        [du, dv]: &Self::HvpState,
        alpha: CotangentFrame<CubeBackend<R, F, I>>,
    ) -> Vec<Option<CotangentFrame<CubeBackend<R, F, I>>>> {
        // `ω` and `Δr` are shared by both blocks; either state slot exposes them.
        let shared = u_state.as_ref().or(v_state.as_ref());

        // `∇ₕ vjp`, with `h = a₁ - Δrᵀ a₀` pushed through `J⁻¹(ω)` into an ω-space velocity.
        let log_hvp = shared.and_then(|(omega, dr)| {
            let h = sum_opt_tensors([
                du.as_ref().map(|du| {
                    -dr.clone()
                        .transpose()
                        .matmul(du.clone().unsqueeze_dim(2))
                        .squeeze_dim(2)
                }),
                dv.clone(),
            ])?;
            let eta = cube::log_jvp(omega.clone(), h.unsqueeze_dim(2)).squeeze_dim(2);
            Some(cube::log_hvp(omega.clone(), eta, alpha.clone()))
        });

        vec![
            u_state.as_ref().map(|(omega, dr)| {
                let vjp = cube::log_vjp(omega.clone(), alpha.clone());
                sum_opt_tensors([
                    du.as_ref()
                        .map(|du| cube_cross(du.clone(), dr.clone().matmul(vjp.clone()))),
                    dv.as_ref()
                        .map(|dv| -dr.clone().matmul(cube_cross(dv.clone(), vjp.clone()))),
                    log_hvp.clone().map(|log_hvp| -dr.clone().matmul(log_hvp)),
                ])
                .unwrap_or_else(|| alpha.zeros_like())
            }),
            v_state
                .as_ref()
                .map(|_| log_hvp.clone().unwrap_or_else(|| alpha.zeros_like())),
        ]
    }
}

impl<F, I> DiffOperation<NdArrayBackend<F, I>> for SO3InverseRetraction
where
    F: NdArrayFloat + num_traits::Float,
    I: NdArrayInt,
    NdArrayTensor: From<SharedArray<F>> + From<SharedArray<I>>,
{
    /// `(ω, Δr)` for each input, present exactly when that input is differentiable.
    type VjpState = (
        Option<(Point<NdArrayBackend<F, I>>, Tensor<NdArrayBackend<F, I>, 3>)>,
        Option<(Point<NdArrayBackend<F, I>>, Tensor<NdArrayBackend<F, I>, 3>)>,
    );
    type HvpState = [Option<TangentVector<NdArrayBackend<F, I>>>; 2];

    fn forward(
        &self,
        inputs: Vec<Input<NdArrayBackend<F, I>>>,
        diff_state: DiffModeState<NdArrayBackend<F, I>>,
    ) -> DiffOperationForward<NdArrayBackend<F, I>, Self::VjpState, Self::HvpState> {
        let inputs: Vec<Point<NdArrayBackend<F, I>>> =
            inputs.into_iter().map(Input::into_float).collect();
        let [u, v] = inputs.try_into().unwrap();
        let b = u.dims()[0];
        let dr = u
            .reshape([b, 3, 3])
            .transpose()
            .matmul(v.reshape([b, 3, 3]));
        let omega = ndarray::log(dr.clone().reshape([b, 9]));
        match diff_state {
            DiffModeState::ForwardOnly => DiffOperationForward::forward_only(omega),
            DiffModeState::Jvp(tvecs) => {
                let [du, dv] = tvecs.try_into().unwrap();
                DiffOperationForward::jvp(
                    omega.clone(),
                    sum_opt_tensors([
                        du.map(|du| {
                            -ndarray::log_jvp(omega.clone(), dr.clone().transpose().matmul(du))
                        }),
                        dv.map(|dv| ndarray::log_jvp(omega.clone(), dv)),
                    ]),
                )
            }
            DiffModeState::Vjp(diffs) => {
                let [udiff, vdiff] = diffs.try_into().unwrap();
                DiffOperationForward::vjp(
                    omega.clone(),
                    (
                        udiff.then(|| (omega.clone(), dr.clone())),
                        vdiff.then_some((omega, dr)),
                    ),
                )
            }
            DiffModeState::Hvp {
                diffs,
                tangents: tvecs,
            } => {
                let [du, dv] = tvecs.try_into().unwrap();
                let [udiff, vdiff] = diffs.try_into().unwrap();
                DiffOperationForward::hvp(
                    omega.clone(),
                    sum_opt_tensors([
                        du.as_ref().map(|du| {
                            -ndarray::log_jvp(
                                omega.clone(),
                                dr.clone().transpose().matmul(du.clone().unsqueeze_dim(2)),
                            )
                            .squeeze_dim(2)
                        }),
                        dv.as_ref().map(|dv| {
                            ndarray::log_jvp(omega.clone(), dv.clone().unsqueeze_dim(2))
                                .squeeze_dim(2)
                        }),
                    ]),
                    // `diffs` (not the tangents) says whether an input traces back to a free
                    // variable, so it is what gates the state `vjp` short-circuits on. A
                    // differentiable input may still arrive with no seeded tangent.
                    (
                        udiff.then(|| (omega.clone(), dr.clone())),
                        vdiff.then_some((omega, dr)),
                    ),
                    [du, dv],
                )
            }
        }
    }

    fn vjp(
        &self,
        (u_state, v_state): &Self::VjpState,
        alpha: CotangentFrame<NdArrayBackend<F, I>>,
    ) -> Vec<Option<CotangentFrame<NdArrayBackend<F, I>>>> {
        vec![
            u_state.as_ref().map(|(omega, dr)| {
                -dr.clone()
                    .matmul(ndarray::log_vjp(omega.clone(), alpha.clone()))
            }),
            v_state
                .as_ref()
                .map(|(omega, _)| ndarray::log_vjp(omega.clone(), alpha.clone())),
        ]
    }

    fn hvp(
        &self,
        (u_state, v_state): &Self::VjpState,
        [du, dv]: &Self::HvpState,
        alpha: CotangentFrame<NdArrayBackend<F, I>>,
    ) -> Vec<Option<CotangentFrame<NdArrayBackend<F, I>>>> {
        // `ω` and `Δr` are shared by both blocks; either state slot exposes them.
        let shared = u_state.as_ref().or(v_state.as_ref());

        // `∇ₕ vjp`, with `h = a₁ - Δrᵀ a₀` pushed through `J⁻¹(ω)` into an ω-space velocity.
        let log_hvp = shared.and_then(|(omega, dr)| {
            let h = sum_opt_tensors([
                du.as_ref().map(|du| {
                    -dr.clone()
                        .transpose()
                        .matmul(du.clone().unsqueeze_dim(2))
                        .squeeze_dim(2)
                }),
                dv.clone(),
            ])?;
            let eta = ndarray::log_jvp(omega.clone(), h.unsqueeze_dim(2)).squeeze_dim(2);
            Some(ndarray::log_hvp(omega.clone(), eta, alpha.clone()))
        });

        vec![
            u_state.as_ref().map(|(omega, dr)| {
                let vjp = ndarray::log_vjp(omega.clone(), alpha.clone());
                sum_opt_tensors([
                    du.as_ref()
                        .map(|du| ndarray_cross(du.clone(), dr.clone().matmul(vjp.clone()))),
                    dv.as_ref()
                        .map(|dv| -dr.clone().matmul(ndarray_cross(dv.clone(), vjp.clone()))),
                    log_hvp.clone().map(|log_hvp| -dr.clone().matmul(log_hvp)),
                ])
                .unwrap_or_else(|| alpha.zeros_like())
            }),
            v_state
                .as_ref()
                .map(|_| log_hvp.clone().unwrap_or_else(|| alpha.zeros_like())),
        ]
    }
}
