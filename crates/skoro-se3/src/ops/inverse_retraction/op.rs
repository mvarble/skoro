use skoro_backend::{
    CotangentFrame, CubeBackend, CubeFloat, CubeInt, CubeRuntime, Flex, Point, TangentVector,
    Tensor,
};
use skoro_linalg::vector::cross_product::cube::cross_product as cube_cross;
use skoro_linalg::vector::cross_product::cube::left_broadcast_cross_product as cube_lbcross;
use skoro_linalg::vector::cross_product::flex::cross_product as host_cross;
use skoro_linalg::vector::cross_product::flex::left_broadcast_cross_product as host_lbcross;
use skoro_operation::{
    Input,
    diff::{DiffModeState, DiffOperation, DiffOperationForward},
    sum_opt_tensors,
};
use skoro_so3::ops::log::{cube, flex};

use super::SE3InverseRetraction;

// The inverse of the `SE(3)` retraction `R((r, p), (ω, v)) = (r Exp(ω), p + r v)`:
// ```
// Rinv((r₀, p₀), (r₁, p₁)) = (ω, q),   Δr = r₀ᵀ r₁,   ω = Log(Δr),   q = r₀ᵀ (p₁ - p₀)
// ```
// Both inputs carry the body-frame chart `(a, b) ↦ (r Exp(a), p + r b)`; the output is Euclidean,
// so its "tangent" is an ordinary difference.
//
// **JVP.** With input tangents `(a₀, b₀)` and `(a₁, b₁)`,
// ```
// dω = J⁻¹(ω) (a₁ - Δrᵀ a₀)
// dq = q × a₀ - b₀ + Δr b₁
// ```
//
// **VJP.** Against a cotangent `α = (α_ω, α_q)`,
// ```
// ∂(r₀, p₀) = (-Δr J⁻ᵀ(ω) α_ω - q × α_q,  -α_q)
// ∂(r₁, p₁) = (J⁻ᵀ(ω) α_ω,                 Δrᵀ α_q)
// ```
//
// **HVP.** Writing `h = a₁ - Δrᵀ a₀` for the group-side motion of `ω` and `vjp = J⁻ᵀ(ω) α_ω`,
// and using `dΔr/dt = -[a₀]× Δr + Δr [a₁]×` and `dq/dt = q × a₀ - b₀ + Δr b₁`,
// ```
// ∇∂(r₀, p₀) = (a₀ × (Δr vjp) - Δr (a₁ × vjp) - Δr ∇ₕ vjp
//                - (q × a₀) × α_q + b₀ × α_q - (Δr b₁) × α_q,   0)
// ∇∂(r₁, p₁) = (∇ₕ vjp,                                         -h × (Δrᵀ α_q))
// ```
// Note the `Δrᵀ` in the last term: it is the transpose of the `Δr` appearing in `dΔr/dt`, and the
// rotation-equivariance `(Δrᵀu) × (Δrᵀw) = Δrᵀ(u × w)` is what folds the two contributions
// `Δrᵀ(a₀ × α_q) - a₁ × (Δrᵀ α_q)` into the single cross product above.
impl<R: CubeRuntime, F: CubeFloat, I: CubeInt> DiffOperation<CubeBackend<R, F, I>>
    for SE3InverseRetraction
{
    /// `(ω, q, Δr)` when `(r₀, p₀)` is differentiable, `(ω, Δr)` when `(r₁, p₁)` is.
    type VjpState = (
        Option<(
            Point<CubeBackend<R, F, I>>,
            Point<CubeBackend<R, F, I>>,
            Tensor<CubeBackend<R, F, I>, 3>,
        )>,
        Option<(Point<CubeBackend<R, F, I>>, Tensor<CubeBackend<R, F, I>, 3>)>,
    );
    type HvpState = [Option<TangentVector<CubeBackend<R, F, I>>>; 2];

    fn forward(
        &self,
        inputs: Vec<Input<CubeBackend<R, F, I>>>,
        diff_state: DiffModeState<CubeBackend<R, F, I>>,
    ) -> DiffOperationForward<CubeBackend<R, F, I>, Self::VjpState, Self::HvpState> {
        let inputs: Vec<Point<CubeBackend<R, F, I>>> =
            inputs.into_iter().map(Input::unwrap_float).collect();
        let [rp0, rp1] = inputs.try_into().unwrap();
        let b = rp0.dims()[0];
        let r0 = rp0.clone().slice_dim(1, 0..9).reshape([b, 3, 3]);
        let p0 = rp0.slice_dim(1, 9..12);
        let r1 = rp1.clone().slice_dim(1, 0..9).reshape([b, 3, 3]);
        let p1 = rp1.slice_dim(1, 9..12);
        let r0inv_dp = r0
            .clone()
            .transpose()
            .matmul((p1.clone() - p0.clone()).unsqueeze_dim(2))
            .squeeze_dim(2);
        let dr = r0.clone().transpose().matmul(r1.clone());
        let omega = cube::log(dr.clone().reshape([b, 9]));
        let output = Point::cat(vec![omega.clone(), r0inv_dp.clone()], 1);
        match diff_state {
            DiffModeState::ForwardOnly => DiffOperationForward::forward_only(output),
            DiffModeState::Jvp(tvecs) => {
                let [drp0, drp1] = tvecs.try_into().unwrap();
                DiffOperationForward::jvp(
                    output,
                    sum_opt_tensors([
                        drp0.map(|drp0| {
                            let omega0 = drp0.clone().slice_dim(1, 0..3);
                            let v0 = drp0.clone().slice_dim(1, 3..6);
                            let drot = -cube::log_jvp(
                                omega.clone(),
                                dr.clone().transpose().matmul(omega0.clone()),
                            );
                            let dpsn = cube_lbcross(r0inv_dp.clone(), omega0) - v0;
                            Tensor::cat(vec![drot, dpsn], 1)
                        }),
                        drp1.map(|drp1| {
                            let omega1 = drp1.clone().slice_dim(1, 0..3);
                            let v1 = drp1.clone().slice_dim(1, 3..6);
                            let drot = cube::log_jvp(omega.clone(), omega1);
                            let dpsn = dr.clone().matmul(v1);
                            Tensor::cat(vec![drot, dpsn], 1)
                        }),
                    ]),
                )
            }
            DiffModeState::Vjp(diffs) => {
                let [diff0, diff1] = diffs.try_into().unwrap();
                DiffOperationForward::vjp(
                    output,
                    (
                        diff0.then(|| (omega.clone(), r0inv_dp, dr.clone())),
                        diff1.then_some((omega, dr)),
                    ),
                )
            }
            DiffModeState::Hvp {
                diffs,
                tangents: tvecs,
            } => {
                let [drp0, drp1] = tvecs.try_into().unwrap();
                let [diff0, diff1] = diffs.try_into().unwrap();
                DiffOperationForward::hvp(
                    output,
                    sum_opt_tensors([
                        drp0.as_ref().map(|drp0| {
                            let omega0 = drp0.clone().slice_dim(1, 0..3);
                            let v0 = drp0.clone().slice_dim(1, 3..6);
                            let drot = -cube::log_jvp(
                                omega.clone(),
                                dr.clone()
                                    .transpose()
                                    .matmul(omega0.clone().unsqueeze_dim(2)),
                            )
                            .squeeze_dim(2);
                            let dpsn = cube_cross(r0inv_dp.clone(), omega0) - v0;
                            Tensor::cat(vec![drot, dpsn], 1)
                        }),
                        drp1.as_ref().map(|drp1| {
                            let omega1 = drp1.clone().slice_dim(1, 0..3);
                            let v1 = drp1.clone().slice_dim(1, 3..6);
                            let drot = cube::log_jvp(omega.clone(), omega1.unsqueeze_dim(2))
                                .squeeze_dim(2);
                            let dpsn = dr.clone().matmul(v1.unsqueeze_dim(2)).squeeze_dim(2);
                            Tensor::cat(vec![drot, dpsn], 1)
                        }),
                    ]),
                    // `diffs` (not the tangents) says whether an input traces back to a free
                    // variable, so it is what gates the state `vjp` short-circuits on. A
                    // differentiable input may still arrive with no seeded tangent.
                    (
                        diff0.then(|| (omega.clone(), r0inv_dp, dr.clone())),
                        diff1.then_some((omega, dr)),
                    ),
                    [drp0, drp1],
                )
            }
        }
    }

    fn vjp(
        &self,
        (rp0_state, rp1_state): &Self::VjpState,
        alpha: CotangentFrame<CubeBackend<R, F, I>>,
    ) -> Vec<Option<CotangentFrame<CubeBackend<R, F, I>>>> {
        let alphar = alpha.clone().slice_dim(1, 0..3);
        let alphap = alpha.slice_dim(1, 3..6);
        vec![
            rp0_state.as_ref().map(|(omega, r0inv_dp, dr)| {
                let drot0 = -dr
                    .clone()
                    .matmul(cube::log_vjp(omega.clone(), alphar.clone()))
                    - cube_lbcross(r0inv_dp.clone(), alphap.clone());
                Tensor::cat(vec![drot0, -alphap.clone()], 1)
            }),
            rp1_state.as_ref().map(|(omega, dr)| {
                let drot1 = cube::log_vjp(omega.clone(), alphar.clone());
                let dpsn1 = dr.clone().transpose().matmul(alphap.clone());
                Tensor::cat(vec![drot1, dpsn1], 1)
            }),
        ]
    }

    fn hvp(
        &self,
        (rp0_state, rp1_state): &Self::VjpState,
        [drp0, drp1]: &Self::HvpState,
        alpha: CotangentFrame<CubeBackend<R, F, I>>,
    ) -> Vec<Option<CotangentFrame<CubeBackend<R, F, I>>>> {
        let alphar = alpha.clone().slice_dim(1, 0..3);
        let alphap = alpha.slice_dim(1, 3..6);

        let omega0 = drp0.as_ref().map(|drp0| drp0.clone().slice_dim(1, 0..3));
        let v0 = drp0.as_ref().map(|drp0| drp0.clone().slice_dim(1, 3..6));
        let omega1 = drp1.as_ref().map(|drp1| drp1.clone().slice_dim(1, 0..3));
        let v1 = drp1.as_ref().map(|drp1| drp1.clone().slice_dim(1, 3..6));

        // `Δr` and `ω` are shared by both blocks; either state slot exposes them.
        let dr = rp0_state
            .as_ref()
            .map(|(_, _, dr)| dr.clone())
            .or_else(|| rp1_state.as_ref().map(|(_, dr)| dr.clone()));
        let omega = rp0_state
            .as_ref()
            .map(|(omega, _, _)| omega.clone())
            .or_else(|| rp1_state.as_ref().map(|(omega, _)| omega.clone()));

        // `h = a₁ - Δrᵀ a₀`, the group-side direction along which `ω` moves.
        let h = dr.as_ref().and_then(|dr| {
            sum_opt_tensors([
                omega0.clone().map(|omega0| {
                    -dr.clone()
                        .transpose()
                        .matmul(omega0.unsqueeze_dim(2))
                        .squeeze_dim(2)
                }),
                omega1.clone(),
            ])
        });
        // `log_hvp` differentiates along an ω-space velocity, not a group-side one, so `h` has to
        // be pushed through `J⁻¹(ω)` first (cf. `SO3Log::forward`, which feeds it `log_jvp(ω, ·)`).
        let log_hvp = omega.as_ref().zip(h.as_ref()).map(|(omega, h)| {
            let eta = cube::log_jvp(omega.clone(), h.clone().unsqueeze_dim(2)).squeeze_dim(2);
            cube::log_hvp(omega.clone(), eta, alphar.clone())
        });

        vec![
            rp0_state.as_ref().map(|(omega, r0inv_dp, dr)| {
                let vjp = cube::log_vjp(omega.clone(), alphar.clone());
                let rot = sum_opt_tensors([
                    omega0
                        .clone()
                        .map(|omega0| cube_lbcross(omega0, dr.clone().matmul(vjp.clone()))),
                    omega1
                        .clone()
                        .map(|omega1| -dr.clone().matmul(cube_lbcross(omega1, vjp.clone()))),
                    log_hvp.clone().map(|log_hvp| -dr.clone().matmul(log_hvp)),
                    omega0.clone().map(|omega0| {
                        -cube_lbcross(cube_cross(r0inv_dp.clone(), omega0), alphap.clone())
                    }),
                    v0.clone().map(|v0| cube_lbcross(v0, alphap.clone())),
                    v1.clone().map(|v1| {
                        -cube_lbcross(
                            dr.clone().matmul(v1.unsqueeze_dim(2)).squeeze_dim(2),
                            alphap.clone(),
                        )
                    }),
                ])
                .unwrap_or_else(|| alphar.zeros_like());
                Tensor::cat(vec![rot, alphap.zeros_like()], 1)
            }),
            rp1_state.as_ref().map(|(_, dr)| {
                let rot = log_hvp.clone().unwrap_or_else(|| alphar.zeros_like());
                let trans = h
                    .as_ref()
                    .map(|h| {
                        -cube_lbcross(h.clone(), dr.clone().transpose().matmul(alphap.clone()))
                    })
                    .unwrap_or_else(|| alphap.zeros_like());
                Tensor::cat(vec![rot, trans], 1)
            }),
        ]
    }
}

impl DiffOperation<Flex> for SE3InverseRetraction {
    /// `(ω, q, Δr)` when `(r₀, p₀)` is differentiable, `(ω, Δr)` when `(r₁, p₁)` is.
    type VjpState = (
        Option<(Point<Flex>, Point<Flex>, Tensor<Flex, 3>)>,
        Option<(Point<Flex>, Tensor<Flex, 3>)>,
    );
    type HvpState = [Option<TangentVector<Flex>>; 2];

    fn forward(
        &self,
        inputs: Vec<Input<Flex>>,
        diff_state: DiffModeState<Flex>,
    ) -> DiffOperationForward<Flex, Self::VjpState, Self::HvpState> {
        let inputs: Vec<Point<Flex>> = inputs.into_iter().map(Input::unwrap_float).collect();
        let [rp0, rp1] = inputs.try_into().unwrap();
        let b = rp0.dims()[0];
        let r0 = rp0.clone().slice_dim(1, 0..9).reshape([b, 3, 3]);
        let p0 = rp0.slice_dim(1, 9..12);
        let r1 = rp1.clone().slice_dim(1, 0..9).reshape([b, 3, 3]);
        let p1 = rp1.slice_dim(1, 9..12);
        let r0inv_dp = r0
            .clone()
            .transpose()
            .matmul((p1.clone() - p0.clone()).unsqueeze_dim(2))
            .squeeze_dim(2);
        let dr = r0.clone().transpose().matmul(r1.clone());
        let omega = flex::log(dr.clone().reshape([b, 9]));
        let output = Point::cat(vec![omega.clone(), r0inv_dp.clone()], 1);
        match diff_state {
            DiffModeState::ForwardOnly => DiffOperationForward::forward_only(output),
            DiffModeState::Jvp(tvecs) => {
                let [drp0, drp1] = tvecs.try_into().unwrap();
                DiffOperationForward::jvp(
                    output,
                    sum_opt_tensors([
                        drp0.map(|drp0| {
                            let omega0 = drp0.clone().slice_dim(1, 0..3);
                            let v0 = drp0.clone().slice_dim(1, 3..6);
                            let drot = -flex::log_jvp(
                                omega.clone(),
                                dr.clone().transpose().matmul(omega0.clone()),
                            );
                            let dpsn = host_lbcross(r0inv_dp.clone(), omega0) - v0;
                            Tensor::cat(vec![drot, dpsn], 1)
                        }),
                        drp1.map(|drp1| {
                            let omega1 = drp1.clone().slice_dim(1, 0..3);
                            let v1 = drp1.clone().slice_dim(1, 3..6);
                            let drot = flex::log_jvp(omega.clone(), omega1);
                            let dpsn = dr.clone().matmul(v1);
                            Tensor::cat(vec![drot, dpsn], 1)
                        }),
                    ]),
                )
            }
            DiffModeState::Vjp(diffs) => {
                let [diff0, diff1] = diffs.try_into().unwrap();
                DiffOperationForward::vjp(
                    output,
                    (
                        diff0.then(|| (omega.clone(), r0inv_dp, dr.clone())),
                        diff1.then_some((omega, dr)),
                    ),
                )
            }
            DiffModeState::Hvp {
                diffs,
                tangents: tvecs,
            } => {
                let [drp0, drp1] = tvecs.try_into().unwrap();
                let [diff0, diff1] = diffs.try_into().unwrap();
                DiffOperationForward::hvp(
                    output,
                    sum_opt_tensors([
                        drp0.as_ref().map(|drp0| {
                            let omega0 = drp0.clone().slice_dim(1, 0..3);
                            let v0 = drp0.clone().slice_dim(1, 3..6);
                            let drot = -flex::log_jvp(
                                omega.clone(),
                                dr.clone()
                                    .transpose()
                                    .matmul(omega0.clone().unsqueeze_dim(2)),
                            )
                            .squeeze_dim(2);
                            let dpsn = host_cross(r0inv_dp.clone(), omega0) - v0;
                            Tensor::cat(vec![drot, dpsn], 1)
                        }),
                        drp1.as_ref().map(|drp1| {
                            let omega1 = drp1.clone().slice_dim(1, 0..3);
                            let v1 = drp1.clone().slice_dim(1, 3..6);
                            let drot = flex::log_jvp(omega.clone(), omega1.unsqueeze_dim(2))
                                .squeeze_dim(2);
                            let dpsn = dr.clone().matmul(v1.unsqueeze_dim(2)).squeeze_dim(2);
                            Tensor::cat(vec![drot, dpsn], 1)
                        }),
                    ]),
                    // `diffs` (not the tangents) says whether an input traces back to a free
                    // variable, so it is what gates the state `vjp` short-circuits on. A
                    // differentiable input may still arrive with no seeded tangent.
                    (
                        diff0.then(|| (omega.clone(), r0inv_dp, dr.clone())),
                        diff1.then_some((omega, dr)),
                    ),
                    [drp0, drp1],
                )
            }
        }
    }

    fn vjp(
        &self,
        (rp0_state, rp1_state): &Self::VjpState,
        alpha: CotangentFrame<Flex>,
    ) -> Vec<Option<CotangentFrame<Flex>>> {
        let alphar = alpha.clone().slice_dim(1, 0..3);
        let alphap = alpha.slice_dim(1, 3..6);
        vec![
            rp0_state.as_ref().map(|(omega, r0inv_dp, dr)| {
                let drot0 = -dr
                    .clone()
                    .matmul(flex::log_vjp(omega.clone(), alphar.clone()))
                    - host_lbcross(r0inv_dp.clone(), alphap.clone());
                Tensor::cat(vec![drot0, -alphap.clone()], 1)
            }),
            rp1_state.as_ref().map(|(omega, dr)| {
                let drot1 = flex::log_vjp(omega.clone(), alphar.clone());
                let dpsn1 = dr.clone().transpose().matmul(alphap.clone());
                Tensor::cat(vec![drot1, dpsn1], 1)
            }),
        ]
    }

    fn hvp(
        &self,
        (rp0_state, rp1_state): &Self::VjpState,
        [drp0, drp1]: &Self::HvpState,
        alpha: CotangentFrame<Flex>,
    ) -> Vec<Option<CotangentFrame<Flex>>> {
        let alphar = alpha.clone().slice_dim(1, 0..3);
        let alphap = alpha.slice_dim(1, 3..6);

        let omega0 = drp0.as_ref().map(|drp0| drp0.clone().slice_dim(1, 0..3));
        let v0 = drp0.as_ref().map(|drp0| drp0.clone().slice_dim(1, 3..6));
        let omega1 = drp1.as_ref().map(|drp1| drp1.clone().slice_dim(1, 0..3));
        let v1 = drp1.as_ref().map(|drp1| drp1.clone().slice_dim(1, 3..6));

        // `Δr` and `ω` are shared by both blocks; either state slot exposes them.
        let dr = rp0_state
            .as_ref()
            .map(|(_, _, dr)| dr.clone())
            .or_else(|| rp1_state.as_ref().map(|(_, dr)| dr.clone()));
        let omega = rp0_state
            .as_ref()
            .map(|(omega, _, _)| omega.clone())
            .or_else(|| rp1_state.as_ref().map(|(omega, _)| omega.clone()));

        // `h = a₁ - Δrᵀ a₀`, the group-side direction along which `ω` moves.
        let h = dr.as_ref().and_then(|dr| {
            sum_opt_tensors([
                omega0.clone().map(|omega0| {
                    -dr.clone()
                        .transpose()
                        .matmul(omega0.unsqueeze_dim(2))
                        .squeeze_dim(2)
                }),
                omega1.clone(),
            ])
        });
        // `log_hvp` differentiates along an ω-space velocity, not a group-side one, so `h` has to
        // be pushed through `J⁻¹(ω)` first (cf. `SO3Log::forward`, which feeds it `log_jvp(ω, ·)`).
        let log_hvp = omega.as_ref().zip(h.as_ref()).map(|(omega, h)| {
            let eta = flex::log_jvp(omega.clone(), h.clone().unsqueeze_dim(2)).squeeze_dim(2);
            flex::log_hvp(omega.clone(), eta, alphar.clone())
        });

        vec![
            rp0_state.as_ref().map(|(omega, r0inv_dp, dr)| {
                let vjp = flex::log_vjp(omega.clone(), alphar.clone());
                let rot = sum_opt_tensors([
                    omega0
                        .clone()
                        .map(|omega0| host_lbcross(omega0, dr.clone().matmul(vjp.clone()))),
                    omega1
                        .clone()
                        .map(|omega1| -dr.clone().matmul(host_lbcross(omega1, vjp.clone()))),
                    log_hvp.clone().map(|log_hvp| -dr.clone().matmul(log_hvp)),
                    omega0.clone().map(|omega0| {
                        -host_lbcross(host_cross(r0inv_dp.clone(), omega0), alphap.clone())
                    }),
                    v0.clone().map(|v0| host_lbcross(v0, alphap.clone())),
                    v1.clone().map(|v1| {
                        -host_lbcross(
                            dr.clone().matmul(v1.unsqueeze_dim(2)).squeeze_dim(2),
                            alphap.clone(),
                        )
                    }),
                ])
                .unwrap_or_else(|| alphar.zeros_like());
                Tensor::cat(vec![rot, alphap.zeros_like()], 1)
            }),
            rp1_state.as_ref().map(|(_, dr)| {
                let rot = log_hvp.clone().unwrap_or_else(|| alphar.zeros_like());
                let trans = h
                    .as_ref()
                    .map(|h| {
                        -host_lbcross(h.clone(), dr.clone().transpose().matmul(alphap.clone()))
                    })
                    .unwrap_or_else(|| alphap.zeros_like());
                Tensor::cat(vec![rot, trans], 1)
            }),
        ]
    }
}
