use num_traits::Float;

use skoro_backend::{
    CotangentFrame, CubeBackend, CubeFloat, CubeInt, CubeRuntime, NdArrayBackend, NdArrayFloat,
    NdArrayInt, NdArrayTensor, Point, SharedArray, TangentVector, Tensor,
};
use skoro_linalg::{
    vector::cross_product::cube::cross_product as cube_cross,
    vector::cross_product::cube::left_broadcast_cross_product as cube_lbcross,
    vector::cross_product::ndarray::cross_product as ndarray_cross,
    vector::cross_product::ndarray::left_broadcast_cross_product as ndarray_lbcross,
};
use skoro_operation::{
    Input,
    diff::{DiffModeState, DiffOperation, DiffOperationForward},
    sum_opt_tensors,
};
use skoro_so3::ops::exp::{cube, ndarray};

use super::SE3Retraction;

// The retraction of `SE(3)` used here is
// ```
// R((r, p), (ω, v)) = (r Exp(ω), p + r v)
// ```
// which induces the body-frame chart `(a, b) ↦ (r Exp(a), p + r b)` on the tangent space at
// `(r, p)`. Writing `E = Exp(ω)`, the output point is `y = (r E, p + r v)`, whose own chart is
// `(c, d) ↦ (r E Exp(c), p + r v + r E d)`.
//
// **JVP.** Pushing the input tangent `(a, b)` and the parameter tangent `(dω, dv)` through gives
// ```
// c = Eᵀ a + J(ω) dω
// d = Eᵀ (b + a × v + dv)
// ```
// The `a × v` coupling is what makes the translation block feel the rotational perturbation: moving
// `r` along `a` swings the lever arm `r v`. Both translation blocks carry `Eᵀ`, not `E`, because
// `d` is read off in the *output* frame `r E`.
//
// **VJP.** Transposing the above against a cotangent `α = (α_r, α_p)` yields
// ```
// ∂(r, p) = (E α_r + v × (E α_p),  E α_p)
// ∂(ω, v) = (J(ω)ᵀ α_r,            E α_p)
// ```
//
// **HVP.** Neither VJP block depends on `r` or `p` — only on `ω` and `v` — so the whole
// sensitivity is driven by the parameter tangent `(dω, dv)`. With `g = J(ω) dω` (so that
// `d/dt Exp(ω + t dω) = E [g]×`),
// ```
// ∇∂(r, p) = (E (g × α_r) + dv × (E α_p) + v × (E (g × α_p)),  E (g × α_p))
// ∇∂(ω, v) = (∇_dω J(ω)ᵀ α_r,                                  E (g × α_p))
// ```
impl<R: CubeRuntime, F: CubeFloat, I: CubeInt> DiffOperation<CubeBackend<R, F, I>>
    for SE3Retraction
{
    /// `(E, v)` when `(r, p)` is differentiable, and `(E, ω)` when `(ω, v)` is differentiable.
    type VjpState = (
        Option<(Tensor<CubeBackend<R, F, I>, 3>, Point<CubeBackend<R, F, I>>)>,
        Option<(Tensor<CubeBackend<R, F, I>, 3>, Point<CubeBackend<R, F, I>>)>,
    );
    /// `g = J(ω) dω`, then `dω`, then `dv`.
    type HvpState = [Option<TangentVector<CubeBackend<R, F, I>>>; 3];

    fn forward(
        &self,
        inputs: Vec<Input<CubeBackend<R, F, I>>>,
        diff_state: DiffModeState<CubeBackend<R, F, I>>,
    ) -> DiffOperationForward<CubeBackend<R, F, I>, Self::VjpState, Self::HvpState> {
        let inputs: Vec<Point<CubeBackend<R, F, I>>> =
            inputs.into_iter().map(Input::unwrap_float).collect();
        let [rp, omegav] = inputs.try_into().unwrap();
        let b = rp.dims()[0];
        let r = rp.clone().slice_dim(1, 0..9);
        let p = rp.slice_dim(1, 9..12);
        let omega = omegav.clone().slice_dim(1, 0..3);
        let v = omegav.slice_dim(1, 3..6);
        let rmat = r.reshape([b, 3, 3]);
        let p_plus_rv = p + rmat
            .clone()
            .matmul(v.clone().unsqueeze_dim(2))
            .squeeze_dim(2);
        let expomega = cube::exp(omega.clone()).reshape([b, 3, 3]);
        let rexpomega = rmat.matmul(expomega.clone()).reshape([b, 9]);
        let output = Point::cat(vec![rexpomega, p_plus_rv], 1);
        match diff_state {
            DiffModeState::ForwardOnly => DiffOperationForward::forward_only(output),
            DiffModeState::Jvp(tvecs) => {
                let [drp, domegav] = tvecs.try_into().unwrap();
                DiffOperationForward::jvp(
                    output,
                    sum_opt_tensors([
                        drp.map(|drp| {
                            let a = drp.clone().slice_dim(1, 0..3);
                            let b = drp.slice_dim(1, 3..6);
                            Tensor::cat(
                                vec![
                                    expomega.clone().transpose().matmul(a.clone()),
                                    expomega
                                        .clone()
                                        .transpose()
                                        .matmul(b - cube_lbcross(v.clone(), a)),
                                ],
                                1,
                            )
                        }),
                        domegav.map(|domegav| {
                            let domega = domegav.clone().slice_dim(1, 0..3);
                            let dv = domegav.slice_dim(1, 3..6);
                            Tensor::cat(
                                vec![
                                    cube::exp_jvp(omega.clone(), domega),
                                    expomega.clone().transpose().matmul(dv),
                                ],
                                1,
                            )
                        }),
                    ]),
                )
            }
            DiffModeState::Vjp(diffs) => {
                let [rp_diff, omegav_diff] = diffs.try_into().unwrap();
                DiffOperationForward::vjp(
                    output,
                    (
                        rp_diff.then(|| (expomega.clone(), v)),
                        omegav_diff.then_some((expomega, omega)),
                    ),
                )
            }
            DiffModeState::Hvp {
                diffs,
                tangents: tvecs,
            } => {
                let [drp, domegav] = tvecs.try_into().unwrap();
                let [rp_diff, omegav_diff] = diffs.try_into().unwrap();
                let domega = domegav
                    .as_ref()
                    .map(|domegav| domegav.clone().slice_dim(1, 0..3));
                let dv = domegav.map(|domegav| domegav.slice_dim(1, 3..6));
                let g = domega.as_ref().map(|domega| {
                    cube::exp_jvp(omega.clone(), domega.clone().unsqueeze_dim(2)).squeeze_dim(2)
                });
                DiffOperationForward::hvp(
                    output,
                    sum_opt_tensors([
                        drp.map(|drp| {
                            let a = drp.clone().slice_dim(1, 0..3);
                            let b = drp.slice_dim(1, 3..6);
                            Tensor::cat(
                                vec![
                                    expomega
                                        .clone()
                                        .transpose()
                                        .matmul(a.clone().unsqueeze_dim(2))
                                        .squeeze_dim(2),
                                    expomega
                                        .clone()
                                        .transpose()
                                        .matmul((b - cube_cross(v.clone(), a)).unsqueeze_dim(2))
                                        .squeeze_dim(2),
                                ],
                                1,
                            )
                        }),
                        g.clone().zip(dv.clone()).map(|(g, dv)| {
                            Tensor::cat(
                                vec![
                                    g,
                                    expomega
                                        .clone()
                                        .transpose()
                                        .matmul(dv.unsqueeze_dim(2))
                                        .squeeze_dim(2),
                                ],
                                1,
                            )
                        }),
                    ]),
                    // `diffs` (not the tangents) says whether an input traces back to a free
                    // variable, so it is what gates the state `vjp` short-circuits on. A
                    // differentiable input may still arrive with no seeded tangent.
                    (
                        rp_diff.then(|| (expomega.clone(), v)),
                        omegav_diff.then_some((expomega, omega)),
                    ),
                    [g, domega, dv],
                )
            }
        }
    }

    fn vjp(
        &self,
        (rp_state, omegav_state): &Self::VjpState,
        alpharp: CotangentFrame<CubeBackend<R, F, I>>,
    ) -> Vec<Option<CotangentFrame<CubeBackend<R, F, I>>>> {
        let alphar = alpharp.clone().slice_dim(1, 0..3);
        let alphap = alpharp.slice_dim(1, 3..6);
        vec![
            rp_state.as_ref().map(|(expomega, v)| {
                let e_alphap = expomega.clone().matmul(alphap.clone());
                let rot = expomega.clone().matmul(alphar.clone())
                    + cube_lbcross(v.clone(), e_alphap.clone());
                CotangentFrame::cat(vec![rot, e_alphap], 1)
            }),
            omegav_state.as_ref().map(|(expomega, omega)| {
                CotangentFrame::cat(
                    vec![
                        cube::exp_vjp(omega.clone(), alphar.clone()),
                        expomega.clone().matmul(alphap.clone()),
                    ],
                    1,
                )
            }),
        ]
    }

    fn hvp(
        &self,
        (rp_state, omegav_state): &Self::VjpState,
        [g, domega, dv]: &Self::HvpState,
        alpharp: CotangentFrame<CubeBackend<R, F, I>>,
    ) -> Vec<Option<CotangentFrame<CubeBackend<R, F, I>>>> {
        let alphar = alpharp.clone().slice_dim(1, 0..3);
        let alphap = alpharp.slice_dim(1, 3..6);
        vec![
            rp_state.as_ref().map(|(expomega, v)| {
                // `E (g × α_p)`, shared by the translation block and the lever-arm term.
                let eg_alphap = g.as_ref().map(|g| {
                    expomega
                        .clone()
                        .matmul(cube_lbcross(g.clone(), alphap.clone()))
                });
                let rot = sum_opt_tensors([
                    g.as_ref().map(|g| {
                        expomega
                            .clone()
                            .matmul(cube_lbcross(g.clone(), alphar.clone()))
                    }),
                    dv.as_ref().map(|dv| {
                        cube_lbcross(dv.clone(), expomega.clone().matmul(alphap.clone()))
                    }),
                    eg_alphap
                        .clone()
                        .map(|eg_alphap| cube_lbcross(v.clone(), eg_alphap)),
                ])
                .unwrap_or_else(|| alphar.zeros_like());
                let trans = eg_alphap.unwrap_or_else(|| alphap.zeros_like());
                CotangentFrame::cat(vec![rot, trans], 1)
            }),
            omegav_state.as_ref().map(|(expomega, omega)| {
                let rot = domega
                    .as_ref()
                    .map(|domega| cube::exp_hvp(omega.clone(), domega.clone(), alphar.clone()))
                    .unwrap_or_else(|| alphar.zeros_like());
                let trans = g
                    .as_ref()
                    .map(|g| {
                        expomega
                            .clone()
                            .matmul(cube_lbcross(g.clone(), alphap.clone()))
                    })
                    .unwrap_or_else(|| alphap.zeros_like());
                CotangentFrame::cat(vec![rot, trans], 1)
            }),
        ]
    }
}

impl<F, I> DiffOperation<NdArrayBackend<F, I>> for SE3Retraction
where
    F: NdArrayFloat + Float,
    I: NdArrayInt,
    NdArrayTensor: From<SharedArray<F>> + From<SharedArray<I>>,
{
    /// `(E, v)` when `(r, p)` is differentiable, and `(E, ω)` when `(ω, v)` is differentiable.
    type VjpState = (
        Option<(Tensor<NdArrayBackend<F, I>, 3>, Point<NdArrayBackend<F, I>>)>,
        Option<(Tensor<NdArrayBackend<F, I>, 3>, Point<NdArrayBackend<F, I>>)>,
    );
    /// `g = J(ω) dω`, then `dω`, then `dv`.
    type HvpState = [Option<TangentVector<NdArrayBackend<F, I>>>; 3];

    fn forward(
        &self,
        inputs: Vec<Input<NdArrayBackend<F, I>>>,
        diff_state: DiffModeState<NdArrayBackend<F, I>>,
    ) -> DiffOperationForward<NdArrayBackend<F, I>, Self::VjpState, Self::HvpState> {
        let inputs: Vec<Point<NdArrayBackend<F, I>>> =
            inputs.into_iter().map(Input::unwrap_float).collect();
        let [rp, omegav] = inputs.try_into().unwrap();
        let b = rp.dims()[0];
        let r = rp.clone().slice_dim(1, 0..9);
        let p = rp.slice_dim(1, 9..12);
        let omega = omegav.clone().slice_dim(1, 0..3);
        let v = omegav.slice_dim(1, 3..6);
        let rmat = r.reshape([b, 3, 3]);
        let p_plus_rv = p + rmat
            .clone()
            .matmul(v.clone().unsqueeze_dim(2))
            .squeeze_dim(2);
        let expomega = ndarray::exp(omega.clone()).reshape([b, 3, 3]);
        let rexpomega = rmat.matmul(expomega.clone()).reshape([b, 9]);
        let output = Point::cat(vec![rexpomega, p_plus_rv], 1);
        match diff_state {
            DiffModeState::ForwardOnly => DiffOperationForward::forward_only(output),
            DiffModeState::Jvp(tvecs) => {
                let [drp, domegav] = tvecs.try_into().unwrap();
                DiffOperationForward::jvp(
                    output,
                    sum_opt_tensors([
                        drp.map(|drp| {
                            let a = drp.clone().slice_dim(1, 0..3);
                            let b = drp.slice_dim(1, 3..6);
                            Tensor::cat(
                                vec![
                                    expomega.clone().transpose().matmul(a.clone()),
                                    expomega
                                        .clone()
                                        .transpose()
                                        .matmul(b - ndarray_lbcross(v.clone(), a)),
                                ],
                                1,
                            )
                        }),
                        domegav.map(|domegav| {
                            let domega = domegav.clone().slice_dim(1, 0..3);
                            let dv = domegav.slice_dim(1, 3..6);
                            Tensor::cat(
                                vec![
                                    ndarray::exp_jvp(omega.clone(), domega),
                                    expomega.clone().transpose().matmul(dv),
                                ],
                                1,
                            )
                        }),
                    ]),
                )
            }
            DiffModeState::Vjp(diffs) => {
                let [rp_diff, omegav_diff] = diffs.try_into().unwrap();
                DiffOperationForward::vjp(
                    output,
                    (
                        rp_diff.then(|| (expomega.clone(), v)),
                        omegav_diff.then_some((expomega, omega)),
                    ),
                )
            }
            DiffModeState::Hvp {
                diffs,
                tangents: tvecs,
            } => {
                let [drp, domegav] = tvecs.try_into().unwrap();
                let [rp_diff, omegav_diff] = diffs.try_into().unwrap();
                let domega = domegav
                    .as_ref()
                    .map(|domegav| domegav.clone().slice_dim(1, 0..3));
                let dv = domegav.map(|domegav| domegav.slice_dim(1, 3..6));
                let g = domega.as_ref().map(|domega| {
                    ndarray::exp_jvp(omega.clone(), domega.clone().unsqueeze_dim(2)).squeeze_dim(2)
                });
                DiffOperationForward::hvp(
                    output,
                    sum_opt_tensors([
                        drp.map(|drp| {
                            let a = drp.clone().slice_dim(1, 0..3);
                            let b = drp.slice_dim(1, 3..6);
                            Tensor::cat(
                                vec![
                                    expomega
                                        .clone()
                                        .transpose()
                                        .matmul(a.clone().unsqueeze_dim(2))
                                        .squeeze_dim(2),
                                    expomega
                                        .clone()
                                        .transpose()
                                        .matmul((b - ndarray_cross(v.clone(), a)).unsqueeze_dim(2))
                                        .squeeze_dim(2),
                                ],
                                1,
                            )
                        }),
                        g.clone().zip(dv.clone()).map(|(g, dv)| {
                            Tensor::cat(
                                vec![
                                    g,
                                    expomega
                                        .clone()
                                        .transpose()
                                        .matmul(dv.unsqueeze_dim(2))
                                        .squeeze_dim(2),
                                ],
                                1,
                            )
                        }),
                    ]),
                    // `diffs` (not the tangents) says whether an input traces back to a free
                    // variable, so it is what gates the state `vjp` short-circuits on. A
                    // differentiable input may still arrive with no seeded tangent.
                    (
                        rp_diff.then(|| (expomega.clone(), v)),
                        omegav_diff.then_some((expomega, omega)),
                    ),
                    [g, domega, dv],
                )
            }
        }
    }

    fn vjp(
        &self,
        (rp_state, omegav_state): &Self::VjpState,
        alpharp: CotangentFrame<NdArrayBackend<F, I>>,
    ) -> Vec<Option<CotangentFrame<NdArrayBackend<F, I>>>> {
        let alphar = alpharp.clone().slice_dim(1, 0..3);
        let alphap = alpharp.slice_dim(1, 3..6);
        vec![
            rp_state.as_ref().map(|(expomega, v)| {
                let e_alphap = expomega.clone().matmul(alphap.clone());
                let rot = expomega.clone().matmul(alphar.clone())
                    + ndarray_lbcross(v.clone(), e_alphap.clone());
                CotangentFrame::cat(vec![rot, e_alphap], 1)
            }),
            omegav_state.as_ref().map(|(expomega, omega)| {
                CotangentFrame::cat(
                    vec![
                        ndarray::exp_vjp(omega.clone(), alphar.clone()),
                        expomega.clone().matmul(alphap.clone()),
                    ],
                    1,
                )
            }),
        ]
    }

    fn hvp(
        &self,
        (rp_state, omegav_state): &Self::VjpState,
        [g, domega, dv]: &Self::HvpState,
        alpharp: CotangentFrame<NdArrayBackend<F, I>>,
    ) -> Vec<Option<CotangentFrame<NdArrayBackend<F, I>>>> {
        let alphar = alpharp.clone().slice_dim(1, 0..3);
        let alphap = alpharp.slice_dim(1, 3..6);
        vec![
            rp_state.as_ref().map(|(expomega, v)| {
                // `E (g × α_p)`, shared by the translation block and the lever-arm term.
                let eg_alphap = g.as_ref().map(|g| {
                    expomega
                        .clone()
                        .matmul(ndarray_lbcross(g.clone(), alphap.clone()))
                });
                let rot = sum_opt_tensors([
                    g.as_ref().map(|g| {
                        expomega
                            .clone()
                            .matmul(ndarray_lbcross(g.clone(), alphar.clone()))
                    }),
                    dv.as_ref().map(|dv| {
                        ndarray_lbcross(dv.clone(), expomega.clone().matmul(alphap.clone()))
                    }),
                    eg_alphap
                        .clone()
                        .map(|eg_alphap| ndarray_lbcross(v.clone(), eg_alphap)),
                ])
                .unwrap_or_else(|| alphar.zeros_like());
                let trans = eg_alphap.unwrap_or_else(|| alphap.zeros_like());
                CotangentFrame::cat(vec![rot, trans], 1)
            }),
            omegav_state.as_ref().map(|(expomega, omega)| {
                let rot = domega
                    .as_ref()
                    .map(|domega| ndarray::exp_hvp(omega.clone(), domega.clone(), alphar.clone()))
                    .unwrap_or_else(|| alphar.zeros_like());
                let trans = g
                    .as_ref()
                    .map(|g| {
                        expomega
                            .clone()
                            .matmul(ndarray_lbcross(g.clone(), alphap.clone()))
                    })
                    .unwrap_or_else(|| alphap.zeros_like());
                CotangentFrame::cat(vec![rot, trans], 1)
            }),
        ]
    }
}
