//! The Fisher information root of a Gaussian over `[x, μ, L]`.
//!
//! Let `d` be the manifold dimension of `x` and `μ`, `d_L = d(d+1)/2` that of `L`, and
//! `W = L⁻¹`. The root is
//!
//! ```text
//! C = [ W   −W   0   ]   rows 0 .. d
//!     [ 0    0   C_L ]   rows d .. d + d²
//! ```
//!
//! Over `(x, μ)`, the expected information of `−½‖W(x ⊖ μ)‖²` is `[I, −I]ᵀ WᵀW [I, −I]`, which
//! the first block reproduces. Over `L`, with `Σ = LLᵀ`, the information is
//! `F[q][q'] = ½ tr(Σ⁻¹ ∂_qΣ Σ⁻¹ ∂_{q'}Σ)`; since `Σ⁻¹ ∂_qΣ = L⁻ᵀ (G_q + G_qᵀ) Lᵀ` with
//! `G_q = W E_q`, that trace is `tr((G_q + G_qᵀ)(G_{q'} + G_{q'}ᵀ))`. So column `q` of `C_L` is
//! `vec((G_q + G_qᵀ) / √2)`. The cross blocks vanish in expectation, being linear in the whitened
//! residual.
//!
//! Every entry of `C` is linear in `W`, which is what lets the root be one fixed linear map of
//! the packed whitening.
//!
//! # Accuracy on curved manifolds
//!
//! The `[W, −W]` block takes the Jacobians of `v = x ⊖ μ` with respect to `x` and `μ` to be `I` and
//! `−I`. On `EuclideanD` that is exact. On a group the inverse retraction's Jacobians equal `±I`
//! only at `v = 0`. For `SO3`, whose inverse retraction is `ω = Log(r₀ᵀ r₁)`, they are
//! `J_r⁻¹(ω) = I + ½[ω]× + O(‖ω‖²)` and `−J_r⁻¹(ω) Exp(−ω) = −(I − ½[ω]×) + O(‖ω‖²)`; `SE3`'s
//! translation block adds further terms linear in `v`.
//!
//! Under the model `v` is Gaussian with covariance `Σ`, and a Gaussian's odd moments vanish, so every
//! correction linear in `v` drops out of the expectation. The root is therefore exact on flat spaces
//! and correct to second order in the dispersion on `SO3` and `SE3`: the relative error of `CᵀC` grows
//! like `‖Σ‖`. That matches the order of the approximation a `Normal` on a group already makes, since
//! its density omits the retraction's volume factor `det J_r(ω) = 1 − ‖ω‖²/12 + …`. When `Σ` is
//! isotropic the corrections vanish identically, because each one applies `[v]×` to a multiple of `v`.
//!
//! Against the covariance of the model's own score, with `Σ = σ² L₀L₀ᵀ` for a fixed skewed triangle
//! `L₀`, the measured relative error of `CᵀC` is about `1.2 σ²` on `SO3` and `1.6 σ²` on `SE3`.
//!
//! Two alternatives were rejected. The exact expectation has no closed form on a group. A root built
//! from the residual's Jacobian at the current point — the Gauss–Newton rows, which the `RESIDUAL`
//! binding already provides — is exact at that point, but it is not an expected information, and it
//! would make a Riemannian HMC metric depend on derivatives of the logarithm's Jacobians.

use skoro_backend::Backend;
use skoro_core::error::SkoroResult;
use skoro_linalg::{
    matrix::{FixedLinearMap, linear_map},
    tril::Tril,
};
use skoro_model::q::Q;
use skoro_riemann::matrix::EuclideanMatrixD;

/// Index of entry `(i, j)`, `j ≤ i`, in a row-major packed lower triangle.
fn packed(i: usize, j: usize) -> usize {
    i * (i + 1) / 2 + j
}

/// `K` with `vec(C) = K · vec(W)`: `W = L⁻¹` packed as a `Tril(d)`, `C` the Gaussian Fisher root
/// over `[x, μ, L]` in row-major order.
///
/// Returns `(rows, cols, data)` with `rows = (d + d²)(2d + d(d+1)/2)`, `cols = d(d+1)/2`, and
/// `data` row-major.
pub fn gaussian_root_matrix(d: usize) -> (usize, usize, Vec<f64>) {
    let d_l = d * (d + 1) / 2;
    let c_cols = 2 * d + d_l;
    let c_rows = d + d * d;
    let rows = c_rows * c_cols;
    let cols = d_l;
    let mut k = vec![0.0; rows * cols];
    let cidx = |r: usize, c: usize| r * c_cols + c;

    // The (x, μ) block: C[i][j] = W[i][j], C[i][d + j] = −W[i][j].
    for i in 0..d {
        for j in 0..=i {
            k[cidx(i, j) * cols + packed(i, j)] += 1.0;
            k[cidx(i, d + j) * cols + packed(i, j)] -= 1.0;
        }
    }

    // The L block: C[d + r·d + c][2d + q] = (G_q[r][c] + G_q[c][r]) / √2, with
    // G_q[r][c] = W[r][a] when c == b, for q = packed(a, b).
    let s = std::f64::consts::FRAC_1_SQRT_2;
    for a in 0..d {
        for b in 0..=a {
            let q = packed(a, b);
            for r in 0..d {
                for c in 0..d {
                    let row = cidx(d + r * d + c, 2 * d + q);
                    if c == b && a <= r {
                        k[row * cols + packed(r, a)] += s;
                    }
                    if r == b && a <= c {
                        k[row * cols + packed(c, a)] += s;
                    }
                }
            }
        }
    }

    (rows, cols, k)
}

/// The Gaussian Fisher root over `[x, μ, L]` as a node of type
/// `EuclideanMatrixD(d + d², 2d + d(d+1)/2)`, built from the whitening `w = L⁻¹`.
///
/// `d` is the manifold dimension of the Gaussian's quantity.
///
/// Exact on flat manifolds, and second-order accurate in the dispersion on groups; see the
/// [module documentation](self).
pub fn information_root<B: Backend>(
    w: &Q<B, Tril>,
    d: usize,
) -> SkoroResult<Q<B, EuclideanMatrixD>> {
    let (rows, cols, data) = gaussian_root_matrix(d);
    let map = FixedLinearMap::from_row_major(rows, cols, &data, &Default::default());
    linear_map(w, EuclideanMatrixD(d + d * d, 2 * d + d * (d + 1) / 2), map)
}
