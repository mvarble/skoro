//! The Gaussian Fisher root over `[x, μ, L]` — see
//! `docs/superpowers/specs/2026-09-01-factor-bindings-and-fisher-root-spec.md` §4.1 for the
//! derivation this implements exactly.

use skoro_backend::Backend;
use skoro_core::error::SkoroResult;
use skoro_linalg::matrix::{FixedLinearMap, linear_map};
use skoro_linalg::tril::Tril;
use skoro_model::q::Q;
use skoro_riemann::matrix::EuclideanMatrixD;

/// `K` with `vec(C) = K · vec(W)`, `W = L⁻¹` packed as a `Tril`, `C` the Gaussian Fisher root
/// over `[x, μ, L]`. Returns `(rows, cols, row-major data)` with `rows = (d + d²)(2d + d(d+1)/2)`,
/// `cols = d(d+1)/2`.
pub fn gaussian_root_matrix(d: usize) -> (usize, usize, Vec<f64>) {
    let d_l = d * (d + 1) / 2;
    let c_cols = 2 * d + d_l;
    let c_rows = d + d * d;
    let rows = c_rows * c_cols;
    let cols = d_l;
    let mut k = vec![0.0_f64; rows * cols];
    let packed = |i: usize, j: usize| i * (i + 1) / 2 + j;
    let cidx = |r: usize, c: usize| r * c_cols + c;

    for i in 0..d {
        for j in 0..=i {
            k[cidx(i, j) * cols + packed(i, j)] += 1.0;
            k[cidx(i, d + j) * cols + packed(i, j)] += -1.0;
        }
    }

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

/// The Gaussian Fisher root as a graph node: `EuclideanMatrixD(d + d², 2d + d(d+1)/2)`, built
/// from `w = L⁻¹` (packed `Tril`) through the fixed linear map [`gaussian_root_matrix`]
/// describes.
pub fn information_root<B: Backend>(
    w: &Q<B, Tril>,
    d: usize,
) -> SkoroResult<Q<B, EuclideanMatrixD>> {
    let (rows, cols, data) = gaussian_root_matrix(d);
    let map = FixedLinearMap::from_row_major(rows, cols, &data, &Default::default());
    let d_l = d * (d + 1) / 2;
    linear_map(w, EuclideanMatrixD(d + d * d, 2 * d + d_l), map)
}
