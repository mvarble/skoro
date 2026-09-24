//! Factorizations and solves over [`Dense`], backed by `faer`.
//!
//! Every entry point returns [`SkoroResult`]. That is the design argument made
//! structural: a matrix that has stopped being positive definite is a condition the caller
//! must be able to observe, not a `NaN` that surfaces several thousand iterations later.
//!
//! All but [`lstsq`] could be reimplemented on `Tensor<B, 3>` behind the same names if a batched
//! consumer ever appears. [`lstsq`] could not — `burn` 0.21 has no QR. An algorithm written against
//! it is committed to the host path, deliberately.

use faer::linalg::solvers::{ColPivQr, Llt, Solve, SolveLstsq};
use faer::linalg::triangular_solve::{
    solve_lower_triangular_in_place, solve_upper_triangular_in_place,
};
use faer::{Par, Side};
use skoro_core::error::{SkoroError, SkoroResult};

use crate::dense::Dense;

/// The lower Cholesky factor `L` with `A = L Lᵀ`, or [`SkoroError::NotPositiveDefinite`].
///
/// Only the lower triangle of `spd` is read.
pub fn cholesky(spd: &Dense) -> SkoroResult<Dense> {
    let llt = Llt::new(square(spd)?.as_faer(), Side::Lower)
        .map_err(|_| SkoroError::NotPositiveDefinite { rows: spd.rows() })?;
    Ok(Dense::from_faer_ref(llt.L()))
}

/// Solves `L x = b` for a lower-triangular `L`.
pub fn solve_lower(l: &Dense, b: &Dense) -> SkoroResult<Dense> {
    let mut rhs = compatible(square(l)?.rows(), b)?.clone().into_faer();
    solve_lower_triangular_in_place(l.as_faer(), rhs.as_mut(), Par::Seq);
    Ok(Dense::from_faer(rhs))
}

/// Solves `U x = b` for an upper-triangular `U`.
pub fn solve_upper(u: &Dense, b: &Dense) -> SkoroResult<Dense> {
    let mut rhs = compatible(square(u)?.rows(), b)?.clone().into_faer();
    solve_upper_triangular_in_place(u.as_faer(), rhs.as_mut(), Par::Seq);
    Ok(Dense::from_faer(rhs))
}

/// Solves `A x = b` for a symmetric positive-definite `A`, by Cholesky.
///
/// Errors [`SkoroError::NotPositiveDefinite`] rather than returning `NaN`s. Only the lower
/// triangle of `a` is read.
pub fn spd_solve(a: &Dense, b: &Dense) -> SkoroResult<Dense> {
    let rhs = compatible(square(a)?.rows(), b)?;
    let llt = Llt::new(a.as_faer(), Side::Lower)
        .map_err(|_| SkoroError::NotPositiveDefinite { rows: a.rows() })?;
    Ok(Dense::from_faer(llt.solve(rhs.as_faer())))
}

/// Least squares by column-pivoted QR: `min ‖J δ − r‖`, without forming `JᵀJ`.
///
/// Forming `JᵀJ` squares the condition number; this path works at `κ` rather than `κ²`,
/// which is the whole argument for the host boundary (`docs/skoro.md` §4.2).
///
/// This gives the **undamped** step. Damping is the caller's job and costs nothing extra:
/// `(JᵀJ + λI) δ = −Jᵀr` is exactly `min ‖[J; √λ I] δ − [−r; 0]‖`, so an iteration
/// augments with [`Dense::vstack`] and calls this. The QR path is therefore available at
/// every damping value.
pub fn lstsq(j: &Dense, r: &Dense) -> SkoroResult<Dense> {
    if j.rows() != r.rows() {
        return Err(SkoroError::ShapeMismatch {
            expected: j.rows(),
            found: r.rows(),
        });
    }
    if j.rows() < j.cols() {
        // faer's least-squares solve is defined for a tall or square system; an
        // underdetermined one has no unique answer and would read past the factor.
        return Err(SkoroError::ShapeMismatch {
            expected: j.cols(),
            found: j.rows(),
        });
    }
    let qr = ColPivQr::new(j.as_faer());
    Ok(Dense::from_faer(qr.solve_lstsq(r.as_faer())))
}

/// Reject a non-square matrix where a factorization needs one.
fn square(m: &Dense) -> SkoroResult<&Dense> {
    if m.rows() != m.cols() {
        return Err(SkoroError::ShapeMismatch {
            expected: m.rows(),
            found: m.cols(),
        });
    }
    Ok(m)
}

/// Reject a right-hand side whose row count does not match the system.
fn compatible(n: usize, b: &Dense) -> SkoroResult<&Dense> {
    if b.rows() != n {
        return Err(SkoroError::ShapeMismatch {
            expected: n,
            found: b.rows(),
        });
    }
    Ok(b)
}

/// The whitening of a precision matrix, packed for a `Tril` quantity, with its log-determinant.
///
/// Given a precision `Λ`, returns `(w, ln|det W|)` where `W` is lower-triangular with a positive
/// diagonal and `Wᵀ W = Λ`, and `w` is `W`'s lower triangle in `Tril` packing order — row-major,
/// `(i, j)` at `i(i+1)/2 + j`. A residual `r = W δ` then has `‖r‖² = δᵀ Λ δ`, which is the
/// convention `skoro-normal` already uses for `Normal`, so the result drops straight into a `Tril`
/// constant — for a `JointNormal` built by hand from a precision the caller already has. A
/// marginal computed by elimination arrives as a whitening already
/// ([`Marginal`](crate::marginal::Marginal)), and never needs this.
///
/// **One Cholesky factorization, no inversion.** With `P` the reversal permutation,
/// `PΛP = L Lᵀ` for a lower `L`, and `W = P Lᵀ P` is lower-triangular with
/// `WᵀW = P L Lᵀ P = Λ`. `chol(Λ)ᵀ` would satisfy the same equation but is upper, which a `Tril`
/// cannot hold; the reversal is what makes the factor lower. (This used to go
/// `Λ → Σ = Λ⁻¹ → chol(Σ) → inverse`, through an inversion that cancels analytically but not
/// numerically.)
///
/// [`SkoroError::NotPositiveDefinite`] is the honest failure for a `Λ` that is not positive definite
/// — an under-constrained blanket — and it is reported rather than returned as `NaN`s.
pub fn whitening_from_precision(precision: &Dense) -> SkoroResult<(Vec<f64>, f64)> {
    let n = square(precision)?.rows();
    let reversed = Dense::from_fn(n, n, |i, j| precision.get(n - 1 - i, n - 1 - j));
    let l = cholesky(&reversed)?;

    let mut packed = Vec::with_capacity(n * (n + 1) / 2);
    let mut log_det = 0.0;
    for i in 0..n {
        // `W[i][j] = Lᵀ[n−1−i][n−1−j] = L[n−1−j][n−1−i]`.
        for j in 0..=i {
            packed.push(l.get(n - 1 - j, n - 1 - i));
        }
        log_det += l.get(n - 1 - i, n - 1 - i).ln();
    }
    Ok((packed, log_det))
}
