//! Factorizations and solves over [`Dense`], backed by `faer`.
//!
//! All six entry points return [`SkoroResult`]. That is the design argument made
//! structural: a matrix that has stopped being positive definite is a condition the caller
//! must be able to observe, not a `NaN` that surfaces several thousand iterations later.
//!
//! Five of the six could be reimplemented on `Tensor<B, 3>` behind the same names if a
//! batched consumer ever appears — [`whitening_from_precision`] is a composition of three of the
//! others. [`lstsq`] could not — `burn` 0.21 has no QR. An algorithm written against it is
//! committed to the host path, deliberately.

use faer::linalg::solvers::{ColPivQr, Llt, Solve, SolveLstsq};
use faer::linalg::triangular_solve::{
    solve_lower_triangular_in_place, solve_upper_triangular_in_place,
};
use faer::{Mat, MatRef, Par, Side};
use skoro_core::error::{SkoroError, SkoroResult};

use crate::dense::Dense;

/// The lower Cholesky factor `L` with `A = L Lᵀ`, or [`SkoroError::NotPositiveDefinite`].
///
/// Only the lower triangle of `spd` is read.
pub fn cholesky(spd: &Dense) -> SkoroResult<Dense> {
    let a = to_faer(square(spd)?);
    let llt = Llt::new(a.as_ref(), Side::Lower)
        .map_err(|_| SkoroError::NotPositiveDefinite { rows: spd.rows() })?;
    Ok(from_faer(llt.L()))
}

/// Solves `L x = b` for a lower-triangular `L`.
pub fn solve_lower(l: &Dense, b: &Dense) -> SkoroResult<Dense> {
    let l = to_faer(square(l)?);
    let mut rhs = to_faer(compatible(&l.nrows(), b)?);
    solve_lower_triangular_in_place(l.as_ref(), rhs.as_mut(), Par::Seq);
    Ok(from_faer(rhs.as_ref()))
}

/// Solves `U x = b` for an upper-triangular `U`.
pub fn solve_upper(u: &Dense, b: &Dense) -> SkoroResult<Dense> {
    let u = to_faer(square(u)?);
    let mut rhs = to_faer(compatible(&u.nrows(), b)?);
    solve_upper_triangular_in_place(u.as_ref(), rhs.as_mut(), Par::Seq);
    Ok(from_faer(rhs.as_ref()))
}

/// Solves `A x = b` for a symmetric positive-definite `A`, by Cholesky.
///
/// Errors [`SkoroError::NotPositiveDefinite`] rather than returning `NaN`s. Only the lower
/// triangle of `a` is read.
pub fn spd_solve(a: &Dense, b: &Dense) -> SkoroResult<Dense> {
    let mat = to_faer(square(a)?);
    let rhs = to_faer(compatible(&mat.nrows(), b)?);
    let llt = Llt::new(mat.as_ref(), Side::Lower)
        .map_err(|_| SkoroError::NotPositiveDefinite { rows: a.rows() })?;
    Ok(from_faer(llt.solve(rhs.as_ref()).as_ref()))
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
    let a = to_faer(j);
    let rhs = to_faer(r);
    let qr = ColPivQr::new(a.as_ref());
    Ok(from_faer(qr.solve_lstsq(rhs.as_ref()).as_ref()))
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
fn compatible<'a>(n: &usize, b: &'a Dense) -> SkoroResult<&'a Dense> {
    if b.rows() != *n {
        return Err(SkoroError::ShapeMismatch {
            expected: *n,
            found: b.rows(),
        });
    }
    Ok(b)
}

fn to_faer(m: &Dense) -> Mat<f64> {
    Mat::from_fn(m.rows(), m.cols(), |r, c| m.get(r, c))
}

fn from_faer(m: MatRef<'_, f64>) -> Dense {
    let (rows, cols) = (m.nrows(), m.ncols());
    let mut data = Vec::with_capacity(rows * cols);
    for r in 0..rows {
        for c in 0..cols {
            data.push(m[(r, c)]);
        }
    }
    Dense::from_row_major(rows, cols, data)
        .expect("internal: the row-major length is rows * cols by construction")
}

/// The whitening of a precision matrix, packed for a `Tril` quantity, with its log-determinant.
///
/// Given a precision `Λ`, returns `(w, ln|det W|)` where `W` is lower-triangular with
/// `Wᵀ W = Λ`, and `w` is `W`'s lower triangle in `Tril` packing order — row-major, `(i, j)` at
/// `i(i+1)/2 + j`. A residual `r = W δ` then has `‖r‖² = δᵀ Λ δ`, which is the convention
/// `skoro-normal` already uses for `Normal`, so the result drops straight into a `Tril` constant.
///
/// **This is the composition, not a new factorization**: `Σ = Λ⁻¹` by SPD solve, `L = chol(Σ)`,
/// then `W = L⁻¹` by a triangular solve against the identity. `W` comes out lower-triangular
/// because the inverse of a lower-triangular matrix is lower-triangular, which is what lets it
/// live in a `Tril` at all — `chol(Λ)ᵀ` would satisfy the same equation but is upper.
///
/// The route costs an inversion that cancels analytically but not numerically, and `Λ` is
/// ill-conditioned exactly when the blanket is under-constrained — which is the case a marginal
/// over a poorly observed window produces. [`SkoroError::NotPositiveDefinite`] is the honest
/// failure there, and it is reported rather than returned as `NaN`s.
pub fn whitening_from_precision(precision: &Dense) -> SkoroResult<(Vec<f64>, f64)> {
    let n = square(precision)?.rows();
    let identity = Dense::identity(n);
    let covariance = spd_solve(precision, &identity)?;
    let l = cholesky(&covariance)?;
    let w = solve_lower(&l, &identity)?;

    let mut packed = Vec::with_capacity(n * (n + 1) / 2);
    let mut log_det = 0.0;
    for i in 0..n {
        for j in 0..=i {
            packed.push(w.get(i, j));
        }
        // `W` is triangular, so its determinant is the product of the diagonal — and the diagonal
        // of an inverse-Cholesky factor is strictly positive, so the log is real.
        log_det += w.get(i, i).abs().ln();
    }
    Ok((packed, log_det))
}
