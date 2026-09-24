//! Identities the host factorizations must satisfy.
//!
//! Nothing here is a hand-computed constant: every expectation is an identity
//! (`A A⁻¹ = I`, `chol(L Lᵀ) = L`) or a closed form the fixture was built from. Per the
//! workspace doctrine, a table of numbers generated from the implementation would only
//! lock in whatever the implementation already did.
//!
//! These tests carry no backend parameter — [`Dense`] is `f64` host memory whatever the
//! backend is. The dtype-sensitive half lives in `common/`, wired per backend.

use approx::assert_abs_diff_eq;
use skoro_core::error::SkoroError;
use skoro_dense::dense::Dense;
use skoro_dense::factor::{
    cholesky, lstsq, solve_lower, solve_upper, spd_solve, whitening_from_precision,
};

/// A lower-triangular factor with a positive diagonal, far from the identity: near `I`
/// almost every triangular bug is invisible.
fn lower_factor() -> Dense {
    #[rustfmt::skip]
    let data = vec![
        2.0,  0.0,  0.0,  0.0,
       -1.5,  3.0,  0.0,  0.0,
        0.75, 2.5,  1.25, 0.0,
       -2.0,  0.5, -1.75, 4.0,
    ];
    Dense::from_row_major(4, 4, data).unwrap()
}

fn spd() -> Dense {
    let l = lower_factor();
    l.matmul(&l.transpose()).unwrap()
}

/// A right-hand side with no zeros and no repeats.
fn rhs() -> Dense {
    Dense::from_row_major(4, 1, vec![1.5, -3.25, 7.0, -0.5]).unwrap()
}

fn assert_close(actual: &Dense, expected: &Dense, epsilon: f64) {
    assert_eq!(
        (actual.rows(), actual.cols()),
        (expected.rows(), expected.cols())
    );
    for (a, e) in actual.to_row_major().iter().zip(&expected.to_row_major()) {
        assert_abs_diff_eq!(a, e, epsilon = epsilon);
    }
}

/// The host factorizations, each checked against an identity rather than a stored answer:
/// `chol(L Lᵀ) = L` recovers the factor itself, since Cholesky is unique for a positive diagonal,
/// and the solves are checked by reproducing the right-hand side they were given.
mod factorizations {
    use super::*;

    /// `chol(L Lᵀ) = L`. The factor is recovered, not merely some factor: Cholesky is unique
    /// for a positive diagonal, which `lower_factor` has.
    #[test]
    fn cholesky_recovers_the_factor_it_was_built_from() {
        let l = lower_factor();
        assert_close(&cholesky(&spd()).unwrap(), &l, 1e-12);
    }

    /// `A⁻¹ A = I`, expressed without ever forming an inverse.
    #[test]
    fn spd_solve_of_a_matrix_by_itself_is_the_identity() {
        let a = spd();
        assert_close(&spd_solve(&a, &a).unwrap(), &Dense::identity(4), 1e-10);
    }

    /// `A x = b` for the `x` the solve returned. Checking the residual rather than a stored
    /// `x` keeps the fixture free of implementation-generated numbers.
    #[test]
    fn spd_solve_reproduces_its_right_hand_side() {
        let a = spd();
        let b = rhs();
        let x = spd_solve(&a, &b).unwrap();
        assert_close(&a.matmul(&x).unwrap(), &b, 1e-10);
    }

    #[test]
    fn the_triangular_solves_invert_their_triangles() {
        let l = lower_factor();
        let u = l.transpose();
        let b = rhs();

        let x = solve_lower(&l, &b).unwrap();
        assert_close(&l.matmul(&x).unwrap(), &b, 1e-12);

        let y = solve_upper(&u, &b).unwrap();
        assert_close(&u.matmul(&y).unwrap(), &b, 1e-12);
    }
}

/// The column-pivoted-QR `lstsq`, and why it earns the host boundary. On a consistent system it
/// reproduces the exact solution; on an ill-conditioned Hilbert system it must *beat* the normal
/// equations by orders of magnitude, since forming `JᵀJ` squares the condition number. Row
/// augmentation is the same identity a Levenberg-Marquardt step relies on at every `λ`.
mod least_squares {
    use super::*;

    /// `lstsq` on an overdetermined system whose solution is exact must reproduce it.
    #[test]
    fn lstsq_is_exact_when_the_system_is_consistent() {
        #[rustfmt::skip]
        let j = Dense::from_row_major(4, 2, vec![
             1.0,  0.5,
            -2.0,  1.5,
             0.25, 3.0,
             1.75, -1.0,
        ]).unwrap();
        let x = Dense::from_row_major(2, 1, vec![2.5, -1.25]).unwrap();
        let r = j.matmul(&x).unwrap();

        assert_close(&lstsq(&j, &r).unwrap(), &x, 1e-12);
    }

    /// The `n × n` Hilbert matrix, `H[i][j] = 1 / (i + j + 1)`. `κ(H₆) ≈ 1.5e7`, so `κ² ≈ 2e14`
    /// — inside `f64`'s range for the Cholesky to succeed, and well outside it for the answer
    /// to survive.
    fn hilbert(n: usize) -> Dense {
        let mut data = Vec::with_capacity(n * n);
        for i in 0..n {
            for j in 0..n {
                data.push(1.0 / (i + j + 1) as f64);
            }
        }
        Dense::from_row_major(n, n, data).unwrap()
    }

    /// **Without this test, the argument for the whole host boundary is an assertion.**
    ///
    /// Forming `JᵀJ` squares the condition number. On a Hilbert `J` the normal equations lose
    /// most of the mantissa and the QR path does not, so the two must disagree by orders of
    /// magnitude with `lstsq` on the right side.
    #[test]
    fn lstsq_beats_the_normal_equations_on_an_ill_conditioned_problem() {
        let n = 6;
        let j = hilbert(n);
        let x = Dense::from_row_major(n, 1, (0..n).map(|i| 1.0 + i as f64).collect()).unwrap();
        let r = j.matmul(&x).unwrap();

        let qr = lstsq(&j, &r).unwrap();
        let jt = j.transpose();
        let normal = spd_solve(&jt.matmul(&j).unwrap(), &jt.matmul(&r).unwrap()).unwrap();

        let err = |sol: &Dense| {
            sol.to_row_major()
                .iter()
                .zip(&x.to_row_major())
                .map(|(a, b)| (a - b).abs())
                .fold(0.0f64, f64::max)
        };
        let (qr_err, normal_err) = (err(&qr), err(&normal));

        assert!(
            qr_err < 1e-6,
            "column-pivoted QR should stay near the analytic solution, got {qr_err}"
        );
        assert!(
            normal_err > 100.0 * qr_err,
            "the normal equations should be visibly worse; qr {qr_err}, normal {normal_err}"
        );
    }

    /// The damping identity `docs/skoro.md` §4.2 rests on:
    /// `(JᵀJ + λI) δ = −Jᵀr` **is** `min ‖[J; √λ I] δ − [−r; 0]‖`.
    ///
    /// On a well-conditioned `J` the two routes must agree, which is what licenses a
    /// Levenberg-Marquardt iteration to reach for `lstsq` at every `λ` rather than only at
    /// `λ = 0`.
    #[test]
    fn row_augmentation_is_the_damped_normal_equations() {
        #[rustfmt::skip]
        let j = Dense::from_row_major(4, 3, vec![
             1.0,  0.5, -2.0,
            -2.0,  1.5,  0.25,
             0.25, 3.0,  1.75,
             1.75, -1.0,  0.5,
        ]).unwrap();
        let r = Dense::from_row_major(4, 1, vec![0.75, -1.5, 2.25, -0.25]).unwrap();
        let lambda: f64 = 0.375;

        let augmented = lstsq(
            &j.vstack(&Dense::scaled_identity(3, lambda.sqrt())).unwrap(),
            &r.neg().vstack(&Dense::zeros(3, 1)).unwrap(),
        )
        .unwrap();

        let jt = j.transpose();
        let damped = {
            let mut data = jt.matmul(&j).unwrap().to_row_major();
            for i in 0..3 {
                data[i * (3 + 1)] += lambda;
            }
            Dense::from_row_major(3, 3, data).unwrap()
        };
        let normal = spd_solve(&damped, &jt.matmul(&r.neg()).unwrap()).unwrap();

        assert_close(&augmented, &normal, 1e-10);
    }
}

/// `W` whitens: `‖W δ‖² = δᵀ Λ δ`. The fixture is deliberately correlated, because a diagonal `Λ`
/// would pass under a transposed or mis-packed `W` -- and this is the property the joint-factor
/// path rests on.
mod whitening {
    use super::*;

    /// `W` whitens: `‖W δ‖² = δᵀ Λ δ`, for a `Λ` that is not diagonal.
    ///
    /// The property the joint-factor path depends on, checked against the definition rather than
    /// against the implementation. A diagonal `Λ` would pass under a transposed or mis-packed `W`, so
    /// the fixture is deliberately correlated.
    #[test]
    fn whitening_reproduces_the_quadratic_form() {
        let lambda =
            Dense::from_row_major(3, 3, vec![4.0, 1.0, 0.5, 1.0, 3.0, -0.75, 0.5, -0.75, 2.5])
                .unwrap();

        let (packed, log_det) = whitening_from_precision(&lambda).unwrap();
        assert_eq!(packed.len(), 6, "lower triangle of a 3x3");

        // Rebuild `W` from the `Tril` packing the way `skoro-linalg` reads it: row-major lower
        // triangle, `(i, j)` at `i(i+1)/2 + j`.
        let mut w_rows = vec![0.0; 9];
        for i in 0..3 {
            for j in 0..=i {
                w_rows[i * 3 + j] = packed[i * (i + 1) / 2 + j];
            }
        }
        let w = Dense::from_row_major(3, 3, w_rows).unwrap();

        for delta in [
            vec![1.0, 0.0, 0.0],
            vec![0.0, 1.0, 0.0],
            vec![0.0, 0.0, 1.0],
            vec![1.0, -2.0, 0.5],
            vec![-0.25, 0.75, 3.0],
        ] {
            let d = Dense::from_row_major(3, 1, delta.clone()).unwrap();
            // `δᵀ Λ δ`
            let quadratic = d.transpose().matmul(&lambda.matmul(&d).unwrap()).unwrap();
            // `‖W δ‖²`
            let r = w.matmul(&d).unwrap();
            let whitened: f64 = r.to_row_major().iter().map(|x| x * x).sum();
            assert!(
                (quadratic.get(0, 0) - whitened).abs() < 1e-10,
                "δ = {delta:?}: δᵀΛδ = {}, ‖Wδ‖² = {whitened}",
                quadratic.get(0, 0)
            );
        }

        // `ln|det W|` is half of `ln det Λ`, since `Λ = WᵀW`.
        let expected = 0.5
            * cholesky(&lambda)
                .unwrap()
                .to_row_major()
                .iter()
                .enumerate()
                .filter(|(k, _)| *k % 4 == 0)
                .map(|(_, v)| v.abs().ln() * 2.0)
                .sum::<f64>();
        assert!(
            (log_det - expected).abs() < 1e-10,
            "log_det {log_det} vs {expected}"
        );
    }
}

/// The failures that must arrive as a `SkoroError` at the call site rather than as `NaN`s further
/// downstream: a covariance that has drifted indefinite, a non-square system, a column-count
/// mismatch, and a singular precision. Asserting the variant is the point -- `is_nan` would pass
/// for a routine that silently gave up.
mod rejections {
    use super::*;

    /// Assert on the variant, not on `is_nan` — the whole point of the `SkoroResult` return
    /// is that a covariance which has drifted indefinite is observable at the call site.
    #[test]
    fn an_indefinite_matrix_is_reported_rather_than_returning_nans() {
        // Symmetric, and `[1, -1]` is an eigenvector with eigenvalue −1.
        #[rustfmt::skip]
        let a = Dense::from_row_major(2, 2, vec![
            1.0, 2.0,
            2.0, 1.0,
        ]).unwrap();
        assert_eq!(
            cholesky(&a),
            Err(SkoroError::NotPositiveDefinite { rows: 2 })
        );
        assert_eq!(
            spd_solve(&a, &Dense::identity(2)),
            Err(SkoroError::NotPositiveDefinite { rows: 2 })
        );
    }

    #[test]
    fn a_non_square_system_is_rejected() {
        let m = Dense::zeros(2, 3);
        assert_eq!(
            cholesky(&m),
            Err(SkoroError::ShapeMismatch {
                expected: 2,
                found: 3
            })
        );
    }

    #[test]
    fn vstack_rejects_a_column_count_mismatch() {
        assert_eq!(
            Dense::zeros(2, 3).vstack(&Dense::zeros(1, 4)),
            Err(SkoroError::ShapeMismatch {
                expected: 3,
                found: 4
            })
        );
    }

    /// An under-constrained blanket gives a singular precision, and that is reported rather than
    /// silently producing `NaN`s.
    #[test]
    fn a_singular_precision_is_rejected() {
        // Rank 1: two identical rows.
        let lambda = Dense::from_row_major(2, 2, vec![1.0, 1.0, 1.0, 1.0]).unwrap();
        assert!(matches!(
            whitening_from_precision(&lambda),
            Err(SkoroError::NotPositiveDefinite { .. })
        ));
    }
}
