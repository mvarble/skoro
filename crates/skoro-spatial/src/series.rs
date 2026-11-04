//! The coefficients of the removable singularities, each computed by the core's guarded pattern: a
//! discretization of its argument against a cutoff selects a truncated series below it, and the
//! closed form's argument is replaced by a harmless one on the rows where the series is chosen, as
//! the series' is on the rows where the closed form is, wherever the series could be undefined.
//!
//! One cutoff serves both precisions: the series' truncation at the cutoff is below double
//! precision's rounding, and the closed form just above it agrees with double precision in single
//! precision, its value and its first and second derivatives, within the core's agreement.

use skoro_core::QDyn;

use crate::algebra::{both, choose, one, reciprocal, sqrt};

/// The cutoff in `θ²` below which the series of `a`, `b`, `A`, `B`, `C`, `D` and `E` are used.
pub(crate) const CUTOFF: f64 = 0.1;

/// The cutoff in `x = n²/w²` below which the series of `L` is used.
pub(crate) const LOG_CUTOFF: f64 = 0.01;

/// Which coefficient, of `θ²`.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) enum Coefficient {
    /// `a = cos(θ/2)`, of the exponential.
    A0,
    /// `b = sin(θ/2)/θ`, of the exponential.
    B0,
    /// `A = (1 - cos θ)/θ²`, of the Jacobians.
    A,
    /// `B = (θ - sin θ)/θ³`, of the Jacobians and of `Q`.
    B,
    /// `C = (1 - (θ/2) cot(θ/2))/θ²`, of the Jacobians' inverses.
    C,
    /// `D = (θ² + 2 cos θ - 2)/(2θ⁴)`, of `Q`.
    #[cfg_attr(
        not(test),
        expect(dead_code, reason = "used by the motions' Jacobians")
    )]
    D,
    /// `E = (2θ - 3 sin θ + θ cos θ)/(2θ⁵)`, of `Q`.
    #[cfg_attr(
        not(test),
        expect(dead_code, reason = "used by the motions' Jacobians")
    )]
    E,
}

/// `n!` as a float.
const fn factorial(n: u32) -> f64 {
    let mut f = 1.0;
    let mut k = 2;
    while k <= n {
        f *= k as f64;
        k += 1;
    }
    f
}

/// `(-1)^k`.
const fn sign(k: u32) -> f64 {
    if k.is_multiple_of(2) { 1.0 } else { -1.0 }
}

/// `4^k`.
const fn four(k: u32) -> f64 {
    let mut p = 1.0;
    let mut i = 0;
    while i < k {
        p *= 4.0;
        i += 1;
    }
    p
}

/// The series' coefficients in `θ²`, from the constant term, each kept through `θ¹⁰`, `C`'s
/// through `θ¹²`.
const fn series(c: Coefficient) -> &'static [f64] {
    match c {
        Coefficient::A0 => &A0_SERIES,
        Coefficient::B0 => &B0_SERIES,
        Coefficient::A => &A_SERIES,
        Coefficient::B => &B_SERIES,
        Coefficient::C => &C_SERIES,
        Coefficient::D => &D_SERIES,
        Coefficient::E => &E_SERIES,
    }
}

macro_rules! table {
    ($name:ident, $k:literal, |$i:ident| $term:expr) => {
        const $name: [f64; $k] = {
            let mut out = [0.0; $k];
            let mut $i: u32 = 0;
            while ($i as usize) < $k {
                out[$i as usize] = $term;
                $i += 1;
            }
            out
        };
    };
}

// a = Σ (-1)^k θ^{2k} / (4^k (2k)!)
table!(A0_SERIES, 6, |k| sign(k) / (four(k) * factorial(2 * k)));
// b = Σ (-1)^k θ^{2k} / (2^{2k+1} (2k+1)!)
table!(B0_SERIES, 6, |k| sign(k)
    / (2.0 * four(k) * factorial(2 * k + 1)));
// A = Σ (-1)^k θ^{2k} / (2k+2)!
table!(A_SERIES, 6, |k| sign(k) / factorial(2 * k + 2));
// B = Σ (-1)^k θ^{2k} / (2k+3)!
table!(B_SERIES, 6, |k| sign(k) / factorial(2 * k + 3));
// D = Σ (-1)^k θ^{2k} / (2k+4)!
table!(D_SERIES, 6, |k| sign(k) / factorial(2 * k + 4));
// E = Σ (-1)^k (k+1) θ^{2k} / (2k+5)!
table!(E_SERIES, 6, |k| sign(k) * (k as f64 + 1.0)
    / factorial(2 * k + 5));
// L = (2/w) Σ (-1)^k x^k / (2k+1), the sum without its factor 2/w.
table!(LOG_SERIES, 8, |k| sign(k) / (2 * k + 1) as f64);

/// `C = Σ (-1)^k B_{2k+2} θ^{2k} / (2k+2)!`, with the Bernoulli numbers `B₂ … B₁₄`.
const C_SERIES: [f64; 7] = [
    1.0 / 12.0,
    1.0 / 720.0,
    1.0 / 30240.0,
    1.0 / 1_209_600.0,
    1.0 / 47_900_160.0,
    (691.0 / 2730.0) / 479_001_600.0,
    (7.0 / 6.0) / 87_178_291_200.0,
];

/// `Σ cₖ xᵏ`, by Horner's rule.
pub(crate) fn horner(coefficients: &[f64], x: QDyn) -> QDyn {
    let mut terms = coefficients.iter().rev();
    let (Some(&last), Some(&next)) = (terms.next(), terms.next()) else {
        unreachable!("every series keeps two terms or more")
    };
    let mut sum = x * last + next;
    for &c in terms {
        sum = sum * x + c;
    }
    sum
}

/// The closed form of `c` at `θ` and `θ² = theta2`, both positive.
fn closed(c: Coefficient, theta: QDyn, theta2: QDyn) -> QDyn {
    let inverse2 = reciprocal(theta2);
    match c {
        Coefficient::A0 => (theta * 0.5).cos(),
        Coefficient::B0 => (theta * 0.5).sin() * reciprocal(theta),
        Coefficient::A => (theta.cos() * -1.0 + 1.0) * inverse2,
        Coefficient::B => (theta - theta.sin()) * inverse2 * reciprocal(theta),
        Coefficient::C => {
            // cot(θ/2) = cos(θ/2) e^{-log sin(θ/2)}, positive in its logarithm on (0, 2π).
            let half = theta * 0.5;
            let cot = half.cos() * reciprocal(half.sin());
            (half * cot * -1.0 + 1.0) * inverse2
        }
        Coefficient::D => (theta2 + theta.cos() * 2.0 - 2.0) * inverse2 * inverse2 * 0.5,
        Coefficient::E => {
            let numerator = theta * 2.0 - theta.sin() * 3.0 + theta * theta.cos();
            numerator * inverse2 * inverse2 * reciprocal(theta) * 0.5
        }
    }
}

/// The coefficient `c` at `θ² = theta2`, a `Real`: the series below `CUTOFF`, the closed form
/// above, its argument guarded where the series is chosen.
pub(crate) fn coefficient(c: Coefficient, theta2: QDyn) -> QDyn {
    let small = (theta2 * -1.0 + CUTOFF).is_nonnegative();
    let safe = choose(small, theta2, one());
    let theta = sqrt(safe);
    choose(small, closed(c, theta, safe), horner(series(c), theta2))
}

/// `L = 2 atan2(n, w) / n` of the logarithm, from `n²` and `w`: its series in `x = n²/w²` where
/// `n² < LOG_CUTOFF w²` and `w ≥ 0`, the closed form elsewhere, each guarded where the other is
/// chosen. At `w = 0` the series is chosen only for `n = 0`, which is no unit quaternion.
pub(crate) fn log_ratio(n2: QDyn, w: QDyn) -> QDyn {
    let small = both(
        w.is_nonnegative(),
        (w * w * LOG_CUTOFF - n2).is_nonnegative(),
    );
    let n = sqrt(choose(small, n2, one()));
    let w_safe = choose(small, one(), w);
    let x = n2 * reciprocal(w_safe * w_safe);
    let closed = n.atan2(w) * reciprocal(n) * 2.0;
    let series = horner(&LOG_SERIES, x) * reciprocal(w_safe) * 2.0;
    choose(small, closed, series)
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

    use super::*;
    use crate::algebra::entry;
    use crate::algebra::tests::eval;

    fn closed_on_host(c: Coefficient, t2: f64) -> f64 {
        let t = t2.sqrt();
        match c {
            Coefficient::A0 => (t / 2.0).cos(),
            Coefficient::B0 => (t / 2.0).sin() / t,
            Coefficient::A => (1.0 - t.cos()) / t2,
            Coefficient::B => (t - t.sin()) / (t2 * t),
            Coefficient::C => (1.0 - t / 2.0 / (t / 2.0).tan()) / t2,
            Coefficient::D => (t2 + 2.0 * t.cos() - 2.0) / (2.0 * t2 * t2),
            Coefficient::E => (2.0 * t - 3.0 * t.sin() + t * t.cos()) / (2.0 * t2 * t2 * t),
        }
    }

    #[test]
    fn coefficients_are_their_closed_forms_away_from_the_identity() {
        let all = [
            Coefficient::A0,
            Coefficient::B0,
            Coefficient::A,
            Coefficient::B,
            Coefficient::C,
            Coefficient::D,
            Coefficient::E,
        ];
        for c in all {
            for t2 in [0.5, 2.0, 9.0] {
                let got = eval(|h| coefficient(c, entry(h[0], 0)), &[&[t2]])[0];
                let want = closed_on_host(c, t2);
                assert!(
                    (got - want).abs() < 1e-12 * want.abs().max(1.0),
                    "{c:?} at {t2}"
                );
            }
            // At the identity, the series' constant term.
            let got = eval(|h| coefficient(c, entry(h[0], 0)), &[&[0.0]])[0];
            assert_eq!(got, series(c)[0], "{c:?} at 0");
        }
    }

    #[test]
    fn log_ratio_is_its_closed_form() {
        for (n2, w) in [
            (0.0, 1.0),
            (1e-6, 0.9999995),
            (0.5, 0.5f64.sqrt()),
            (0.99, -0.1),
        ] {
            let got = eval(|h| log_ratio(entry(h[0], 0), entry(h[0], 1)), &[&[n2, w]])[0];
            let n: f64 = f64::sqrt(n2);
            let want = if n2 == 0.0 {
                2.0 / w
            } else {
                2.0 * n.atan2(w) / n
            };
            assert!(
                (got - want).abs() < 1e-12,
                "({n2}, {w}): {got} against {want}"
            );
        }
    }
}
