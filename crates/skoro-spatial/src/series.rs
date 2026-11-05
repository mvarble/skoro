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
    D,
    /// `E = (2θ - 3 sin θ + θ cos θ)/(2θ⁵)`, of `Q`.
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

/// The Bernoulli numbers `B₂, B₄, …, B₂₄`.
const BERNOULLI: [f64; 12] = [
    1.0 / 6.0,
    -1.0 / 30.0,
    1.0 / 42.0,
    -1.0 / 30.0,
    5.0 / 66.0,
    -691.0 / 2730.0,
    7.0 / 6.0,
    -3617.0 / 510.0,
    43867.0 / 798.0,
    -174611.0 / 330.0,
    854513.0 / 138.0,
    -236364091.0 / 2730.0,
];

// C = Σ (-1)^k B_{2k+2} θ^{2k} / (2k+2)!
table!(C_SERIES, 7, |k| sign(k) * BERNOULLI[k as usize]
    / factorial(2 * k + 2));

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

    use std::sync::Arc;

    use skoro_core::scope::constant;
    use skoro_core::testing::check_operation;
    use skoro_core::{Composite, Declares, Space, VectorDyn};
    use skoro_tensor::{HostData, HostValue, Precision};

    use super::*;
    use crate::algebra::entry;
    use crate::algebra::tests::eval;
    use crate::{ExtendedPose, ExtendedPoseMaps, Pose, PoseMaps, Rotation, RotationMaps};

    fn closed_on_host(c: Coefficient, t2: f64) -> f64 {
        let t = (0.5 * t2.ln()).exp();
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

    /// `ζ(s)` for even `s ≥ 2`: exact at `2` and `4`, where a direct sum converges far too slowly,
    /// and beyond them summed to `J = 1000` with the Euler–Maclaurin tail
    /// `J^{1-s}/(s-1) - J^{-s}/2`, whose own error, below `s J^{-s-1}/12`, is far below rounding.
    fn zeta(s: u32) -> f64 {
        use std::f64::consts::PI;
        match s {
            2 => PI * PI / 6.0,
            4 => PI.powi(4) / 90.0,
            _ => {
                let (s, j) = (s as i32, 1000.0f64);
                let sum: f64 = (1..=1000).rev().map(|i| f64::from(i).powi(-s)).sum();
                sum + j.powi(1 - s) / f64::from(s - 1) - 0.5 * j.powi(-s)
            }
        }
    }

    /// `C`'s `k`-th coefficient in `θ²`, `(-1)^k B_{2k+2} / (2k+2)! = 2 ζ(2k+2) / (2π)^{2k+2}`,
    /// independent of `BERNOULLI`.
    fn c_term(k: u32) -> f64 {
        2.0 * zeta(2 * k + 2) / (2.0 * std::f64::consts::PI).powi(2 * k as i32 + 2)
    }

    /// The coefficient's series in `θ²`, summed until its terms are negligible.
    fn summed(c: Coefficient, t2: f64) -> f64 {
        let term = |k: u32| -> f64 {
            let x = t2.powi(k as i32);
            match c {
                Coefficient::A0 => sign(k) * x / (four(k) * factorial(2 * k)),
                Coefficient::B0 => sign(k) * x / (2.0 * four(k) * factorial(2 * k + 1)),
                Coefficient::A => sign(k) * x / factorial(2 * k + 2),
                Coefficient::B => sign(k) * x / factorial(2 * k + 3),
                Coefficient::C => c_term(k) * x,
                Coefficient::D => sign(k) * x / factorial(2 * k + 4),
                Coefficient::E => sign(k) * (k as f64 + 1.0) * x / factorial(2 * k + 5),
            }
        };
        // Summed from the smallest term up, twenty terms being far below rounding at θ² ≤ 0.11.
        (0..20).rev().map(term).sum()
    }

    #[test]
    fn the_bernoulli_numbers_are_the_zeta_values() {
        // B_{2n} = (-1)^{n+1} 2 ζ(2n) (2n)! / (2π)^{2n}.
        for (i, &b) in BERNOULLI.iter().enumerate() {
            let n = i as u32 + 1;
            let want = -sign(n) * 2.0 * zeta(2 * n) * factorial(2 * n)
                / (2.0 * std::f64::consts::PI).powi(2 * n as i32);
            assert!(
                (b - want).abs() <= 1e-14 * want.abs(),
                "B_{}: {b} against {want}",
                2 * n
            );
        }
    }

    /// `L`'s series in `x = n²/w²`, summed until its terms are negligible.
    fn summed_log(n2: f64, w: f64) -> f64 {
        let x = n2 / (w * w);
        let sum: f64 = (0..30u32)
            .rev()
            .map(|k| sign(k) * x.powi(k as i32) / (2 * k + 1) as f64)
            .sum();
        2.0 / w * sum
    }

    const ALL: [Coefficient; 7] = [
        Coefficient::A0,
        Coefficient::B0,
        Coefficient::A,
        Coefficient::B,
        Coefficient::C,
        Coefficient::D,
        Coefficient::E,
    ];

    /// `composite` with a second, `Real` input that adds zero to its output: `check_operation`
    /// asks a composite's second-order map with its only input's velocity missing for a frame as
    /// wide as the velocities, which a composite given no velocity at all cannot know.
    fn with_real(composite: Arc<Composite>) -> Arc<Composite> {
        let mut inputs = composite.inputs().to_vec();
        inputs.push(Space::Real);
        let output = composite.output().clone();
        Composite::define("test.spatial.with_real", Vec::new(), &inputs, |q| {
            let (out, r) = (composite.call(&q[..q.len() - 1]), q[q.len() - 1]);
            match &output {
                Space::Embedded(declaration) => {
                    let zeros = vec![0.0; declaration.dimension()];
                    let zeros = constant(VectorDyn(zeros.len()), zeros.as_slice());
                    out.retract(r * zeros.untyped())
                }
                space => {
                    let zeros = vec![0.0; space.storage()];
                    let zeros = constant(VectorDyn(zeros.len()), zeros.as_slice());
                    out + r * zeros.untyped().reshape(space.clone())
                }
            }
        })
        .unwrap()
    }

    /// A composite of `inputs` applying `f`.
    fn wrap(inputs: &[Space], f: impl FnOnce(&[QDyn]) -> QDyn) -> Arc<Composite> {
        with_real(Composite::define("test.spatial.map", Vec::new(), inputs, f).unwrap())
    }

    fn rows(space: Space, data: Vec<f64>) -> (Space, HostValue) {
        let storage = space.storage();
        let value = HostValue::new(
            Some(data.len() / storage),
            storage,
            HostData::Continuous(data),
        );
        (space, value.unwrap())
    }

    /// A composite to check, by name, with its inputs.
    type Check = (&'static str, Arc<Composite>, Vec<(Space, HostValue)>);

    /// The quaternion of the turn by `angle` about a fixed axis.
    fn turn(angle: f64) -> [f64; 4] {
        let axis = [0.3f64, -0.8, 0.5];
        let n = axis.iter().map(|v| v * v).sum::<f64>().sqrt();
        let (s, c) = (0.5 * angle).sin_cos();
        [c, s * axis[0] / n, s * axis[1] / n, s * axis[2] / n]
    }

    skoro_core::precision_tests! {
        fn series_meet_closed_forms(target) {
            // In double precision, each coefficient against its series summed to convergence, on
            // both sides of its cutoff.
            if target.precision() == Precision::Double {
                for c in ALL {
                    // Below the cutoff the series, against its sum to convergence: its truncation
                    // is below rounding. Above it the closed form, as written, its θ the
                    // composite's e^{½ log θ²}.
                    let below = 0.9 * CUTOFF;
                    let got = eval(|h| coefficient(c, entry(h[0], 0)), &[&[below]])[0];
                    let want = summed(c, below);
                    assert!((got - want).abs() <= 1e-15, "{c:?} at θ² = {below}: {got} against {want}");
                    let above = 1.1 * CUTOFF;
                    let got = eval(|h| coefficient(c, entry(h[0], 0)), &[&[above]])[0];
                    let want = closed_on_host(c, above);
                    assert!((got - want).abs() <= 1e-15, "{c:?} at θ² = {above}: {got} against {want}");
                }
                for x in [0.9 * LOG_CUTOFF, 1.1 * LOG_CUTOFF] {
                    let w = (1.0 / (1.0 + x)).sqrt();
                    let n2 = x * w * w;
                    let got = eval(|h| log_ratio(entry(h[0], 0), entry(h[0], 1)), &[&[n2, w]])[0];
                    let want = if x < LOG_CUTOFF {
                        summed_log(n2, w)
                    } else {
                        let n = (0.5 * n2.ln()).exp();
                        2.0 * n.atan2(w) / n
                    };
                    assert!((got - want).abs() <= 1e-15 * want.abs(), "L at x = {x}: {got} against {want}");
                }
            }

            // Every composite that uses a coefficient, at the identity and at both sides of each
            // cutoff: its passes against finite differences, at both precisions.
            let lengths = [0.0, (0.9 * CUTOFF).sqrt(), (1.1 * CUTOFF).sqrt(), 1.3];
            let omega: Vec<f64> = lengths
                .iter()
                .flat_map(|l| [0.3 * l, -0.8 * l, 0.5 * l].map(|v| v / 0.98995f64.sqrt()))
                .collect();
            let angles = [
                0.0,
                (0.9 * CUTOFF).sqrt(),
                (1.1 * CUTOFF).sqrt(),
                2.0 * (0.9 * LOG_CUTOFF).sqrt().atan(),
                2.0 * (1.1 * LOG_CUTOFF).sqrt().atan(),
                2.5,
            ];
            let quaternions: Vec<[f64; 4]> = angles.iter().map(|a| turn(*a)).collect();
            let translations = [0.7, -1.1, 0.4, 0.2, 0.9, -0.6];
            let with = |k: usize, base: &[f64], width: usize| -> Vec<f64> {
                base.chunks(width)
                    .flat_map(|b| b.iter().copied().chain(translations[..3 * k].iter().copied()).collect::<Vec<_>>())
                    .collect()
            };
            let real = |n: usize| rows(Space::Real, (0..n).map(|i| 0.3 + 0.1 * i as f64).collect());
            let vector3 = rows(Space::Vector(3), omega.clone());
            let n = lengths.len();
            let rotation = Rotation.space();
            let stored: Vec<f64> = quaternions.concat();
            let m = quaternions.len();
            let checks: Vec<Check> = vec![
                ("exp", wrap(&[Space::Vector(3)], |q| crate::rotation::exp(q[0].narrow()).untyped()), vec![vector3.clone(), real(n)]),
                ("jl", wrap(&[Space::Vector(3)], |q| crate::so3::jl(q[0].narrow()).untyped()), vec![vector3.clone(), real(n)]),
                ("jl_inv", wrap(&[Space::Vector(3)], |q| crate::so3::jl_inv(q[0].narrow()).untyped()), vec![vector3.clone(), real(n)]),
                ("log", wrap(std::slice::from_ref(&rotation), |q| q[0].narrow::<Rotation>().log().untyped()), vec![rows(rotation.clone(), stored.clone()), real(m)]),
                ("log_shorter", wrap(std::slice::from_ref(&rotation), |q| q[0].narrow::<Rotation>().log_shorter().untyped()), vec![rows(rotation.clone(), stored.clone()), real(m)]),
                ("pose.exp", wrap(&[Space::Vector(6)], |q| crate::pose::exp(q[0].narrow()).untyped()), vec![rows(Space::Vector(6), with(1, &omega, 3)), real(n)]),
                ("pose.jl", wrap(&[Space::Vector(6)], |q| crate::pose::jl(q[0].narrow()).untyped()), vec![rows(Space::Vector(6), with(1, &omega, 3)), real(n)]),
                ("pose.jl_inv", wrap(&[Space::Vector(6)], |q| crate::pose::jl_inv(q[0].narrow()).untyped()), vec![rows(Space::Vector(6), with(1, &omega, 3)), real(n)]),
                ("pose.log", wrap(&[Pose.space()], |q| q[0].narrow::<Pose>().log().untyped()), vec![rows(Pose.space(), with(1, &stored, 4)), real(m)]),
                ("pose.log_shorter", wrap(&[Pose.space()], |q| q[0].narrow::<Pose>().log_shorter().untyped()), vec![rows(Pose.space(), with(1, &stored, 4)), real(m)]),
                ("extended_pose.exp", wrap(&[Space::Vector(9)], |q| crate::extended::exp(q[0].narrow()).untyped()), vec![rows(Space::Vector(9), with(2, &omega, 3)), real(n)]),
                ("extended_pose.jl", wrap(&[Space::Vector(9)], |q| crate::extended::jl(q[0].narrow()).untyped()), vec![rows(Space::Vector(9), with(2, &omega, 3)), real(n)]),
                ("extended_pose.jl_inv", wrap(&[Space::Vector(9)], |q| crate::extended::jl_inv(q[0].narrow()).untyped()), vec![rows(Space::Vector(9), with(2, &omega, 3)), real(n)]),
                ("extended_pose.log", wrap(&[ExtendedPose.space()], |q| q[0].narrow::<ExtendedPose>().log().untyped()), vec![rows(ExtendedPose.space(), with(2, &stored, 4)), real(m)]),
                ("extended_pose.log_shorter", wrap(&[ExtendedPose.space()], |q| q[0].narrow::<ExtendedPose>().log_shorter().untyped()), vec![rows(ExtendedPose.space(), with(2, &stored, 4)), real(m)]),
            ];
            for (name, composite, inputs) in checks {
                if let Err(error) = check_operation(&*composite.operation(), &inputs, &target) {
                    panic!("{name}: {error}");
                }
            }
        }
    }
}
