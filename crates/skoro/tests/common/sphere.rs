//! The unit sphere `S²` of the acceptance test: storage 3, dimension 2, with the basis `B(x)`
//! smooth off the south pole and the inverse stereographic retraction scaled so that its
//! derivative at 0 is `B(x)`.

use std::sync::{Arc, LazyLock};

use skoro::primitive::Select;
use skoro::scope::{apply, constant};
use skoro::tensor::{HostData, HostValue};
use skoro::{Composite, Declares, Embedded, IntoHostValue, QDyn, Real, Space, Typed, Volume};

/// `1 / a`, written `e^{-log a}`, for `a` positive on the domain.
pub fn reciprocal(a: QDyn) -> QDyn {
    (-a.log()).exp()
}

/// `(x₁, x₂)`, `x₃` and `c = 1 / (1 + x₃)`.
fn parts(x: QDyn) -> (QDyn, QDyn, QDyn) {
    let x12 = x.slice(0..2);
    let x3 = x.slice(2..3).reshape(Space::Real);
    (x12, x3, reciprocal(x3 + 1.0))
}

/// `B(x) t`: `(t - c x₁₂ a, -a)` for `a = x₁₂·t`.
pub fn basis_times(x: QDyn, t: QDyn) -> QDyn {
    let (x12, _, c) = parts(x);
    let a = (x12 * t).sum();
    skoro::scope::concatenate(&[t - (c * a) * x12, -a], Space::Vector(3))
}

/// `B(x)ᵀ y`: `y₁₂ - (c e + y₃) x₁₂` for `e = x₁₂·y₁₂`.
pub fn basis_transpose(x: QDyn, y: QDyn) -> QDyn {
    let (x12, _, c) = parts(x);
    let (y12, y3) = (y.slice(0..2), y.slice(2..3).reshape(Space::Real));
    let e = (x12 * y12).sum();
    y12 - (c * e + y3) * x12
}

/// `R(x, t) = ((1 - s) x + B(x) t) / (1 + s)`, `s = |t|² / 4`.
pub fn retraction() -> Arc<Composite> {
    Composite::define(
        "sphere.retract",
        Vec::new(),
        &[Space::Vector(3), Space::Vector(2)],
        |q| {
            let (x, t) = (q[0], q[1]);
            let s = (t * t).sum() * 0.25;
            reciprocal(s + 1.0) * ((-s + 1.0) * x + basis_times(x, t))
        },
    )
    .unwrap()
}

/// `Q(x, y) = 2 B(x)ᵀ y / (1 + xᵀ y)`.
pub fn inverse() -> Arc<Composite> {
    Composite::define(
        "sphere.local",
        Vec::new(),
        &[Space::Vector(3), Space::Vector(3)],
        |q| {
            let (x, y) = (q[0], q[1]);
            let dot = (x * y).sum();
            (reciprocal(dot + 1.0) * 2.0) * basis_transpose(x, y)
        },
    )
    .unwrap()
}

/// `P(w) = w / |w|`.
pub fn projection() -> Arc<Composite> {
    Composite::define("sphere.project", Vec::new(), &[Space::Vector(3)], |q| {
        let w = q[0];
        (-((w * w).sum().log() * 0.5)).exp() * w
    })
    .unwrap()
}

/// `Λ(x, t) = -2 log(1 + |t|² / 4)`, against surface measure, which the orthonormal basis
/// normalizes: the chart is global and full.
pub fn log_volume() -> Arc<Composite> {
    Composite::define(
        "sphere.log_volume",
        Vec::new(),
        &[Space::Vector(3), Space::Vector(2)],
        |q| ((q[1] * q[1]).sum() * 0.25 + 1.0).log() * -2.0,
    )
    .unwrap()
}

pub static SPHERE: LazyLock<Arc<Embedded>> = LazyLock::new(|| {
    let (r, q, p) = (retraction(), inverse(), projection());
    Embedded::declare("sphere", Vec::new(), 3, 2, r, q, Some(p))
        .and_then(|sphere| sphere.with_volume(Volume::global(log_volume())))
        .unwrap()
});

/// The sphere's marker.
#[derive(Clone, Copy, Debug)]
pub struct Sphere;

impl Typed for Sphere {
    fn admits(space: &Space) -> bool {
        matches!(space, Space::Embedded(e) if **e == **SPHERE)
    }
}

impl Declares for Sphere {
    fn space(&self) -> Space {
        Space::Embedded(SPHERE.clone())
    }
}

/// `B(x)` itself, `3 × 2` by rows, as the mathematics writes it.
pub fn basis(x: [f64; 3]) -> [[f64; 2]; 3] {
    let c = 1.0 / (1.0 + x[2]);
    [
        [1.0 - c * x[0] * x[0], -c * x[0] * x[1]],
        [-c * x[0] * x[1], 1.0 - c * x[1] * x[1]],
        [-x[0], -x[1]],
    ]
}

/// Points of the sphere in `U`, the sphere without its south pole, one per row.
pub fn points(rows: usize) -> Vec<[f64; 3]> {
    (0..rows)
        .map(|i| {
            let (theta, phi) = (0.4 + 1.3 * i as f64, 0.7 + 0.9 * i as f64);
            let z = (theta.cos() * 0.8).clamp(-0.6, 0.95);
            let r = (1.0 - z * z).sqrt();
            [r * phi.cos(), r * phi.sin(), z]
        })
        .collect()
}

/// How the private sphere computes its exponential map near `t = 0`.
#[derive(Clone, Copy, PartialEq)]
pub enum Near {
    /// The closed form alone: `0 / 0` at `t = 0`.
    Closed,
    /// Switched to its series by a selection, the closed form still evaluated at `t = 0`.
    Selected,
    /// Switched, and the closed form's input guarded by a second selection.
    Guarded,
}

/// The squared radius below which the exponential retraction and its log-volume use their series,
/// chosen so that their closed forms above it meet the GPU's error designed for
/// (archive/wgpu-accuracy/math, "Conditions on a cutoff").
pub const EXP_CUT: f64 = 0.1;

/// The squared radius below which the logarithm's ratio `θ / r` uses its series. Its closed form
/// divides an `atan2`, which loses nothing to the GPU, and its series `1 + r²/6` holds only near
/// `r = 0`.
pub const LOG_CUT: f64 = 1e-8;

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

/// `cos r = Σ (-1)^k r^{2k} / (2k)!`, eight terms, in `r²`.
pub const COS_SERIES: [f64; 8] = {
    let mut out = [0.0; 8];
    let mut k = 0;
    while k < 8 {
        out[k as usize] = sign(k) / factorial(2 * k);
        k += 1;
    }
    out
};

/// `sin r / r = Σ (-1)^k r^{2k} / (2k+1)!`, eight terms, in `r²`.
pub const SINC_SERIES: [f64; 8] = {
    let mut out = [0.0; 8];
    let mut k = 0;
    while k < 8 {
        out[k as usize] = sign(k) / factorial(2 * k + 1);
        k += 1;
    }
    out
};

/// The Bernoulli numbers `B₂, B₄, …, B₂₀`.
const BERNOULLI: [f64; 10] = [
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
];

/// `Λ = log(sin r / r) = Σ_{n ≥ 1} (-1)^n 2^{2n-1} B_{2n} r^{2n} / (n (2n)!)`, ten terms, in `r²`
/// from its coefficient of `r²`: `-r²/6 - r⁴/180 - r⁶/2835 - …`.
pub const LOG_VOLUME_SERIES: [f64; 10] = {
    let mut out = [0.0; 10];
    let mut i = 0;
    while i < 10 {
        let n = i as u32 + 1;
        let mut power = 0.5;
        let mut j = 0;
        while j < n {
            power *= 4.0;
            j += 1;
        }
        out[i] = sign(n) * power * BERNOULLI[i] / (n as f64 * factorial(2 * n));
        i += 1;
    }
    out
};

/// `Σ cₖ xᵏ`, by Horner's rule.
pub fn horner(coefficients: &[f64], x: QDyn) -> QDyn {
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

pub fn select(index: QDyn, values: [QDyn; 2]) -> QDyn {
    apply(Arc::new(Select), &[index, values[0], values[1]])
}

/// The exponential map, `R(x, t) = cos|t| x + (sin|t| / |t|) B(x) t`.
pub fn exponential(near: Near) -> Arc<Composite> {
    let name = match near {
        Near::Closed => "sphere.exp.closed",
        Near::Selected => "sphere.exp.selected",
        Near::Guarded => "sphere.exp.guarded",
    };
    Composite::define(
        name,
        Vec::new(),
        &[Space::Vector(3), Space::Vector(2)],
        move |q| {
            let (x, t) = (q[0], q[1]);
            let ss = (t * t).sum();
            let small = (-ss + EXP_CUT).is_nonnegative();
            let one = constant(Real, 1.0).untyped();
            let input = if near == Near::Guarded {
                select(small, [ss, one])
            } else {
                ss
            };
            let n = (input.log() * 0.5).exp();
            let (cos, sinc) = (n.cos(), n.sin() * reciprocal(n));
            let (cos, sinc) = match near {
                Near::Closed => (cos, sinc),
                Near::Selected | Near::Guarded => (
                    select(small, [cos, horner(&COS_SERIES, ss)]),
                    select(small, [sinc, horner(&SINC_SERIES, ss)]),
                ),
            };
            cos * x + sinc * basis_times(x, t)
        },
    )
    .unwrap()
}

/// Its inverse, `Q(x, y) = (θ / r) B(x)ᵀ y` with `r = |B(x)ᵀ y|` and `θ = atan2(r, xᵀ y)`, guarded
/// at `r = 0` as the guarded retraction is.
pub fn logarithm() -> Arc<Composite> {
    Composite::define(
        "sphere.log",
        Vec::new(),
        &[Space::Vector(3), Space::Vector(3)],
        |q| {
            let (x, y) = (q[0], q[1]);
            let bty = basis_transpose(x, y);
            let rr = (bty * bty).sum();
            let small = (-rr + LOG_CUT).is_nonnegative();
            let one = constant(Real, 1.0).untyped();
            let r = (select(small, [rr, one]).log() * 0.5).exp();
            let theta = r.atan2((x * y).sum());
            let ratio = select(small, [theta * reciprocal(r), rr * (1.0 / 6.0) + 1.0]);
            ratio * bty
        },
    )
    .unwrap()
}

pub fn exponential_sphere(near: Near) -> Arc<Embedded> {
    let name = match near {
        Near::Closed => "sphere.exp.closed",
        Near::Selected => "sphere.exp.selected",
        Near::Guarded => "sphere.exp.guarded",
    };
    Embedded::declare(name, Vec::new(), 3, 2, exponential(near), logarithm(), None).unwrap()
}

/// The exponential chart's log-volume against surface measure, `log(sin r / r)` with
/// `r = |t|`: guarded as its retraction is, below the same cutoff, by its series
/// (`LOG_VOLUME_SERIES`), or computed as written.
pub fn exponential_log_volume(guarded: bool) -> Arc<Composite> {
    let name = if guarded {
        "sphere.exp.log_volume"
    } else {
        "sphere.exp.log_volume.closed"
    };
    Composite::define(
        name,
        Vec::new(),
        &[Space::Vector(3), Space::Vector(2)],
        move |q| {
            let rr = (q[1] * q[1]).sum();
            if !guarded {
                let r = (rr.log() * 0.5).exp();
                return (r.sin() * reciprocal(r)).log();
            }
            let small = (-rr + EXP_CUT).is_nonnegative();
            let one = constant(Real, 1.0).untyped();
            let r = (select(small, [rr, one]).log() * 0.5).exp();
            let closed = (r.sin() * reciprocal(r)).log();
            select(small, [closed, rr * horner(&LOG_VOLUME_SERIES, rr)])
        },
    )
    .unwrap()
}

/// `|t|² ≤ π²`: the ball on which the exponential chart is a diffeomorphism, closed, which
/// differs from it by a boundary of measure zero.
pub fn ball_of_radius_pi() -> Arc<Composite> {
    Composite::define(
        "sphere.exp.domain",
        Vec::new(),
        &[Space::Vector(3), Space::Vector(2)],
        |q| (-(q[1] * q[1]).sum() + std::f64::consts::PI.powi(2)).is_nonnegative(),
    )
    .unwrap()
}

/// The guarded exponential sphere with a volume: `name` declared with `volume`.
pub fn measured_exponential_sphere(name: &'static str, volume: Volume) -> Arc<Embedded> {
    Embedded::declare(
        name,
        Vec::new(),
        3,
        2,
        exponential(Near::Guarded),
        logarithm(),
        None,
    )
    .and_then(|sphere| sphere.with_volume(volume))
    .unwrap()
}

pub static EXPONENTIAL: std::sync::LazyLock<Arc<Embedded>> = std::sync::LazyLock::new(|| {
    measured_exponential_sphere(
        "sphere.exp.measured",
        Volume::within(exponential_log_volume(true), ball_of_radius_pi()),
    )
});

/// The exponential sphere's marker.
#[derive(Clone, Copy, Debug)]
pub struct Exponential;

impl Typed for Exponential {
    fn admits(space: &Space) -> bool {
        matches!(space, Space::Embedded(e) if **e == **EXPONENTIAL)
    }
}

impl Declares for Exponential {
    fn space(&self) -> Space {
        Space::Embedded(EXPONENTIAL.clone())
    }
}

/// The acceptance-test sphere under another name, declared without a volume: a type with no
/// reference measure.
pub static UNMEASURED: LazyLock<Arc<Embedded>> = LazyLock::new(|| {
    let (r, q, p) = (retraction(), inverse(), projection());
    Embedded::declare("sphere.unmeasured", Vec::new(), 3, 2, r, q, Some(p)).unwrap()
});

/// The unmeasured sphere's marker.
#[derive(Clone, Copy, Debug)]
pub struct Unmeasured;

impl Typed for Unmeasured {
    fn admits(space: &Space) -> bool {
        matches!(space, Space::Embedded(e) if **e == **UNMEASURED)
    }
}

impl Declares for Unmeasured {
    fn space(&self) -> Space {
        Space::Embedded(UNMEASURED.clone())
    }
}

/// A point of a sphere, shared by every row.
fn on_the_sphere(point: [f64; 3]) -> Option<HostValue> {
    HostValue::new(None, 3, HostData::Continuous(point.to_vec()))
}

impl IntoHostValue<Sphere> for [f64; 3] {
    fn into_host_value(self) -> Option<HostValue> {
        on_the_sphere(self)
    }
}

impl IntoHostValue<Exponential> for [f64; 3] {
    fn into_host_value(self) -> Option<HostValue> {
        on_the_sphere(self)
    }
}

impl IntoHostValue<Unmeasured> for [f64; 3] {
    fn into_host_value(self) -> Option<HostValue> {
        on_the_sphere(self)
    }
}
