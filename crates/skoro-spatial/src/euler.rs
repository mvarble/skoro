//! Euler angles: the conventions, the rotation they give, and the angles of a rotation.
//!
//! An intrinsic sequence `(i, j, k)` gives `q = q_i(α) ⊗ q_j(β) ⊗ q_k(γ)`; the extrinsic one, about
//! the fixed axes, is the intrinsic rotation of the reversed sequence with the angles reversed.
//! Both directions are composites per intrinsic sequence, `skoro.spatial.from_euler` and
//! `skoro.spatial.euler` with the sequence's name as their parameter.

use std::f64::consts::PI;
use std::sync::{Arc, LazyLock};

use skoro_core::scope::concatenate;
use skoro_core::{Composite, Param, Q, QDyn, Real, Space, Typed, Vector, VectorDyn};

use crate::Rotation;
use crate::algebra::{choose, entry, mul, one, quaternion, real, sqrt, zero};
use crate::rotation::{composite_with, onto, rotation_space, stored};

/// A coordinate axis.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum Axis {
    X,
    Y,
    Z,
}

impl Axis {
    /// Its unit vector.
    pub fn unit(self) -> [f64; 3] {
        match self {
            Axis::X => [1.0, 0.0, 0.0],
            Axis::Y => [0.0, 1.0, 0.0],
            Axis::Z => [0.0, 0.0, 1.0],
        }
    }

    /// `0`, `1` or `2`.
    pub(crate) fn index(self) -> usize {
        match self {
            Axis::X => 0,
            Axis::Y => 1,
            Axis::Z => 2,
        }
    }
}

/// The twelve sequences of axes, no two neighbours equal: six of Tait–Bryan type, three distinct
/// axes, and six proper, the first axis again last.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum Sequence {
    Xyz,
    Xzy,
    Yxz,
    Yzx,
    Zxy,
    Zyx,
    Xyx,
    Xzx,
    Yxy,
    Yzy,
    Zxz,
    Zyz,
}

impl Sequence {
    /// Every sequence.
    pub const ALL: [Sequence; 12] = [
        Sequence::Xyz,
        Sequence::Xzy,
        Sequence::Yxz,
        Sequence::Yzx,
        Sequence::Zxy,
        Sequence::Zyx,
        Sequence::Xyx,
        Sequence::Xzx,
        Sequence::Yxy,
        Sequence::Yzy,
        Sequence::Zxz,
        Sequence::Zyz,
    ];

    /// The axes, in the order the angles are listed.
    pub fn axes(self) -> [Axis; 3] {
        use Axis::{X, Y, Z};
        match self {
            Sequence::Xyz => [X, Y, Z],
            Sequence::Xzy => [X, Z, Y],
            Sequence::Yxz => [Y, X, Z],
            Sequence::Yzx => [Y, Z, X],
            Sequence::Zxy => [Z, X, Y],
            Sequence::Zyx => [Z, Y, X],
            Sequence::Xyx => [X, Y, X],
            Sequence::Xzx => [X, Z, X],
            Sequence::Yxy => [Y, X, Y],
            Sequence::Yzy => [Y, Z, Y],
            Sequence::Zxz => [Z, X, Z],
            Sequence::Zyz => [Z, Y, Z],
        }
    }

    /// Whether the last axis is the first.
    pub fn is_proper(self) -> bool {
        let [i, _, k] = self.axes();
        i == k
    }

    /// The same axes in the reverse order.
    pub fn reversed(self) -> Sequence {
        let [i, j, k] = self.axes();
        let reversed = [k, j, i];
        Sequence::ALL
            .into_iter()
            .find(|s| s.axes() == reversed)
            .unwrap_or(self)
    }

    /// Its name, as `"zyx"`.
    pub fn name(self) -> &'static str {
        match self {
            Sequence::Xyz => "xyz",
            Sequence::Xzy => "xzy",
            Sequence::Yxz => "yxz",
            Sequence::Yzx => "yzx",
            Sequence::Zxy => "zxy",
            Sequence::Zyx => "zyx",
            Sequence::Xyx => "xyx",
            Sequence::Xzx => "xzx",
            Sequence::Yxy => "yxy",
            Sequence::Yzy => "yzy",
            Sequence::Zxz => "zxz",
            Sequence::Zyz => "zyz",
        }
    }

    fn position(self) -> usize {
        Sequence::ALL
            .iter()
            .position(|s| *s == self)
            .unwrap_or_default()
    }

    /// `i`, `j`, `l` the axis other than `i` and `j`, and `ε = ±1` with `e_i × e_j = ε e_l`.
    pub(crate) fn frame(self) -> (usize, usize, usize, f64) {
        let [i, j, _] = self.axes();
        let (i, j) = (i.index(), j.index());
        let epsilon = if (j + 3 - i) % 3 == 1 { 1.0 } else { -1.0 };
        (i, j, 3 - i - j, epsilon)
    }
}

/// About the axes as the previous rotations left them, or about the fixed axes.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum Frame {
    Intrinsic,
    Extrinsic,
}

/// A convention of Euler angles: a sequence of axes, and the frame each turn is about.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct Euler {
    pub sequence: Sequence,
    pub frame: Frame,
}

impl Euler {
    /// Yaw, pitch and roll `(ψ, ϑ, φ)`, intrinsic about `(z, y, x)`: the aerospace convention.
    pub const YAW_PITCH_ROLL: Euler = Euler {
        sequence: Sequence::Zyx,
        frame: Frame::Intrinsic,
    };

    /// The axes, in the order the angles are listed.
    pub fn axes(self) -> [Axis; 3] {
        self.sequence.axes()
    }

    /// The intrinsic sequence whose composites compute this convention, and whether its angles
    /// are listed in reverse.
    pub(crate) fn intrinsic(self) -> (Sequence, bool) {
        match self.frame {
            Frame::Intrinsic => (self.sequence, false),
            Frame::Extrinsic => (self.sequence.reversed(), true),
        }
    }
}

/// The threshold `c` of the lock: `β` is taken at the lock where the smaller squared norm is at
/// most this fraction of the quaternion's. On such a row the convention `γ = 0` reproduces the
/// rotation to within `2 asin √c = 2·10⁻¹⁷`, below double precision's rounding of a unit
/// quaternion, `2⁻⁵³ ≈ 1.1·10⁻¹⁶`; and `c` times a squared norm of at most 2 stays a normal number
/// in single precision, so the band is the same at both precisions. The threshold decides
/// finiteness, not the round trip: a row at the lock only up to rounding is read by the general
/// formulas, ill-conditioned in `α` and `γ` alone, and still reproduces its rotation.
pub(crate) const LOCK: f64 = 1e-34;

/// `q_e(θ)` in storage, `(cos(θ/2), sin(θ/2) e)`.
fn elementary(axis: Axis, angle: QDyn) -> QDyn {
    let half = angle * 0.5;
    let (c, s) = (half.cos(), half.sin());
    let mut u = [zero(), zero(), zero()];
    u[axis.index()] = s;
    quaternion(c, concatenate(&u, Space::Vector(3)))
}

/// `θ` brought to `(-π, π]` as `atan2(sin θ, cos θ)`.
fn wrap(angle: QDyn) -> QDyn {
    angle.sin().atan2(angle.cos())
}

/// The storage of `q_i(α) ⊗ q_j(β) ⊗ q_k(γ)` for the intrinsic `sequence`.
fn from_euler_stored(sequence: Sequence, angles: QDyn) -> QDyn {
    let [i, j, k] = sequence.axes();
    let product = mul(
        elementary(i, entry(angles, 0)),
        elementary(j, entry(angles, 1)),
    );
    mul(product, elementary(k, entry(angles, 2)))
}

/// The angles `(α, β, γ)` of the intrinsic `sequence` of the storage of a quaternion: the statement
/// euler, read off `q` for a proper sequence and off its rearrangement for a Tait–Bryan one.
fn euler_stored(sequence: Sequence, q: QDyn) -> QDyn {
    let (i, j, l, epsilon) = sequence.frame();
    let at = |axis: usize| entry(q, 1 + axis);
    let w0 = entry(q, 0);
    let (w, ui, uj, ul) = if sequence.is_proper() {
        (w0, at(i), at(j), at(l))
    } else {
        // (w - u_j, u_i - ε u_k, u_j + w, u_k + ε u_i), with k = l.
        (
            w0 - at(j),
            at(i) - at(l) * epsilon,
            at(j) + w0,
            at(l) + at(i) * epsilon,
        )
    };
    let (na, nb) = (w * w + ui * ui, uj * uj + ul * ul);
    let total = na + nb;
    // β at 0 for a proper sequence, -π/2 for a Tait–Bryan one, where (u_j, u_l) vanishes; at π or
    // π/2 where (w, u_i) does.
    let low = (total * LOCK - nb).is_nonnegative();
    let high = (total * LOCK - na).is_nonnegative();
    let root_a = sqrt(choose(high, na, one()));
    let root_b = sqrt(choose(low, nb, one()));
    let beta = choose(
        low,
        choose(high, root_b.atan2(root_a) * 2.0, real(PI)),
        zero(),
    );
    let sigma = ui.atan2(choose(high, w, one()));
    let delta = (ul * epsilon).atan2(choose(low, uj, one()));
    let alpha = choose(low, choose(high, sigma + delta, delta * 2.0), sigma * 2.0);
    let gamma = choose(low, choose(high, sigma - delta, zero()), zero());
    let (beta, gamma) = if sequence.is_proper() {
        (beta, gamma)
    } else {
        (beta - PI / 2.0, gamma * -epsilon)
    };
    concatenate(&[wrap(alpha), beta, wrap(gamma)], Space::Vector(3))
}

fn params(sequence: Sequence) -> Vec<Param> {
    vec![Param::Text(sequence.name().into())]
}

static FROM_EULER: LazyLock<Vec<Arc<Composite>>> = LazyLock::new(|| {
    Sequence::ALL
        .into_iter()
        .map(|sequence| {
            composite_with(
                "skoro.spatial.from_euler",
                params(sequence),
                &[Space::Vector(3)],
                |q| onto(from_euler_stored(sequence, q[0])),
            )
        })
        .collect()
});

static EULER: LazyLock<Vec<Arc<Composite>>> = LazyLock::new(|| {
    Sequence::ALL
        .into_iter()
        .map(|sequence| {
            composite_with(
                "skoro.spatial.euler",
                params(sequence),
                &[rotation_space()],
                |q| euler_stored(sequence, stored(q[0])),
            )
        })
        .collect()
});

/// `(c, b, a)` of `(a, b, c)`.
fn reverse(angles: QDyn) -> QDyn {
    concatenate(
        &[angles.slice(2..3), angles.slice(1..2), angles.slice(0..1)],
        Space::Vector(3),
    )
}

/// The rotation of the angles of `convention`.
#[track_caller]
pub(crate) fn from_euler(angles: Q<Vector<3>>, convention: Euler) -> Q<Rotation> {
    let (sequence, reversed) = convention.intrinsic();
    let angles = angles.untyped();
    let angles = if reversed { reverse(angles) } else { angles };
    FROM_EULER[sequence.position()].call(&[angles]).narrow()
}

/// The angles of `convention` of a rotation, `α` and `γ` in (-π, π].
#[track_caller]
pub(crate) fn euler(q: Q<Rotation>, convention: Euler) -> Q<Vector<3>> {
    let (sequence, reversed) = convention.intrinsic();
    let angles = EULER[sequence.position()].call(&[q.untyped()]);
    let angles = if reversed { reverse(angles) } else { angles };
    angles.narrow()
}

/// The residual of angles on the circle, elementwise: `a ⊖ b = atan2(sin(a - b), cos(a - b))`, in
/// (-π, π]. It is cut where `a - b` is an odd multiple of π, as far as an angle can be from the one
/// it is compared with, and elsewhere has the derivatives of `a - b`, across the wrap of either
/// angle too. `b + angle_residual(a, b)` is `a` brought within a half-turn of `b`, continuous along
/// any path that stays within a half-turn of it.
#[track_caller]
pub fn angle_residual<M: Angles>(a: Q<M>, b: Q<M>) -> Q<M> {
    let d = a.untyped() - b.untyped();
    d.sin().atan2(d.cos()).narrow()
}

/// The markers whose values may be angles: `Real`, `Vector<N>` and `VectorDyn`.
pub trait Angles: Typed + sealed::Sealed {}

mod sealed {
    pub trait Sealed {}
}

impl sealed::Sealed for Real {}
impl Angles for Real {}
impl<const N: usize> sealed::Sealed for Vector<N> {}
impl<const N: usize> Angles for Vector<N> {}
impl sealed::Sealed for VectorDyn {}
impl Angles for VectorDyn {}

/// The angles of the intrinsic `sequence` of `q`, on the host: the composite's formulas with
/// branches where it selects.
pub(crate) fn euler_on_host(sequence: Sequence, q: [f64; 4]) -> [f64; 3] {
    let (i, j, l, epsilon) = sequence.frame();
    let at = |axis: usize| q[1 + axis];
    let (w, ui, uj, ul) = if sequence.is_proper() {
        (q[0], at(i), at(j), at(l))
    } else {
        (
            q[0] - at(j),
            at(i) - epsilon * at(l),
            at(j) + q[0],
            at(l) + epsilon * at(i),
        )
    };
    let (na, nb) = (w * w + ui * ui, uj * uj + ul * ul);
    let total = na + nb;
    let (low, high) = (nb <= LOCK * total, na <= LOCK * total);
    let wrap = |a: f64| a.sin().atan2(a.cos());
    let (alpha, beta, gamma) = if low {
        (2.0 * ui.atan2(w), 0.0, 0.0)
    } else if high {
        (2.0 * (epsilon * ul).atan2(uj), PI, 0.0)
    } else {
        let (sigma, delta) = (ui.atan2(w), (epsilon * ul).atan2(uj));
        (
            sigma + delta,
            2.0 * nb.sqrt().atan2(na.sqrt()),
            sigma - delta,
        )
    };
    if sequence.is_proper() {
        [wrap(alpha), beta, wrap(gamma)]
    } else {
        [wrap(alpha), beta - PI / 2.0, wrap(-epsilon * gamma)]
    }
}

/// `q_i(α) ⊗ q_j(β) ⊗ q_k(γ)` of the intrinsic `sequence`, on the host.
pub(crate) fn from_euler_on_host(sequence: Sequence, angles: [f64; 3]) -> [f64; 4] {
    let elementary = |axis: Axis, angle: f64| {
        let (s, c) = (0.5 * angle).sin_cos();
        let mut q = [c, 0.0, 0.0, 0.0];
        q[1 + axis.index()] = s;
        q
    };
    let [i, j, k] = sequence.axes();
    let product = crate::host::compose(elementary(i, angles[0]), elementary(j, angles[1]));
    crate::host::compose(product, elementary(k, angles[2]))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sequences_know_their_frames() {
        assert_eq!(Sequence::Xyz.frame(), (0, 1, 2, 1.0));
        assert_eq!(Sequence::Zyx.frame(), (2, 1, 0, -1.0));
        assert_eq!(Sequence::Zxz.frame(), (2, 0, 1, 1.0));
        assert_eq!(Sequence::Yxy.frame(), (1, 0, 2, -1.0));
        for s in Sequence::ALL {
            assert_eq!(s.reversed().reversed(), s);
            assert_eq!(s.is_proper(), s.reversed() == s);
            let [i, j, k] = s.axes();
            assert!(i != j && j != k);
        }
    }
}
