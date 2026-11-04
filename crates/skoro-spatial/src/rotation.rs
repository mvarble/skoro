//! The maps of the rotations: `*` composing and acting, the methods of [`RotationMaps`] on a
//! handle, and the free functions that make a rotation.
//!
//! Each map of more than a few nodes is a composite named `skoro.spatial.<map>`, defined once; the
//! operators and methods call it, so that a wrong input is refused at the caller's line.

use std::sync::{Arc, LazyLock};

use skoro_core::scope::{concatenate, constant, project};
use skoro_core::{
    Composite, Declares, Matrix, Param, Q, QDyn, QMul, Real, Space, Vector, VectorDyn,
};

use crate::algebra::{
    self, choose, cross, dot, norm, normalize, parts, quaternion, reciprocal, sqrt,
};
use crate::series::{Coefficient, coefficient, log_ratio};
use crate::{Axis, Euler, Rotation};

/// A composite of the crate, defined once; `composites_pass_check_operation` defines every one.
pub(crate) fn composite(
    name: &'static str,
    inputs: &[Space],
    body: impl FnOnce(&[QDyn]) -> QDyn,
) -> Arc<Composite> {
    composite_with(name, Vec::new(), inputs, body)
}

/// A composite of the crate with parameters, defined once.
pub(crate) fn composite_with(
    name: &'static str,
    params: Vec<Param>,
    inputs: &[Space],
    body: impl FnOnce(&[QDyn]) -> QDyn,
) -> Arc<Composite> {
    match Composite::define(name, params, inputs, body) {
        Ok(composite) => composite,
        Err(_) => unreachable!("the crate's composites are checked by `types_are_declared`"),
    }
}

/// The space of a rotation.
pub(crate) fn rotation_space() -> Space {
    Rotation.space()
}

/// `ι(q)`, the storage of a rotation.
pub(crate) fn stored(q: QDyn) -> QDyn {
    q.include()
}

/// `P(w)`, storage projected onto the rotations.
pub(crate) fn onto(w: QDyn) -> QDyn {
    project(Rotation, w).untyped()
}

static COMPOSE: LazyLock<Arc<Composite>> = LazyLock::new(|| {
    composite(
        "skoro.spatial.compose",
        &[rotation_space(), rotation_space()],
        |q| onto(algebra::mul(stored(q[0]), stored(q[1]))),
    )
});

static INVERSE: LazyLock<Arc<Composite>> = LazyLock::new(|| {
    composite("skoro.spatial.inverse", &[rotation_space()], |q| {
        onto(algebra::conjugate(stored(q[0])))
    })
});

static BETWEEN: LazyLock<Arc<Composite>> = LazyLock::new(|| {
    composite(
        "skoro.spatial.between",
        &[rotation_space(), rotation_space()],
        |q| onto(algebra::mul(algebra::conjugate(stored(q[0])), stored(q[1]))),
    )
});

static ROTATE: LazyLock<Arc<Composite>> = LazyLock::new(|| {
    composite(
        "skoro.spatial.rotate",
        &[rotation_space(), Space::Vector(3)],
        |q| algebra::rotate(stored(q[0]), q[1]),
    )
});

static UNROTATE: LazyLock<Arc<Composite>> = LazyLock::new(|| {
    composite(
        "skoro.spatial.unrotate",
        &[rotation_space(), Space::Vector(3)],
        |q| algebra::unrotate(stored(q[0]), q[1]),
    )
});

static MATRIX: LazyLock<Arc<Composite>> = LazyLock::new(|| {
    composite("skoro.spatial.matrix", &[rotation_space()], |q| {
        algebra::matrix(stored(q[0]))
    })
});

/// `Exp(ω) = P(a, b ω)`, with `a` and `b` of `θ² = ω·ω`.
pub(crate) fn exp_stored(omega: QDyn) -> QDyn {
    let theta2 = dot(omega, omega);
    let (a, b) = (
        coefficient(Coefficient::A0, theta2),
        coefficient(Coefficient::B0, theta2),
    );
    quaternion(a, b * omega)
}

/// `Log(q) = L(n², w) u`, of the storage of a quaternion.
pub(crate) fn log_stored(q: QDyn) -> QDyn {
    let (w, u) = parts(q);
    log_ratio(dot(u, u), w) * u
}

/// `s(q) q`, the storage on the side `w ≥ 0`.
pub(crate) fn shorter(q: QDyn) -> QDyn {
    let (w, _) = parts(q);
    choose(w.is_nonnegative(), -q, q)
}

static EXP: LazyLock<Arc<Composite>> = LazyLock::new(|| {
    composite("skoro.spatial.exp", &[Space::Vector(3)], |q| {
        onto(exp_stored(q[0]))
    })
});

static LOG: LazyLock<Arc<Composite>> = LazyLock::new(|| {
    composite("skoro.spatial.log", &[rotation_space()], |q| {
        log_stored(stored(q[0]))
    })
});

static LOG_SHORTER: LazyLock<Arc<Composite>> = LazyLock::new(|| {
    composite("skoro.spatial.log_shorter", &[rotation_space()], |q| {
        log_stored(shorter(stored(q[0])))
    })
});

static ABOUT: LazyLock<Arc<Composite>> = LazyLock::new(|| {
    composite(
        "skoro.spatial.about",
        &[Space::Vector(3), Space::Real],
        |q| {
            let half = q[1] * 0.5;
            onto(quaternion(half.cos(), half.sin() * normalize(q[0])))
        },
    )
});

/// `(a, θ)` of `Log`, as a `Vector(4)`: `θ = 2 atan2(n, w)` and `a = u / n`, with the axis
/// `(1, 0, 0)` and the angle `2 atan2(0, w)` selected where `n² = 0`.
static AXIS_ANGLE: LazyLock<Arc<Composite>> = LazyLock::new(|| {
    composite("skoro.spatial.axis_angle", &[rotation_space()], |q| {
        let (w, u) = parts(stored(q[0]));
        let n2 = dot(u, u);
        let identity = (n2 * -1.0).is_nonnegative();
        let n = sqrt(choose(identity, n2, algebra::one()));
        let fixed = constant(Vector::<3>, [1.0, 0.0, 0.0]).untyped();
        let axis = choose(identity, u * reciprocal(n), fixed);
        let angle = choose(identity, n, algebra::zero()).atan2(w) * 2.0;
        concatenate(&[axis, angle], Space::Vector(4))
    })
});

static FROM_TWO_VECTORS: LazyLock<Arc<Composite>> = LazyLock::new(|| {
    composite(
        "skoro.spatial.from_two_vectors",
        &[Space::Vector(3), Space::Vector(3)],
        |q| {
            let (a, b) = (q[0], q[1]);
            onto(quaternion(norm(a) * norm(b) + dot(a, b), cross(a, b)))
        },
    )
});

/// `sign⟨q, r⟩ q`, by selection, projected.
static ALIGN: LazyLock<Arc<Composite>> = LazyLock::new(|| {
    composite(
        "skoro.spatial.align",
        &[rotation_space(), rotation_space()],
        |q| {
            let (z, r) = (stored(q[0]), stored(q[1]));
            onto(choose(dot(z, r).is_nonnegative(), -z, z))
        },
    )
});

/// The four arguments `c_m(X)` of a stored `Matrix(3, 3)`, each a `Vector(4)` linear in `X`.
fn arguments(x: QDyn) -> [QDyn; 4] {
    // X_ij, counted from 1 as the mathematics counts them, is stored at (i - 1) + 3 (j - 1).
    let at = |i: usize, j: usize| algebra::entry(x, (i - 1) + 3 * (j - 1));
    let (x11, x22, x33) = (at(1, 1), at(2, 2), at(3, 3));
    let (x12, x13, x21, x23, x31, x32) =
        (at(1, 2), at(1, 3), at(2, 1), at(2, 3), at(3, 1), at(3, 2));
    let four = |a: QDyn, b: QDyn, c: QDyn, d: QDyn| concatenate(&[a, b, c, d], Space::Vector(4));
    [
        four(x11 + x22 + x33 + 1.0, x32 - x23, x13 - x31, x21 - x12),
        four(x32 - x23, x11 - x22 - x33 + 1.0, x12 + x21, x13 + x31),
        four(x13 - x31, x12 + x21, x22 - x11 - x33 + 1.0, x23 + x32),
        four(x21 - x12, x13 + x31, x23 + x32, x33 - x11 - x22 + 1.0),
    ]
}

/// `P(c_m(X))` with `m` the index of the largest of the diagonal entries `c_m(X)_m`, by a
/// tournament of selections, the argument chosen before it is projected.
static FROM_MATRIX: LazyLock<Arc<Composite>> = LazyLock::new(|| {
    composite("skoro.spatial.from_matrix", &[Space::Matrix(3, 3)], |q| {
        let c = arguments(q[0]);
        let d = |m: usize| algebra::entry(c[m], m);
        let first = (d(1) - d(0)).is_nonnegative();
        let second = (d(3) - d(2)).is_nonnegative();
        let (a, da) = (choose(first, c[0], c[1]), choose(first, d(0), d(1)));
        let (b, db) = (choose(second, c[2], c[3]), choose(second, d(2), d(3)));
        onto(choose((db - da).is_nonnegative(), a, b))
    })
});

/// `q₁ ⊗ Exp(t Log(q̄₁ ⊗ q₂))` in storage, or with the rotation vector of the shorter turn.
pub(crate) fn slerp_stored(a: QDyn, b: QDyn, t: QDyn, shorter_turn: bool) -> QDyn {
    let r = algebra::mul(algebra::conjugate(a), b);
    let r = if shorter_turn { shorter(r) } else { r };
    algebra::mul(a, exp_stored(t * log_stored(r)))
}

static SLERP: LazyLock<Arc<Composite>> = LazyLock::new(|| {
    composite(
        "skoro.spatial.slerp",
        &[rotation_space(), rotation_space(), Space::Real],
        |q| onto(slerp_stored(stored(q[0]), stored(q[1]), q[2], false)),
    )
});

static SLERP_SHORTER: LazyLock<Arc<Composite>> = LazyLock::new(|| {
    composite(
        "skoro.spatial.slerp_shorter",
        &[rotation_space(), rotation_space(), Space::Real],
        |q| onto(slerp_stored(stored(q[0]), stored(q[1]), q[2], true)),
    )
});

/// `2 atan2(|vec r|, |w(r)|)` with `r = q̄₁ ⊗ q₂`, the absolute value a selection.
static ANGLE: LazyLock<Arc<Composite>> = LazyLock::new(|| {
    composite(
        "skoro.spatial.angle",
        &[rotation_space(), rotation_space()],
        |q| {
            let r = algebra::mul(algebra::conjugate(stored(q[0])), stored(q[1]));
            let (w, u) = parts(r);
            let magnitude = choose(w.is_nonnegative(), -w, w);
            norm(u).atan2(magnitude) * 2.0
        },
    )
});

/// `q₁ * q₂ = P(q₁ ⊗ q₂)`.
impl QMul for Rotation {
    type Output = Rotation;
    #[track_caller]
    fn mul(lhs: Q<Rotation>, rhs: Q<Rotation>) -> Q<Rotation> {
        COMPOSE.call(&[lhs.untyped(), rhs.untyped()]).narrow()
    }
}

/// `q * x = π(q) x`.
impl QMul<Vector<3>> for Rotation {
    type Output = Vector<3>;
    #[track_caller]
    fn mul(lhs: Q<Rotation>, rhs: Q<Vector<3>>) -> Q<Vector<3>> {
        ROTATE.call(&[lhs.untyped(), rhs.untyped()]).narrow()
    }
}

/// `q * x = π(q) x`, refusing a vector not of length 3 (`SpaceMismatch`).
impl QMul<VectorDyn> for Rotation {
    type Output = VectorDyn;
    #[track_caller]
    fn mul(lhs: Q<Rotation>, rhs: Q<VectorDyn>) -> Q<VectorDyn> {
        ROTATE.call(&[lhs.untyped(), rhs.untyped()]).narrow()
    }
}

/// `q * X = π(q) X`: each column rotated.
impl<const N: usize> QMul<Matrix<3, N>> for Rotation {
    type Output = Matrix<3, N>;
    #[track_caller]
    fn mul(lhs: Q<Rotation>, rhs: Q<Matrix<3, N>>) -> Q<Matrix<3, N>> {
        lhs.matrix().matmul(rhs)
    }
}

/// The maps of a rotation, as methods of its handle.
pub trait RotationMaps: Sized {
    /// `P(q̄)`.
    fn inverse(self) -> Q<Rotation>;
    /// `P(q̄₁ ⊗ q₂)`: the rotation from `self` to `other`, in the frame of `self`.
    fn between(self, other: Q<Rotation>) -> Q<Rotation>;
    /// `π(q)ᵀ x`: a vector of the outer frame in the frame of `self`.
    fn unrotate(self, x: Q<Vector<3>>) -> Q<Vector<3>>;
    /// `π(q)`, the rotation matrix.
    fn matrix(self) -> Q<Matrix<3, 3>>;
    /// The adjoint, `π(q)`, carrying a perturbation in the frame of `self` to the outer frame:
    /// `q ⊕ t = (1 ⊕ π(q) t) q`.
    fn adjoint(self) -> Q<Matrix<3, 3>>;
    /// `Log(q) = 2 atan2(n, w) u / n`: the inverse of `exp`, of length in [0, 2π). `Log(-q) ≠
    /// Log(q)`. Undefined at `-1`: non-finite exactly there, and near it of length near 2π with
    /// its direction ill-conditioned, set by rounding within rounding of `-1`.
    fn log(self) -> Q<Vector<3>>;
    /// The rotation vector of the shorter turn, `Log(s(q) q)` with `s(q)` the sign of `w`: the
    /// same at `±q`, of length in [0, π], and a discretization at half-turns.
    fn log_shorter(self) -> Q<Vector<3>>;
    /// The axis and the angle of `log`, the angle in [0, 2π); the axis `(1, 0, 0)` at the
    /// identity, where it is a discretization.
    fn axis_angle(self) -> (Q<Vector<3>>, Q<Real>);
    /// `sign⟨q, r⟩ q`: the point over this rotation on the side of `reference`, a discretization
    /// where the two are orthogonal.
    fn aligned_with(self, reference: Q<Rotation>) -> Q<Rotation>;
    /// The angles of `convention`, in the order of its axes: `α` and `γ` in (-π, π], `β` in
    /// [0, π] for a proper sequence and in [-π/2, π/2] for a Tait–Bryan one. `α` and `γ` are cut
    /// where they are π: their values jump there by 2π, and their derivatives do not, so angles
    /// are compared by [`angle_residual`](crate::angle_residual) rather than subtracted. At gimbal
    /// lock, `γ` is 0, a discretization; near it, `α` and `γ` alone are ill-conditioned, their
    /// derivatives growing like `1 / sin(β/2)`, while `β` and the combination of `α` and `γ` the
    /// lock determines stay accurate.
    fn euler(self, convention: Euler) -> Q<Vector<3>>;
    /// The angle between two rotations, `|rv(q̄₁ ⊗ q₂)| ∈ [0, π]`: the length of the shorter turn
    /// from one to the other. It has no derivative where it is 0, where its passes are
    /// non-finite, and where it is π, a discretization whose passes are those of the side
    /// selected.
    fn angle_to(self, other: Q<Rotation>) -> Q<Real>;
    /// `q₁ ⊗ Exp(t Log(q̄₁ ⊗ q₂))`: the geodesic of the unit quaternions from `self` at `t = 0` to
    /// `other` at `t = 1`, by the arc that avoids `-self`.
    fn slerp(self, other: Q<Rotation>, t: Q<Real>) -> Q<Rotation>;
    /// The same along the shorter of the two turns from one rotation to the other: a
    /// discretization where they are equally short.
    fn slerp_shorter(self, other: Q<Rotation>, t: Q<Real>) -> Q<Rotation>;
}

impl RotationMaps for Q<Rotation> {
    #[track_caller]
    fn inverse(self) -> Q<Rotation> {
        INVERSE.call(&[self.untyped()]).narrow()
    }

    #[track_caller]
    fn between(self, other: Q<Rotation>) -> Q<Rotation> {
        BETWEEN.call(&[self.untyped(), other.untyped()]).narrow()
    }

    #[track_caller]
    fn unrotate(self, x: Q<Vector<3>>) -> Q<Vector<3>> {
        UNROTATE.call(&[self.untyped(), x.untyped()]).narrow()
    }

    #[track_caller]
    fn matrix(self) -> Q<Matrix<3, 3>> {
        MATRIX.call(&[self.untyped()]).narrow()
    }

    #[track_caller]
    fn adjoint(self) -> Q<Matrix<3, 3>> {
        self.matrix()
    }

    #[track_caller]
    fn log(self) -> Q<Vector<3>> {
        LOG.call(&[self.untyped()]).narrow()
    }

    #[track_caller]
    fn log_shorter(self) -> Q<Vector<3>> {
        LOG_SHORTER.call(&[self.untyped()]).narrow()
    }

    #[track_caller]
    fn axis_angle(self) -> (Q<Vector<3>>, Q<Real>) {
        let both: Q<Vector<4>> = AXIS_ANGLE.call(&[self.untyped()]).narrow();
        (
            both.slice(0..3).reshape(Vector::<3>),
            both.slice(3..4).reshape(Real),
        )
    }

    #[track_caller]
    fn aligned_with(self, reference: Q<Rotation>) -> Q<Rotation> {
        ALIGN.call(&[self.untyped(), reference.untyped()]).narrow()
    }

    #[track_caller]
    fn euler(self, convention: Euler) -> Q<Vector<3>> {
        crate::euler::euler(self, convention)
    }

    #[track_caller]
    fn angle_to(self, other: Q<Rotation>) -> Q<Real> {
        ANGLE.call(&[self.untyped(), other.untyped()]).narrow()
    }

    #[track_caller]
    fn slerp(self, other: Q<Rotation>, t: Q<Real>) -> Q<Rotation> {
        SLERP
            .call(&[self.untyped(), other.untyped(), t.untyped()])
            .narrow()
    }

    #[track_caller]
    fn slerp_shorter(self, other: Q<Rotation>, t: Q<Real>) -> Q<Rotation> {
        SLERP_SHORTER
            .call(&[self.untyped(), other.untyped(), t.untyped()])
            .narrow()
    }
}

/// The identity rotation, `(1, 0, 0, 0)`.
#[track_caller]
pub fn identity() -> Q<Rotation> {
    constant(Rotation, [1.0, 0.0, 0.0, 0.0])
}

/// `Exp(ω) = (cos(θ/2), sin(θ/2) ω / θ)`, `θ = |ω|`: the rotation about `ω` by its length.
#[track_caller]
pub fn exp(omega: Q<Vector<3>>) -> Q<Rotation> {
    EXP.call(&[omega.untyped()]).narrow()
}

/// The rotation by `angle` about `axis ≠ 0`: `(cos(θ/2), sin(θ/2) a / |a|)`.
#[track_caller]
pub fn about(axis: Q<Vector<3>>, angle: Q<Real>) -> Q<Rotation> {
    ABOUT.call(&[axis.untyped(), angle.untyped()]).narrow()
}

/// `q_e(θ) = (cos(θ/2), sin(θ/2) e)`, the rotation by `angle` about a coordinate axis.
#[track_caller]
pub fn elementary(axis: Axis, angle: Q<Real>) -> Q<Rotation> {
    about(constant(Vector::<3>, axis.unit()), angle)
}

/// The rotation of least angle carrying the direction of `a` to that of `b`,
/// `P(|a| |b| + a·b, a × b)`. Undefined where `b` is opposite to `a`, or either is zero:
/// non-finite where the argument vanishes exactly, and ill-conditioned near it, its error the
/// rounding over the angle by which `b` falls short of `-a`, so that at directions opposite only
/// up to rounding it is finite and meaningless.
#[track_caller]
pub fn from_two_vectors(a: Q<Vector<3>>, b: Q<Vector<3>>) -> Q<Rotation> {
    FROM_TWO_VECTORS.call(&[a.untyped(), b.untyped()]).narrow()
}

/// The rotation of the Euler angles `angles` of `convention`, listed in the order of its axes:
/// `q_i(α) ⊗ q_j(β) ⊗ q_k(γ)` for an intrinsic sequence `(i, j, k)`, `q_k(γ) ⊗ q_j(β) ⊗ q_i(α)`
/// for an extrinsic one.
#[track_caller]
pub fn from_euler(angles: Q<Vector<3>>, convention: Euler) -> Q<Rotation> {
    crate::euler::from_euler(angles, convention)
}

/// The unit quaternion over a rotation matrix, `P(c_m(X))` with `m` the index of the largest of
/// the four candidates' diagonal entries: the point over `X` whose `m`-th coordinate is positive.
/// A discretization where the choice of `m` changes; a matrix near a rotation gives a nearby unit
/// quaternion.
#[track_caller]
pub fn from_matrix(m: Q<Matrix<3, 3>>) -> Q<Rotation> {
    FROM_MATRIX.call(&[m.untyped()]).narrow()
}
