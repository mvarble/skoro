//! The maps of the rotations: `*` composing and acting, the methods of [`RotationMaps`] on a
//! handle, and the free functions that make a rotation.
//!
//! Each map of more than a few nodes is a composite named `skoro.spatial.<map>`, defined once; the
//! operators and methods call it, so that a wrong input is refused at the caller's line.

use std::sync::{Arc, LazyLock};

use skoro_core::scope::{concatenate, constant, project};
use skoro_core::{Composite, Declares, Matrix, Q, QDyn, QMul, Real, Space, Vector, VectorDyn};

use crate::algebra::{
    self, choose, cross, dot, norm, normalize, parts, quaternion, reciprocal, sqrt,
};
use crate::series::{Coefficient, coefficient, log_ratio};
use crate::{Axis, Rotation};

/// A composite of the crate, defined once; `composites_pass_check_operation` defines every one.
pub(crate) fn composite(
    name: &'static str,
    inputs: &[Space],
    body: impl FnOnce(&[QDyn]) -> QDyn,
) -> Arc<Composite> {
    match Composite::define(name, Vec::new(), inputs, body) {
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
    /// `Log(q) = 2 atan2(n, w) u / n`: the inverse of `exp`, of length in [0, 2π); non-finite at
    /// `-1`. `Log(-q) ≠ Log(q)`.
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
/// `P(|a| |b| + a·b, a × b)`; non-finite where `b` is opposite to `a`, or either is zero.
#[track_caller]
pub fn from_two_vectors(a: Q<Vector<3>>, b: Q<Vector<3>>) -> Q<Rotation> {
    FROM_TWO_VECTORS.call(&[a.untyped(), b.untyped()]).narrow()
}
