//! The maps of the rotations: `*` composing and acting, the methods of [`RotationMaps`] on a
//! handle, and the free functions that make a rotation.
//!
//! Each map of more than a few nodes is a composite named `skoro.spatial.<map>`, defined once; the
//! operators and methods call it, so that a wrong input is refused at the caller's line.

use std::sync::{Arc, LazyLock};

use skoro_core::scope::{constant, project};
use skoro_core::{Composite, Declares, Matrix, Q, QDyn, QMul, Space, Vector, VectorDyn};

use crate::Rotation;
use crate::algebra::{self, choose, dot, parts, quaternion};
use crate::series::{Coefficient, coefficient, log_ratio};

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
