//! The maps of the extended poses `(q, v, p)`, with the velocity carried as the position is.
//! Tangent coordinates are `(φ, η, ρ)`, rotation first, in the frame of the extended pose; its
//! action on points is that of its pose.

use std::sync::{Arc, LazyLock};

use skoro_core::scope::{constant, project};
use skoro_core::{Composite, Declares, Matrix, Q, QDyn, QMul, Real, Space, Vector};

use crate::pose::{
    adjoint_stored, between_stored, compose_stored, exp_motion_stored, interpolate_stored,
    inverse_stored, jl_inv_motion_stored, jl_motion_stored, join, log_motion_stored, matrix_stored,
};
use crate::rotation::composite;
use crate::{ExtendedPose, Pose, Rotation};

fn extended_space() -> Space {
    ExtendedPose.space()
}

/// `P` onto the extended poses.
fn onto_extended(w: QDyn) -> QDyn {
    project(ExtendedPose, w).untyped()
}

static COMPOSE: LazyLock<Arc<Composite>> = LazyLock::new(|| {
    composite(
        "skoro.spatial.extended_pose.compose",
        &[extended_space(), extended_space()],
        |q| onto_extended(compose_stored(q[0].include(), q[1].include(), 2)),
    )
});

static INVERSE: LazyLock<Arc<Composite>> = LazyLock::new(|| {
    composite(
        "skoro.spatial.extended_pose.inverse",
        &[extended_space()],
        |q| onto_extended(inverse_stored(q[0].include(), 2)),
    )
});

static BETWEEN: LazyLock<Arc<Composite>> = LazyLock::new(|| {
    composite(
        "skoro.spatial.extended_pose.between",
        &[extended_space(), extended_space()],
        |q| onto_extended(between_stored(q[0].include(), q[1].include(), 2)),
    )
});

static MATRIX: LazyLock<Arc<Composite>> = LazyLock::new(|| {
    composite(
        "skoro.spatial.extended_pose.matrix",
        &[extended_space()],
        |q| matrix_stored(q[0].include(), 2),
    )
});

static ADJOINT: LazyLock<Arc<Composite>> = LazyLock::new(|| {
    composite(
        "skoro.spatial.extended_pose.adjoint",
        &[extended_space()],
        |q| adjoint_stored(q[0].include(), 2),
    )
});

static EXP: LazyLock<Arc<Composite>> = LazyLock::new(|| {
    composite(
        "skoro.spatial.extended_pose.exp",
        &[Space::Vector(9)],
        |q| onto_extended(exp_motion_stored(q[0], 2)),
    )
});

static LOG: LazyLock<Arc<Composite>> = LazyLock::new(|| {
    composite(
        "skoro.spatial.extended_pose.log",
        &[extended_space()],
        |q| log_motion_stored(q[0].include(), 2, false),
    )
});

static LOG_SHORTER: LazyLock<Arc<Composite>> = LazyLock::new(|| {
    composite(
        "skoro.spatial.extended_pose.log_shorter",
        &[extended_space()],
        |q| log_motion_stored(q[0].include(), 2, true),
    )
});

static JL: LazyLock<Arc<Composite>> = LazyLock::new(|| {
    composite("skoro.spatial.extended_pose.jl", &[Space::Vector(9)], |q| {
        jl_motion_stored(q[0], 2)
    })
});

static JL_INV: LazyLock<Arc<Composite>> = LazyLock::new(|| {
    composite(
        "skoro.spatial.extended_pose.jl_inv",
        &[Space::Vector(9)],
        |q| jl_inv_motion_stored(q[0], 2),
    )
});

static INTERPOLATE: LazyLock<Arc<Composite>> = LazyLock::new(|| {
    composite(
        "skoro.spatial.extended_pose.interpolate",
        &[extended_space(), extended_space(), Space::Real],
        |q| {
            onto_extended(interpolate_stored(
                q[0].include(),
                q[1].include(),
                q[2],
                2,
                false,
            ))
        },
    )
});

static INTERPOLATE_SHORTER: LazyLock<Arc<Composite>> = LazyLock::new(|| {
    composite(
        "skoro.spatial.extended_pose.interpolate_shorter",
        &[extended_space(), extended_space(), Space::Real],
        |q| {
            onto_extended(interpolate_stored(
                q[0].include(),
                q[1].include(),
                q[2],
                2,
                true,
            ))
        },
    )
});

/// `T₁ * T₂ = (P(q₁ ⊗ q₂), v₁ + R₁ v₂, p₁ + R₁ p₂)`.
impl QMul for ExtendedPose {
    type Output = ExtendedPose;
    #[track_caller]
    fn mul(lhs: Q<ExtendedPose>, rhs: Q<ExtendedPose>) -> Q<ExtendedPose> {
        COMPOSE.call(&[lhs.untyped(), rhs.untyped()]).narrow()
    }
}

/// The maps of an extended pose, as methods of its handle.
pub trait ExtendedPoseMaps: Sized {
    /// `(P(q̄), -Rᵀ v, -Rᵀ p)`.
    fn inverse(self) -> Q<ExtendedPose>;
    /// `T₁⁻¹ T₂`.
    fn between(self, other: Q<ExtendedPose>) -> Q<ExtendedPose>;
    /// The rotation `q`.
    fn rotation(self) -> Q<Rotation>;
    /// The velocity `v`.
    fn velocity(self) -> Q<Vector<3>>;
    /// The position `p`.
    fn position(self) -> Q<Vector<3>>;
    /// The pose `(q, p)`.
    fn pose(self) -> Q<Pose>;
    /// The `5 × 5` matrix with the columns `(R, v, p)` over `((0, 1, 0), (0, 0, 1))`.
    fn matrix(self) -> Q<Matrix<5, 5>>;
    /// The adjoint `((R, 0, 0), ([v]× R, R, 0), ([p]× R, 0, R))` in the order `(φ, η, ρ)`.
    fn adjoint(self) -> Q<Matrix<9, 9>>;
    /// `Log(q, v, p) = (φ, J_l(φ)⁻¹ v, J_l(φ)⁻¹ p)` with `φ = Log q`; non-finite at `q = -1`.
    fn log(self) -> Q<Vector<9>>;
    /// The same with the rotation vector of the shorter turn.
    fn log_shorter(self) -> Q<Vector<9>>;
    /// `T₁ Exp(t Log(T₁⁻¹ T₂))`, from `self` at `t = 0` to `other` at `t = 1`.
    fn interpolate(self, other: Q<ExtendedPose>, t: Q<Real>) -> Q<ExtendedPose>;
    /// The same along the shorter turn.
    fn interpolate_shorter(self, other: Q<ExtendedPose>, t: Q<Real>) -> Q<ExtendedPose>;
}

impl ExtendedPoseMaps for Q<ExtendedPose> {
    #[track_caller]
    fn inverse(self) -> Q<ExtendedPose> {
        INVERSE.call(&[self.untyped()]).narrow()
    }

    #[track_caller]
    fn between(self, other: Q<ExtendedPose>) -> Q<ExtendedPose> {
        BETWEEN.call(&[self.untyped(), other.untyped()]).narrow()
    }

    #[track_caller]
    fn rotation(self) -> Q<Rotation> {
        project(Rotation, self.untyped().include().slice(0..4))
    }

    #[track_caller]
    fn velocity(self) -> Q<Vector<3>> {
        self.untyped().include().slice(4..7).narrow()
    }

    #[track_caller]
    fn position(self) -> Q<Vector<3>> {
        self.untyped().include().slice(7..10).narrow()
    }

    #[track_caller]
    fn pose(self) -> Q<Pose> {
        let x = self.untyped().include();
        project(Pose, join(x.slice(0..4), &[x.slice(7..10)]))
    }

    #[track_caller]
    fn matrix(self) -> Q<Matrix<5, 5>> {
        MATRIX.call(&[self.untyped()]).narrow()
    }

    #[track_caller]
    fn adjoint(self) -> Q<Matrix<9, 9>> {
        ADJOINT.call(&[self.untyped()]).narrow()
    }

    #[track_caller]
    fn log(self) -> Q<Vector<9>> {
        LOG.call(&[self.untyped()]).narrow()
    }

    #[track_caller]
    fn log_shorter(self) -> Q<Vector<9>> {
        LOG_SHORTER.call(&[self.untyped()]).narrow()
    }

    #[track_caller]
    fn interpolate(self, other: Q<ExtendedPose>, t: Q<Real>) -> Q<ExtendedPose> {
        INTERPOLATE
            .call(&[self.untyped(), other.untyped(), t.untyped()])
            .narrow()
    }

    #[track_caller]
    fn interpolate_shorter(self, other: Q<ExtendedPose>, t: Q<Real>) -> Q<ExtendedPose> {
        INTERPOLATE_SHORTER
            .call(&[self.untyped(), other.untyped(), t.untyped()])
            .narrow()
    }
}

/// The identity extended pose.
#[track_caller]
pub fn identity() -> Q<ExtendedPose> {
    constant(ExtendedPose, ([1.0, 0.0, 0.0, 0.0], [0.0; 3], [0.0; 3]))
}

/// The extended pose of the rotation `q`, the velocity `v` and the position `p`.
#[track_caller]
pub fn from_parts(q: Q<Rotation>, v: Q<Vector<3>>, p: Q<Vector<3>>) -> Q<ExtendedPose> {
    project(
        ExtendedPose,
        join(q.untyped().include(), &[v.untyped(), p.untyped()]),
    )
}

/// `Exp(φ, η, ρ) = (Exp φ, J_l(φ) η, J_l(φ) ρ)`.
#[track_caller]
pub fn exp(xi: Q<Vector<9>>) -> Q<ExtendedPose> {
    EXP.call(&[xi.untyped()]).narrow()
}

/// The left Jacobian of `exp`: `J_l(φ)` on the diagonal, `Q(φ, η)` and `Q(φ, ρ)` below it in the
/// first block column, in the order `(φ, η, ρ)`.
#[track_caller]
pub fn jl(xi: Q<Vector<9>>) -> Q<Matrix<9, 9>> {
    JL.call(&[xi.untyped()]).narrow()
}

/// The right Jacobian, `𝒥_l(-ξ)`.
#[track_caller]
pub fn jr(xi: Q<Vector<9>>) -> Q<Matrix<9, 9>> {
    JL.call(&[(-xi).untyped()]).narrow()
}

/// The left Jacobian's inverse: `J_l⁻¹` on the diagonal, `-J_l⁻¹ Q(φ, η) J_l⁻¹` and
/// `-J_l⁻¹ Q(φ, ρ) J_l⁻¹` below it; non-finite at `|φ| = 2π`.
#[track_caller]
pub fn jl_inv(xi: Q<Vector<9>>) -> Q<Matrix<9, 9>> {
    JL_INV.call(&[xi.untyped()]).narrow()
}

/// The right Jacobian's inverse, `𝒥_l(-ξ)⁻¹`.
#[track_caller]
pub fn jr_inv(xi: Q<Vector<9>>) -> Q<Matrix<9, 9>> {
    JL_INV.call(&[(-xi).untyped()]).narrow()
}
