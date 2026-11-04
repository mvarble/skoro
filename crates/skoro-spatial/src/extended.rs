//! The maps of the extended poses `(q, v, p)`, with the velocity carried as the position is.
//! Tangent coordinates are `(φ, η, ρ)`, rotation first, in the frame of the extended pose; its
//! action on points is that of its pose.

use std::sync::{Arc, LazyLock};

use skoro_core::scope::{constant, project};
use skoro_core::{Composite, Declares, Matrix, Q, QDyn, QMul, Space, Vector};

use crate::pose::{
    adjoint_stored, between_stored, compose_stored, inverse_stored, join, matrix_stored,
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
