//! The maps of the rigid motions: `*` composing and acting on points, the methods of
//! [`PoseMaps`] on a handle, and the free functions that make a motion.
//!
//! The arithmetic of a motion with `k` translations, a rotation `q` followed by `t₁ … t_k`, is
//! written once here on storage, for the poses (`k = 1`) and the extended poses (`k = 2`):
//! composition `(P(q₁ ⊗ q₂), tᵢ + R₁ t'ᵢ)`, inverse `(P(q̄), -Rᵀ tᵢ)`, the homogeneous matrix with
//! the columns `(R, t₁, …, t_k)`, and the adjoint with `R` on its diagonal and `[tᵢ]× R` in its
//! first block column.

use std::sync::{Arc, LazyLock};

use skoro_core::scope::{concatenate, constant, product, project};
use skoro_core::{Composite, Declares, Matrix, ProductDyn, Q, QDyn, QMul, Space, Vector};

use crate::algebra::{self, conjugate, mul, rotate, unrotate, zero};
use crate::rotation::composite;
use crate::{Pose, Rotation};

/// The rotation's storage and the translations of a motion's storage.
pub(crate) fn split(x: QDyn, k: usize) -> (QDyn, Vec<QDyn>) {
    let translations = (0..k).map(|i| x.slice(4 + 3 * i..7 + 3 * i)).collect();
    (x.slice(0..4), translations)
}

/// A motion's storage from its rotation's and its translations.
pub(crate) fn join(q: QDyn, translations: &[QDyn]) -> QDyn {
    let parts: Vec<QDyn> = std::iter::once(q)
        .chain(translations.iter().copied())
        .collect();
    concatenate(&parts, Space::Vector(4 + 3 * translations.len()))
}

/// `(q₁ ⊗ q₂, tᵢ + R₁ t'ᵢ)`, before the projection.
pub(crate) fn compose_stored(a: QDyn, b: QDyn, k: usize) -> QDyn {
    let ((qa, ta), (qb, tb)) = (split(a, k), split(b, k));
    let moved: Vec<QDyn> = ta
        .iter()
        .zip(&tb)
        .map(|(t, s)| *t + rotate(qa, *s))
        .collect();
    join(mul(qa, qb), &moved)
}

/// `(q̄, -Rᵀ tᵢ)`.
pub(crate) fn inverse_stored(x: QDyn, k: usize) -> QDyn {
    let (q, t) = split(x, k);
    let moved: Vec<QDyn> = t.iter().map(|t| -unrotate(q, *t)).collect();
    join(conjugate(q), &moved)
}

/// `(q̄₁ ⊗ q₂, R₁ᵀ (t'ᵢ - tᵢ))`.
pub(crate) fn between_stored(a: QDyn, b: QDyn, k: usize) -> QDyn {
    let ((qa, ta), (qb, tb)) = (split(a, k), split(b, k));
    let moved: Vec<QDyn> = ta
        .iter()
        .zip(&tb)
        .map(|(t, s)| unrotate(qa, *s - *t))
        .collect();
    join(mul(conjugate(qa), qb), &moved)
}

/// The zero `Vector(3)`.
pub(crate) fn zeros3() -> QDyn {
    constant(Vector::<3>, [0.0; 3]).untyped()
}

/// A square matrix of `n × n` blocks of `3 × 3`, each a stored `Matrix(3, 3)` or zero, stored
/// column by column.
pub(crate) fn blocks(b: &[Vec<Option<QDyn>>]) -> QDyn {
    let n = b.len();
    let mut pieces = Vec::with_capacity(9 * n * n);
    for column in 0..n {
        for j in 0..3 {
            for row in b {
                pieces.push(match row[column] {
                    Some(block) => block.slice(3 * j..3 * j + 3),
                    None => zeros3(),
                });
            }
        }
    }
    concatenate(&pieces, Space::Matrix(3 * n, 3 * n))
}

/// The homogeneous matrix of a motion: the columns `(R, t₁, …, t_k)` over `(0, I_k)`.
pub(crate) fn matrix_stored(x: QDyn, k: usize) -> QDyn {
    let (q, t) = split(x, k);
    let r = algebra::matrix(q);
    let size = 3 + k;
    let mut pieces = Vec::new();
    for j in 0..3 {
        pieces.push(r.slice(3 * j..3 * j + 3));
        pieces.extend((0..k).map(|_| zero()));
    }
    for (i, t) in t.iter().enumerate() {
        pieces.push(*t);
        pieces.extend((0..k).map(|row| if row == i { algebra::one() } else { zero() }));
    }
    concatenate(&pieces, Space::Matrix(size, size))
}

/// The adjoint of a motion, in the order `(φ, t₁, …)`: `R` on the diagonal, `[tᵢ]× R` below it
/// in the first block column.
pub(crate) fn adjoint_stored(x: QDyn, k: usize) -> QDyn {
    let (q, t) = split(x, k);
    let r = algebra::matrix(q);
    let rows: Vec<Vec<Option<QDyn>>> = (0..=k)
        .map(|row| {
            (0..=k)
                .map(|column| match (row, column) {
                    (row, column) if row == column => Some(r),
                    (row, 0) => Some(algebra::hat(t[row - 1]).matmul(r)),
                    _ => None,
                })
                .collect()
        })
        .collect();
    blocks(&rows)
}

fn pose_space() -> Space {
    Pose.space()
}

/// `P` onto the poses.
fn onto_pose(w: QDyn) -> QDyn {
    project(Pose, w).untyped()
}

static COMPOSE: LazyLock<Arc<Composite>> = LazyLock::new(|| {
    composite(
        "skoro.spatial.pose.compose",
        &[pose_space(), pose_space()],
        |q| onto_pose(compose_stored(q[0].include(), q[1].include(), 1)),
    )
});

static INVERSE: LazyLock<Arc<Composite>> = LazyLock::new(|| {
    composite("skoro.spatial.pose.inverse", &[pose_space()], |q| {
        onto_pose(inverse_stored(q[0].include(), 1))
    })
});

static BETWEEN: LazyLock<Arc<Composite>> = LazyLock::new(|| {
    composite(
        "skoro.spatial.pose.between",
        &[pose_space(), pose_space()],
        |q| onto_pose(between_stored(q[0].include(), q[1].include(), 1)),
    )
});

static TRANSFORM: LazyLock<Arc<Composite>> = LazyLock::new(|| {
    composite(
        "skoro.spatial.pose.transform",
        &[pose_space(), Space::Vector(3)],
        |q| {
            let (r, t) = split(q[0].include(), 1);
            rotate(r, q[1]) + t[0]
        },
    )
});

static ROTATE: LazyLock<Arc<Composite>> = LazyLock::new(|| {
    composite(
        "skoro.spatial.pose.rotate",
        &[pose_space(), Space::Vector(3)],
        |q| rotate(q[0].include().slice(0..4), q[1]),
    )
});

static UNTRANSFORM: LazyLock<Arc<Composite>> = LazyLock::new(|| {
    composite(
        "skoro.spatial.pose.untransform",
        &[pose_space(), Space::Vector(3)],
        |q| {
            let (r, t) = split(q[0].include(), 1);
            unrotate(r, q[1] - t[0])
        },
    )
});

static MATRIX: LazyLock<Arc<Composite>> = LazyLock::new(|| {
    composite("skoro.spatial.pose.matrix", &[pose_space()], |q| {
        matrix_stored(q[0].include(), 1)
    })
});

static ADJOINT: LazyLock<Arc<Composite>> = LazyLock::new(|| {
    composite("skoro.spatial.pose.adjoint", &[pose_space()], |q| {
        adjoint_stored(q[0].include(), 1)
    })
});

/// `T₁ * T₂ = (P(q₁ ⊗ q₂), p₁ + R₁ p₂)`.
impl QMul for Pose {
    type Output = Pose;
    #[track_caller]
    fn mul(lhs: Q<Pose>, rhs: Q<Pose>) -> Q<Pose> {
        COMPOSE.call(&[lhs.untyped(), rhs.untyped()]).narrow()
    }
}

/// `T * x = R x + p`, the action on a point.
impl QMul<Vector<3>> for Pose {
    type Output = Vector<3>;
    #[track_caller]
    fn mul(lhs: Q<Pose>, rhs: Q<Vector<3>>) -> Q<Vector<3>> {
        TRANSFORM.call(&[lhs.untyped(), rhs.untyped()]).narrow()
    }
}

/// The maps of a rigid motion, as methods of its handle. Tangent coordinates are `(φ, ρ)`,
/// rotation first, in the frame of the motion.
pub trait PoseMaps: Sized {
    /// `(P(q̄), -Rᵀ p)`.
    fn inverse(self) -> Q<Pose>;
    /// `T₁⁻¹ T₂`: the motion from `self` to `other`, in the frame of `self`.
    fn between(self, other: Q<Pose>) -> Q<Pose>;
    /// `R x`: a direction carried by the rotation alone.
    fn rotate(self, x: Q<Vector<3>>) -> Q<Vector<3>>;
    /// `Rᵀ (x - p)`: a point of the outer frame in the frame of `self`.
    fn untransform(self, x: Q<Vector<3>>) -> Q<Vector<3>>;
    /// The rotation `q`.
    fn rotation(self) -> Q<Rotation>;
    /// The position `p`.
    fn position(self) -> Q<Vector<3>>;
    /// The homogeneous matrix, `R` and `p` over `(0, 1)`.
    fn matrix(self) -> Q<Matrix<4, 4>>;
    /// The adjoint `((R, 0), ([p]× R, R))` in the order `(φ, ρ)`, carrying a perturbation in the
    /// frame of `self` to the outer frame to first order.
    fn adjoint(self) -> Q<Matrix<6, 6>>;
    /// The same storage as a value of the product `S³ × ℝ³`, whose chart moves the position in a
    /// fixed frame.
    #[expect(
        clippy::wrong_self_convention,
        reason = "a handle is Copy, and taken by value as every map takes it"
    )]
    fn as_product(self) -> Q<ProductDyn>;
}

impl PoseMaps for Q<Pose> {
    #[track_caller]
    fn inverse(self) -> Q<Pose> {
        INVERSE.call(&[self.untyped()]).narrow()
    }

    #[track_caller]
    fn between(self, other: Q<Pose>) -> Q<Pose> {
        BETWEEN.call(&[self.untyped(), other.untyped()]).narrow()
    }

    #[track_caller]
    fn rotate(self, x: Q<Vector<3>>) -> Q<Vector<3>> {
        ROTATE.call(&[self.untyped(), x.untyped()]).narrow()
    }

    #[track_caller]
    fn untransform(self, x: Q<Vector<3>>) -> Q<Vector<3>> {
        UNTRANSFORM.call(&[self.untyped(), x.untyped()]).narrow()
    }

    #[track_caller]
    fn rotation(self) -> Q<Rotation> {
        project(Rotation, self.untyped().include().slice(0..4))
    }

    #[track_caller]
    fn position(self) -> Q<Vector<3>> {
        self.untyped().include().slice(4..7).narrow()
    }

    #[track_caller]
    fn matrix(self) -> Q<Matrix<4, 4>> {
        MATRIX.call(&[self.untyped()]).narrow()
    }

    #[track_caller]
    fn adjoint(self) -> Q<Matrix<6, 6>> {
        ADJOINT.call(&[self.untyped()]).narrow()
    }

    #[track_caller]
    fn as_product(self) -> Q<ProductDyn> {
        product(&[self.rotation().untyped(), self.position().untyped()])
    }
}

/// The identity motion.
#[track_caller]
pub fn identity() -> Q<Pose> {
    constant(Pose, ([1.0, 0.0, 0.0, 0.0], [0.0; 3]))
}

/// The motion of the rotation `q` and the position `p`.
#[track_caller]
pub fn from_parts(q: Q<Rotation>, p: Q<Vector<3>>) -> Q<Pose> {
    project(Pose, join(q.untyped().include(), &[p.untyped()]))
}

/// The motion of a value of the product `S³ × ℝ³`: the identity on storage (`SpaceMismatch` unless
/// the product is of `Rotation` and `Vector<3>`).
#[track_caller]
pub fn from_product(x: Q<ProductDyn>) -> Q<Pose> {
    from_parts(x.factor(0).narrow(), x.factor(1).narrow())
}
