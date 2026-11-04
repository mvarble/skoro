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
use skoro_core::{Composite, Declares, Matrix, ProductDyn, Q, QDyn, QMul, Real, Space, Vector};

use crate::algebra::{self, conjugate, dot, mul, rotate, unrotate, zero};
use crate::rotation::{composite, exp_stored, log_stored, shorter, slerp_stored};
use crate::series::{Coefficient, coefficient};
use crate::so3::{jl_inv_stored, jl_stored};
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

/// `Exp(φ, t₁, …) = (Exp φ, J_l(φ) tᵢ)`, before the projection, of a `Vector(3 + 3k)`.
pub(crate) fn exp_motion_stored(xi: QDyn, k: usize) -> QDyn {
    let phi = xi.slice(0..3);
    let jl = jl_stored(phi);
    let moved: Vec<QDyn> = (0..k)
        .map(|i| jl.matmul(xi.slice(3 + 3 * i..6 + 3 * i)))
        .collect();
    join(exp_stored(phi), &moved)
}

/// `Log(q, t₁, …) = (φ, J_l(φ)⁻¹ tᵢ)` with `φ = Log q`, or `Log(s(q) q)` for the shorter turn.
pub(crate) fn log_motion_stored(x: QDyn, k: usize, shorter_turn: bool) -> QDyn {
    let (q, t) = split(x, k);
    let q = if shorter_turn { shorter(q) } else { q };
    let phi = log_stored(q);
    let inverse = jl_inv_stored(phi);
    let parts: Vec<QDyn> = std::iter::once(phi)
        .chain(t.iter().map(|t| inverse.matmul(*t)))
        .collect();
    concatenate(&parts, Space::Vector(3 + 3 * k))
}

/// `Q(φ, ρ) = ½ Ψ + B (ΦΨ + ΨΦ + ΦΨΦ) + D (Φ²Ψ + ΨΦ² - 3 ΦΨΦ) + E (ΦΨΦ² + Φ²ΨΦ)`, with
/// `Φ = [φ]×` and `Ψ = [ρ]×`: the block below the diagonal of the motions' left Jacobian.
static MOTION_Q: LazyLock<Arc<Composite>> = LazyLock::new(|| {
    composite(
        "skoro.spatial.motion_q",
        &[Space::Vector(3), Space::Vector(3)],
        |q| {
            let (phi, rho) = (q[0], q[1]);
            let theta2 = dot(phi, phi);
            let (b, d, e) = (
                coefficient(Coefficient::B, theta2),
                coefficient(Coefficient::D, theta2),
                coefficient(Coefficient::E, theta2),
            );
            let (f, s) = (algebra::hat(phi), algebra::hat(rho));
            let (fs, sf) = (f.matmul(s), s.matmul(f));
            let fsf = fs.matmul(f);
            let (ffs, sff) = (f.matmul(fs), sf.matmul(f));
            let first = fs + sf + fsf;
            let second = ffs + sff - fsf * 3.0;
            let third = fsf.matmul(f) + f.matmul(fsf);
            s * 0.5 + b * first + d * second + e * third
        },
    )
});

/// The left Jacobian of a motion's exponential, `J_l(φ)` on the diagonal and `Q(φ, tᵢ)` below it
/// in the first block column.
pub(crate) fn jl_motion_stored(xi: QDyn, k: usize) -> QDyn {
    let phi = xi.slice(0..3);
    let jl = jl_stored(phi);
    let rows: Vec<Vec<Option<QDyn>>> = (0..=k)
        .map(|row| {
            (0..=k)
                .map(|column| match (row, column) {
                    (row, column) if row == column => Some(jl),
                    (row, 0) => Some(MOTION_Q.call(&[phi, xi.slice(3 * row..3 * row + 3)])),
                    _ => None,
                })
                .collect()
        })
        .collect();
    blocks(&rows)
}

/// Its inverse: `J_l(φ)⁻¹` on the diagonal and `-J_l⁻¹ Q(φ, tᵢ) J_l⁻¹` below it.
pub(crate) fn jl_inv_motion_stored(xi: QDyn, k: usize) -> QDyn {
    let phi = xi.slice(0..3);
    let inverse = jl_inv_stored(phi);
    let rows: Vec<Vec<Option<QDyn>>> = (0..=k)
        .map(|row| {
            (0..=k)
                .map(|column| match (row, column) {
                    (row, column) if row == column => Some(inverse),
                    (row, 0) => {
                        let q = MOTION_Q.call(&[phi, xi.slice(3 * row..3 * row + 3)]);
                        Some(-inverse.matmul(q).matmul(inverse))
                    }
                    _ => None,
                })
                .collect()
        })
        .collect();
    blocks(&rows)
}

/// `T₁ Exp(t Log(T₁⁻¹ T₂))` in storage, or with the rotation vector of the shorter turn: the
/// screw motion at a constant twist.
pub(crate) fn interpolate_stored(a: QDyn, b: QDyn, t: QDyn, k: usize, shorter_turn: bool) -> QDyn {
    let twist = log_motion_stored(between_stored(a, b, k), k, shorter_turn);
    compose_stored(a, exp_motion_stored(t * twist, k), k)
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

static EXP: LazyLock<Arc<Composite>> = LazyLock::new(|| {
    composite("skoro.spatial.pose.exp", &[Space::Vector(6)], |q| {
        onto_pose(exp_motion_stored(q[0], 1))
    })
});

static LOG: LazyLock<Arc<Composite>> = LazyLock::new(|| {
    composite("skoro.spatial.pose.log", &[pose_space()], |q| {
        log_motion_stored(q[0].include(), 1, false)
    })
});

static LOG_SHORTER: LazyLock<Arc<Composite>> = LazyLock::new(|| {
    composite("skoro.spatial.pose.log_shorter", &[pose_space()], |q| {
        log_motion_stored(q[0].include(), 1, true)
    })
});

static JL: LazyLock<Arc<Composite>> = LazyLock::new(|| {
    composite("skoro.spatial.pose.jl", &[Space::Vector(6)], |q| {
        jl_motion_stored(q[0], 1)
    })
});

static JL_INV: LazyLock<Arc<Composite>> = LazyLock::new(|| {
    composite("skoro.spatial.pose.jl_inv", &[Space::Vector(6)], |q| {
        jl_inv_motion_stored(q[0], 1)
    })
});

/// The interpolation of the rotation along its geodesic and of the position along a line.
fn interpolate_parts_stored(a: QDyn, b: QDyn, t: QDyn, shorter_turn: bool) -> QDyn {
    let ((qa, pa), (qb, pb)) = (split(a, 1), split(b, 1));
    let q = slerp_stored(qa, qb, t, shorter_turn);
    join(q, &[pa[0] * (t * -1.0 + 1.0) + pb[0] * t])
}

fn interpolation(name: &'static str, body: fn(QDyn, QDyn, QDyn) -> QDyn) -> Arc<Composite> {
    composite(name, &[pose_space(), pose_space(), Space::Real], |q| {
        onto_pose(body(q[0].include(), q[1].include(), q[2]))
    })
}

static INTERPOLATE: LazyLock<Arc<Composite>> = LazyLock::new(|| {
    interpolation("skoro.spatial.pose.interpolate", |a, b, t| {
        interpolate_stored(a, b, t, 1, false)
    })
});

static INTERPOLATE_SHORTER: LazyLock<Arc<Composite>> = LazyLock::new(|| {
    interpolation("skoro.spatial.pose.interpolate_shorter", |a, b, t| {
        interpolate_stored(a, b, t, 1, true)
    })
});

static INTERPOLATE_PARTS: LazyLock<Arc<Composite>> = LazyLock::new(|| {
    interpolation("skoro.spatial.pose.interpolate_parts", |a, b, t| {
        interpolate_parts_stored(a, b, t, false)
    })
});

static INTERPOLATE_PARTS_SHORTER: LazyLock<Arc<Composite>> = LazyLock::new(|| {
    interpolation("skoro.spatial.pose.interpolate_parts_shorter", |a, b, t| {
        interpolate_parts_stored(a, b, t, true)
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
    /// `Log(q, p) = (φ, J_l(φ)⁻¹ p)` with `φ = Log q`: the inverse of `pose::exp`; non-finite at
    /// `q = -1`.
    fn log(self) -> Q<Vector<6>>;
    /// The same with the rotation vector of the shorter turn: the same at `±q`, a discretization
    /// at half-turns.
    fn log_shorter(self) -> Q<Vector<6>>;
    /// `T₁ Exp(t Log(T₁⁻¹ T₂))`: the screw motion at a constant twist from `self` at `t = 0` to
    /// `other` at `t = 1`.
    fn interpolate(self, other: Q<Pose>, t: Q<Real>) -> Q<Pose>;
    /// The same along the shorter turn.
    fn interpolate_shorter(self, other: Q<Pose>, t: Q<Real>) -> Q<Pose>;
    /// `(q₁ ⊗ Exp(t Log(q̄₁ ⊗ q₂)), (1 - t) p₁ + t p₂)`: the rotation along its geodesic and the
    /// position along a line, the geodesic of the product.
    fn interpolate_parts(self, other: Q<Pose>, t: Q<Real>) -> Q<Pose>;
    /// The same along the shorter turn.
    fn interpolate_parts_shorter(self, other: Q<Pose>, t: Q<Real>) -> Q<Pose>;
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
    fn log(self) -> Q<Vector<6>> {
        LOG.call(&[self.untyped()]).narrow()
    }

    #[track_caller]
    fn log_shorter(self) -> Q<Vector<6>> {
        LOG_SHORTER.call(&[self.untyped()]).narrow()
    }

    #[track_caller]
    fn interpolate(self, other: Q<Pose>, t: Q<Real>) -> Q<Pose> {
        INTERPOLATE
            .call(&[self.untyped(), other.untyped(), t.untyped()])
            .narrow()
    }

    #[track_caller]
    fn interpolate_shorter(self, other: Q<Pose>, t: Q<Real>) -> Q<Pose> {
        INTERPOLATE_SHORTER
            .call(&[self.untyped(), other.untyped(), t.untyped()])
            .narrow()
    }

    #[track_caller]
    fn interpolate_parts(self, other: Q<Pose>, t: Q<Real>) -> Q<Pose> {
        INTERPOLATE_PARTS
            .call(&[self.untyped(), other.untyped(), t.untyped()])
            .narrow()
    }

    #[track_caller]
    fn interpolate_parts_shorter(self, other: Q<Pose>, t: Q<Real>) -> Q<Pose> {
        INTERPOLATE_PARTS_SHORTER
            .call(&[self.untyped(), other.untyped(), t.untyped()])
            .narrow()
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

/// `Exp(φ, ρ) = (Exp φ, J_l(φ) ρ)`: the motion at the constant body twist `(φ, ρ)` for unit time.
#[track_caller]
pub fn exp(xi: Q<Vector<6>>) -> Q<Pose> {
    EXP.call(&[xi.untyped()]).narrow()
}

/// The left Jacobian of `exp`, `Exp(ξ + ε) = Exp(𝒥_l(ξ) ε) Exp(ξ)` to first order:
/// `((J_l(φ), 0), (Q(φ, ρ), J_l(φ)))` in the order `(φ, ρ)`.
#[track_caller]
pub fn jl(xi: Q<Vector<6>>) -> Q<Matrix<6, 6>> {
    JL.call(&[xi.untyped()]).narrow()
}

/// The right Jacobian, `Exp(ξ + ε) = Exp(ξ) Exp(𝒥_r(ξ) ε)` to first order: `𝒥_l(-ξ)`.
#[track_caller]
pub fn jr(xi: Q<Vector<6>>) -> Q<Matrix<6, 6>> {
    JL.call(&[(-xi).untyped()]).narrow()
}

/// `𝒥_l(ξ)⁻¹ = ((J_l⁻¹, 0), (-J_l⁻¹ Q J_l⁻¹, J_l⁻¹))`; non-finite at `|φ| = 2π`.
#[track_caller]
pub fn jl_inv(xi: Q<Vector<6>>) -> Q<Matrix<6, 6>> {
    JL_INV.call(&[xi.untyped()]).narrow()
}

/// `𝒥_r(ξ)⁻¹ = 𝒥_l(-ξ)⁻¹`; non-finite at `|φ| = 2π`.
#[track_caller]
pub fn jr_inv(xi: Q<Vector<6>>) -> Q<Matrix<6, 6>> {
    JL_INV.call(&[(-xi).untyped()]).narrow()
}
