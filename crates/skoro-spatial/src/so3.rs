//! The Euclidean maps of rotation vectors: the hat, the vee, and the Jacobians of the exponential
//! and their inverses, each `I + c₁ [ω]× + c₂ [ω]×²`.

use std::sync::{Arc, LazyLock};

use skoro_core::scope::{concatenate, constant};
use skoro_core::{Composite, Matrix, Q, QDyn, Space, Vector};

use crate::algebra::{self, dot, entry};
use crate::rotation::composite;
use crate::series::{Coefficient, coefficient};

/// `[ω]×`, the skew matrix with `[ω]× x = ω × x`.
#[track_caller]
pub fn hat(omega: Q<Vector<3>>) -> Q<Matrix<3, 3>> {
    algebra::hat(omega.untyped()).narrow()
}

/// The vector of the skew part of `m`, `½ (m₃₂ - m₂₃, m₁₃ - m₃₁, m₂₁ - m₁₂)`: the inverse of `hat`
/// on skew matrices.
#[track_caller]
pub fn vee(m: Q<Matrix<3, 3>>) -> Q<Vector<3>> {
    // m_ij is stored at i + 3j.
    let m = m.untyped();
    let difference = |a: usize, b: usize| (entry(m, a) - entry(m, b)) * 0.5;
    concatenate(
        &[difference(5, 7), difference(6, 2), difference(1, 3)],
        Space::Vector(3),
    )
    .narrow()
}

/// The identity `Matrix(3, 3)`.
pub(crate) fn identity3() -> QDyn {
    let rows = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
    constant(Matrix::<3, 3>, rows).untyped()
}

/// `I + c₁ Ω + c₂ Ω²` with `Ω = [ω]×`.
fn polynomial(omega: QDyn, c1: QDyn, c2: QDyn) -> QDyn {
    let h = algebra::hat(omega);
    identity3() + c1 * h + c2 * h.matmul(h)
}

/// `J_l(ω) = I + A Ω + B Ω²`, of the storage of `ω`.
pub(crate) fn jl_stored(omega: QDyn) -> QDyn {
    let theta2 = dot(omega, omega);
    let (a, b) = (
        coefficient(Coefficient::A, theta2),
        coefficient(Coefficient::B, theta2),
    );
    polynomial(omega, a, b)
}

/// `J_l(ω)⁻¹ = I - ½ Ω + C Ω²`, of the storage of `ω`.
pub(crate) fn jl_inv_stored(omega: QDyn) -> QDyn {
    let theta2 = dot(omega, omega);
    let half = constant(skoro_core::Real, -0.5).untyped();
    polynomial(omega, half, coefficient(Coefficient::C, theta2))
}

static JL: LazyLock<Arc<Composite>> =
    LazyLock::new(|| composite("skoro.spatial.jl", &[Space::Vector(3)], |q| jl_stored(q[0])));

static JL_INV: LazyLock<Arc<Composite>> = LazyLock::new(|| {
    composite("skoro.spatial.jl_inv", &[Space::Vector(3)], |q| {
        jl_inv_stored(q[0])
    })
});

/// The left Jacobian of the exponential, `Exp(ω + ε) = Exp(J_l(ω) ε) Exp(ω)` to first order:
/// `I + A Ω + B Ω²`, `A = (1 - cos θ)/θ²`, `B = (θ - sin θ)/θ³`.
#[track_caller]
pub fn jl(omega: Q<Vector<3>>) -> Q<Matrix<3, 3>> {
    JL.call(&[omega.untyped()]).narrow()
}

/// The right Jacobian, `Exp(ω + ε) = Exp(ω) Exp(J_r(ω) ε)` to first order: `J_l(-ω)`.
#[track_caller]
pub fn jr(omega: Q<Vector<3>>) -> Q<Matrix<3, 3>> {
    JL.call(&[(-omega).untyped()]).narrow()
}

/// `J_l(ω)⁻¹ = I - ½ Ω + C Ω²`, `C = (1 - (θ/2) cot(θ/2))/θ²`; non-finite at `|ω| = 2π`.
#[track_caller]
pub fn jl_inv(omega: Q<Vector<3>>) -> Q<Matrix<3, 3>> {
    JL_INV.call(&[omega.untyped()]).narrow()
}

/// `J_r(ω)⁻¹ = J_l(-ω)⁻¹ = I + ½ Ω + C Ω²`; non-finite at `|ω| = 2π`.
#[track_caller]
pub fn jr_inv(omega: Q<Vector<3>>) -> Q<Matrix<3, 3>> {
    JL_INV.call(&[(-omega).untyped()]).narrow()
}
