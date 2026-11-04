//! The Euclidean maps of rotation vectors: the hat and the vee.

use skoro_core::scope::concatenate;
use skoro_core::{Matrix, Q, Space, Vector};

use crate::algebra::{self, entry};

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
