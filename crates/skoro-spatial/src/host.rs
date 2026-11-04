//! Alignment and conversions on the host, in double precision, on quaternions `(w, x, y, z)`: for
//! the values a model is given, which are fixed before they are observed.

use crate::Euler;
use crate::euler::{euler_on_host, from_euler_on_host};

/// `sign⟨z, r⟩ z`, normalized: the point over `z`'s rotation on the side of `reference`, `z` itself
/// where the two are orthogonal.
pub fn align(z: [f64; 4], reference: [f64; 4]) -> [f64; 4] {
    let inner: f64 = z.iter().zip(&reference).map(|(a, b)| a * b).sum();
    let norm = z.iter().map(|v| v * v).sum::<f64>().sqrt();
    let scale = if inner >= 0.0 { 1.0 } else { -1.0 } / norm;
    z.map(|v| v * scale)
}

/// `a ⊗ b`.
pub fn compose(a: [f64; 4], b: [f64; 4]) -> [f64; 4] {
    [
        a[0] * b[0] - a[1] * b[1] - a[2] * b[2] - a[3] * b[3],
        a[0] * b[1] + a[1] * b[0] + a[2] * b[3] - a[3] * b[2],
        a[0] * b[2] - a[1] * b[3] + a[2] * b[0] + a[3] * b[1],
        a[0] * b[3] + a[1] * b[2] - a[2] * b[1] + a[3] * b[0],
    ]
}

/// The rotation of the Euler angles of `convention`, listed in the order of its axes.
pub fn from_euler(angles: [f64; 3], convention: Euler) -> [f64; 4] {
    let (sequence, reversed) = convention.intrinsic();
    let angles = if reversed {
        [angles[2], angles[1], angles[0]]
    } else {
        angles
    };
    from_euler_on_host(sequence, angles)
}

/// The Euler angles of `convention` of `q`, as `RotationMaps::euler` computes them.
pub fn euler(q: [f64; 4], convention: Euler) -> [f64; 3] {
    let (sequence, reversed) = convention.intrinsic();
    let angles = euler_on_host(sequence, q);
    if reversed {
        [angles[2], angles[1], angles[0]]
    } else {
        angles
    }
}
