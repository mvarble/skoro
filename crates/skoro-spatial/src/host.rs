//! Alignment and conversions on the host, in double precision, on quaternions `(w, x, y, z)`: for
//! the values a model is given, which are fixed before they are observed.

/// `sign⟨z, r⟩ z`, normalized: the point over `z`'s rotation on the side of `reference`, `z` itself
/// where the two are orthogonal.
pub fn align(z: [f64; 4], reference: [f64; 4]) -> [f64; 4] {
    let inner: f64 = z.iter().zip(&reference).map(|(a, b)| a * b).sum();
    let norm = z.iter().map(|v| v * v).sum::<f64>().sqrt();
    let scale = if inner >= 0.0 { 1.0 } else { -1.0 } / norm;
    z.map(|v| v * scale)
}
