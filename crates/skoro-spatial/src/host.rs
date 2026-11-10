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

/// `T₁ * T₂ = (P(q₁ ⊗ q₂), p₁ + R₁ p₂)` on poses in storage order, `(w, x, y, z)` then the
/// position: as `Q<Pose> * Q<Pose>` computes it, with no choice of the quaternion's sign.
pub fn pose_compose(a: [f64; 7], b: [f64; 7]) -> [f64; 7] {
    let q = compose([a[0], a[1], a[2], a[3]], [b[0], b[1], b[2], b[3]]);
    let p = pose_transform(a, [b[4], b[5], b[6]]);
    [q[0], q[1], q[2], q[3], p[0], p[1], p[2]]
}

/// `T * x = R x + p`: as `Q<Pose> * Q<Vector<3>>` computes it.
pub fn pose_transform(t: [f64; 7], x: [f64; 3]) -> [f64; 3] {
    let r = matrix([t[0], t[1], t[2], t[3]]);
    std::array::from_fn(|i| t[4 + i] + r[i][0] * x[0] + r[i][1] * x[1] + r[i][2] * x[2])
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

/// `π(q)` of a unit quaternion, by rows.
pub fn matrix(q: [f64; 4]) -> [[f64; 3]; 3] {
    let [w, x, y, z] = q;
    [
        [
            1.0 - 2.0 * (y * y + z * z),
            2.0 * (x * y - w * z),
            2.0 * (x * z + w * y),
        ],
        [
            2.0 * (x * y + w * z),
            1.0 - 2.0 * (x * x + z * z),
            2.0 * (y * z - w * x),
        ],
        [
            2.0 * (x * z - w * y),
            2.0 * (y * z + w * x),
            1.0 - 2.0 * (x * x + y * y),
        ],
    ]
}

/// The unit quaternion over a rotation matrix given by rows, `P(c_m(X))` with `m` the index of
/// the largest diagonal entry `c_m(X)_m`, as `rotation::from_matrix` chooses it.
pub fn from_matrix(m: [[f64; 3]; 3]) -> [f64; 4] {
    let at = |i: usize, j: usize| m[i - 1][j - 1];
    let (x11, x22, x33) = (at(1, 1), at(2, 2), at(3, 3));
    let c = [
        [
            1.0 + x11 + x22 + x33,
            at(3, 2) - at(2, 3),
            at(1, 3) - at(3, 1),
            at(2, 1) - at(1, 2),
        ],
        [
            at(3, 2) - at(2, 3),
            1.0 + x11 - x22 - x33,
            at(1, 2) + at(2, 1),
            at(1, 3) + at(3, 1),
        ],
        [
            at(1, 3) - at(3, 1),
            at(1, 2) + at(2, 1),
            1.0 - x11 + x22 - x33,
            at(2, 3) + at(3, 2),
        ],
        [
            at(2, 1) - at(1, 2),
            at(1, 3) + at(3, 1),
            at(2, 3) + at(3, 2),
            1.0 - x11 - x22 + x33,
        ],
    ];
    let a = if c[1][1] >= c[0][0] { 1 } else { 0 };
    let b = if c[3][3] >= c[2][2] { 3 } else { 2 };
    let m = if c[b][b] >= c[a][a] { b } else { a };
    let norm = c[m].iter().map(|v| v * v).sum::<f64>().sqrt();
    c[m].map(|v| v / norm)
}

/// `Exp(ω) = (cos(θ/2), sin(θ/2) ω / θ)`.
pub fn exp(omega: [f64; 3]) -> [f64; 4] {
    let theta = omega.iter().map(|v| v * v).sum::<f64>().sqrt();
    let b = if theta < 1e-4 {
        0.5 - theta * theta / 48.0
    } else {
        (0.5 * theta).sin() / theta
    };
    [
        (0.5 * theta).cos(),
        b * omega[0],
        b * omega[1],
        b * omega[2],
    ]
}

/// `Log(q) = 2 atan2(n, w) u / n`, of length in [0, 2π); undefined at `-1`: non-finite exactly
/// there, and near it of length near 2π with a direction set by rounding.
pub fn log(q: [f64; 4]) -> [f64; 3] {
    let u = [q[1], q[2], q[3]];
    let n = u.iter().map(|v| v * v).sum::<f64>().sqrt();
    let ratio = if n == 0.0 && q[0] > 0.0 {
        2.0 / q[0]
    } else if n == 0.0 {
        f64::NAN
    } else {
        2.0 * n.atan2(q[0]) / n
    };
    u.map(|v| ratio * v)
}
