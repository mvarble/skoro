//! Rotations and motions on the host, written with matrices, as references for the tests.

pub type M3 = [[f64; 3]; 3];

pub fn identity3() -> M3 {
    [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
}

pub fn qmul(a: [f64; 4], b: [f64; 4]) -> [f64; 4] {
    [
        a[0] * b[0] - a[1] * b[1] - a[2] * b[2] - a[3] * b[3],
        a[0] * b[1] + a[1] * b[0] + a[2] * b[3] - a[3] * b[2],
        a[0] * b[2] - a[1] * b[3] + a[2] * b[0] + a[3] * b[1],
        a[0] * b[3] + a[1] * b[2] - a[2] * b[1] + a[3] * b[0],
    ]
}

pub fn conj(q: [f64; 4]) -> [f64; 4] {
    [q[0], -q[1], -q[2], -q[3]]
}

/// `π(q)`, by rows, of a unit quaternion.
pub fn rot(q: [f64; 4]) -> M3 {
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

pub fn mm(a: M3, b: M3) -> M3 {
    std::array::from_fn(|i| std::array::from_fn(|j| (0..3).map(|k| a[i][k] * b[k][j]).sum()))
}

pub fn mv(a: M3, x: [f64; 3]) -> [f64; 3] {
    std::array::from_fn(|i| (0..3).map(|k| a[i][k] * x[k]).sum())
}

pub fn transpose(a: M3) -> M3 {
    std::array::from_fn(|i| std::array::from_fn(|j| a[j][i]))
}

pub fn add3(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    std::array::from_fn(|i| a[i] + b[i])
}

pub fn sub3(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    std::array::from_fn(|i| a[i] - b[i])
}

pub fn scale3(c: f64, a: [f64; 3]) -> [f64; 3] {
    a.map(|v| c * v)
}

pub fn cross(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

pub fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

pub fn norm(a: &[f64]) -> f64 {
    dot(a, a).sqrt()
}

/// `[ω]×`, by rows.
pub fn hat(w: [f64; 3]) -> M3 {
    [[0.0, -w[2], w[1]], [w[2], 0.0, -w[0]], [-w[1], w[0], 0.0]]
}

/// A matrix by rows, stored column by column as the core stores it.
pub fn stored<const M: usize, const N: usize>(a: [[f64; N]; M]) -> Vec<f64> {
    (0..N).flat_map(|j| (0..M).map(move |i| a[i][j])).collect()
}

/// A dense matrix by rows of size `n × n`, from storage column by column.
pub fn from_stored(data: &[f64], n: usize) -> Vec<Vec<f64>> {
    (0..n)
        .map(|i| (0..n).map(|j| data[i + n * j]).collect())
        .collect()
}

/// `exp(A)` of a dense square matrix by rows, by scaling and squaring a Taylor series.
pub fn expm(a: &[Vec<f64>]) -> Vec<Vec<f64>> {
    let n = a.len();
    let size = a.iter().flatten().map(|v| v.abs()).fold(0.0, f64::max) * n as f64;
    let squarings = size.max(1.0).log2().ceil() as i32 + 4;
    let scale = 0.5f64.powi(squarings);
    let a: Vec<Vec<f64>> = a
        .iter()
        .map(|r| r.iter().map(|v| v * scale).collect())
        .collect();
    let mut result = dense_identity(n);
    let mut term = dense_identity(n);
    for k in 1..30 {
        term = dense_mul(&term, &a)
            .into_iter()
            .map(|r| r.into_iter().map(|v| v / k as f64).collect())
            .collect();
        for i in 0..n {
            for j in 0..n {
                result[i][j] += term[i][j];
            }
        }
    }
    for _ in 0..squarings {
        result = dense_mul(&result, &result);
    }
    result
}

pub fn dense_identity(n: usize) -> Vec<Vec<f64>> {
    (0..n)
        .map(|i| (0..n).map(|j| if i == j { 1.0 } else { 0.0 }).collect())
        .collect()
}

pub fn dense_mul(a: &[Vec<f64>], b: &[Vec<f64>]) -> Vec<Vec<f64>> {
    let (n, m, p) = (a.len(), b.len(), b[0].len());
    (0..n)
        .map(|i| {
            (0..p)
                .map(|j| (0..m).map(|k| a[i][k] * b[k][j]).sum())
                .collect()
        })
        .collect()
}

/// The homogeneous matrix of a motion's storage with `k` translations, by rows: the columns
/// `(R, t₁, …, t_k)` over `(0, I_k)`.
pub fn homogeneous(x: &[f64], k: usize) -> Vec<Vec<f64>> {
    let r = rot([x[0], x[1], x[2], x[3]]);
    let n = 3 + k;
    let mut h = dense_identity(n);
    for i in 0..3 {
        for j in 0..3 {
            h[i][j] = r[i][j];
        }
        for t in 0..k {
            h[i][3 + t] = x[4 + 3 * t + i];
        }
    }
    h
}

/// The storage of the inverse motion: `(q̄, -Rᵀ tᵢ)`.
pub fn motion_inverse(x: &[f64], k: usize) -> Vec<f64> {
    let q = [x[0], x[1], x[2], x[3]];
    let rt = transpose(rot(q));
    let mut out = conj(q).to_vec();
    for t in 0..k {
        let v = [x[4 + 3 * t], x[5 + 3 * t], x[6 + 3 * t]];
        out.extend(mv(rt, v).map(|c| -c));
    }
    out
}

/// A dense matrix by rows, stored column by column.
pub fn dense_stored(a: &[Vec<f64>]) -> Vec<f64> {
    let (m, n) = (a.len(), a[0].len());
    (0..n).flat_map(|j| (0..m).map(move |i| a[i][j])).collect()
}
