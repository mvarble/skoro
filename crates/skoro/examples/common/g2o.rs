//! A 3D pose graph in the text format of g2o: its reader, which converts each edge's information
//! to the precision of skoro's residual, and a one-turn loop written in that format.
//!
//! Self-contained, on the standard library and `skoro_spatial::host` alone, so that a test can
//! include it by path without the rest of the examples' common modules.

use std::collections::BTreeMap;
use std::f64::consts::PI;
use std::fmt::Write as _;

use skoro_spatial::host;

/// A 3D pose graph as g2o writes it: the vertices' initial poses and the edges' measurements.
pub struct Graph {
    /// By index, each `(w, x, y, z, px, py, pz)`, the quaternion reordered from g2o's
    /// `qx qy qz qw` and normalized, its sign kept.
    pub vertices: BTreeMap<usize, [f64; 7]>,
    pub edges: Vec<Edge>,
    /// The lines that were neither a vertex nor an edge of the 3D pose graph, skipped.
    pub skipped: usize,
}

/// One edge: the measured motion from one vertex to another, and its precision.
pub struct Edge {
    pub from: usize,
    pub to: usize,
    /// The measured motion of `X_from⁻¹ X_to`, as a vertex's pose is stored.
    pub measurement: [f64; 7],
    /// The precision of skoro's residual, rotation first, 6 × 6 row by row: from g2o's
    /// information `Ω` of its error, translation first, as `AᵀΩA` with `A(τ_R, τ_T) =
    /// (τ_T, τ_R / 2)`, a quarter of `Ω`'s rotation block, half its coupling block, and its
    /// translation block.
    pub precision: [f64; 36],
}

/// A pose `x y z qx qy qz qw` of g2o, as skoro stores it, the quaternion normalized and its sign
/// kept.
fn pose(fields: &[f64]) -> Result<[f64; 7], String> {
    let q = [fields[6], fields[3], fields[4], fields[5]];
    let norm = q.iter().map(|v| v * v).sum::<f64>().sqrt();
    if !(norm > 0.0 && norm.is_finite()) {
        return Err("a quaternion that is zero or not finite".to_string());
    }
    Ok([
        q[0] / norm,
        q[1] / norm,
        q[2] / norm,
        q[3] / norm,
        fields[0],
        fields[1],
        fields[2],
    ])
}

/// The precision of skoro's residual from g2o's information, both 6 × 6 row by row.
pub fn precision(information: &[f64; 36]) -> [f64; 36] {
    // Index i of skoro's residual (rotation, then translation) is index `g2o(i)` of g2o's error
    // (translation, then rotation), scaled by `scale(i)`: e = -A τ to first order.
    let g2o = |i: usize| if i < 3 { i + 3 } else { i - 3 };
    let scale = |i: usize| if i < 3 { 0.5 } else { 1.0 };
    std::array::from_fn(|ij| {
        let (i, j) = (ij / 6, ij % 6);
        scale(i) * scale(j) * information[g2o(i) * 6 + g2o(j)]
    })
}

/// Reads `VERTEX_SE3:QUAT` and `EDGE_SE3:QUAT` lines; other lines are skipped and counted, and a
/// malformed one is an error naming its line.
pub fn read(text: &str) -> Result<Graph, String> {
    let mut graph = Graph {
        vertices: BTreeMap::new(),
        edges: Vec::new(),
        skipped: 0,
    };
    for (number, line) in text.lines().enumerate() {
        let mut words = line.split_whitespace();
        let Some(tag) = words.next() else { continue };
        let at = |e: String| format!("line {}: {e}", number + 1);
        let numbers = |words: std::str::SplitWhitespace<'_>| -> Result<Vec<f64>, String> {
            words
                .map(|w| {
                    w.parse::<f64>()
                        .map_err(|_| format!("`{w}` is not a number"))
                })
                .collect()
        };
        match tag {
            "VERTEX_SE3:QUAT" => {
                let fields = numbers(words).map_err(at)?;
                if fields.len() != 8 {
                    return Err(at(format!("{} fields where 8 were expected", fields.len())));
                }
                graph.vertices.insert(
                    index(fields[0]).map_err(at)?,
                    pose(&fields[1..]).map_err(at)?,
                );
            }
            "EDGE_SE3:QUAT" => {
                let fields = numbers(words).map_err(at)?;
                if fields.len() != 30 {
                    return Err(at(format!(
                        "{} fields where 30 were expected",
                        fields.len()
                    )));
                }
                let mut information = [0.0; 36];
                let mut upper = fields[9..].iter();
                for i in 0..6 {
                    for j in i..6 {
                        let v = *upper.next().ok_or_else(|| at("too few entries".into()))?;
                        information[i * 6 + j] = v;
                        information[j * 6 + i] = v;
                    }
                }
                graph.edges.push(Edge {
                    from: index(fields[0]).map_err(at)?,
                    to: index(fields[1]).map_err(at)?,
                    measurement: pose(&fields[2..9]).map_err(at)?,
                    precision: precision(&information),
                });
            }
            _ => graph.skipped += 1,
        }
    }
    Ok(graph)
}

fn index(v: f64) -> Result<usize, String> {
    if v >= 0.0 && v.fract() == 0.0 {
        Ok(v as usize)
    } else {
        Err(format!("`{v}` is not an index"))
    }
}

/// A standard normal draw from a seeded generator: splitmix64 and Box–Muller.
struct Noise(u64);

impl Noise {
    fn uniform(&mut self) -> f64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^= z >> 31;
        ((z >> 11) as f64 + 0.5) / (1u64 << 53) as f64
    }

    fn normal(&mut self) -> f64 {
        let (u, v) = (self.uniform(), self.uniform());
        (-2.0 * u.ln()).sqrt() * (2.0 * PI * v).cos()
    }
}

/// `T₁⁻¹ T₂` of two poses in storage order.
pub fn between(a: [f64; 7], b: [f64; 7]) -> [f64; 7] {
    let inverse_q = [a[0], -a[1], -a[2], -a[3]];
    let q = host::compose(inverse_q, [b[0], b[1], b[2], b[3]]);
    let r = host::matrix([a[0], a[1], a[2], a[3]]);
    let d = [b[4] - a[4], b[5] - a[5], b[6] - a[6]];
    let p: [f64; 3] = std::array::from_fn(|i| r[0][i] * d[0] + r[1][i] * d[1] + r[2][i] * d[2]);
    [q[0], q[1], q[2], q[3], p[0], p[1], p[2]]
}

/// A pose moved by noise in its own frame: the rotation by `exp(φ)` on the right, the position
/// by `R ρ`.
fn perturbed(x: [f64; 7], phi: [f64; 3], rho: [f64; 3]) -> [f64; 7] {
    let e = host::exp(phi);
    host::pose_compose(x, [e[0], e[1], e[2], e[3], rho[0], rho[1], rho[2]])
}

/// One line of g2o's format for a pose, `x y z qx qy qz qw`.
fn written(x: [f64; 7]) -> String {
    format!(
        "{} {} {} {} {} {} {}",
        x[4], x[5], x[6], x[1], x[2], x[3], x[0]
    )
}

/// The information of g2o's error for isotropic noises `σ_t` in position and `σ_r` in rotation:
/// its rotation coordinate is half an angle, so its information is `4 / σ_r²`.
fn information(sigma_t: f64, sigma_r: f64) -> String {
    let mut out = String::new();
    for i in 0..6 {
        for j in i..6 {
            let v = match (i == j, i < 3) {
                (true, true) => 1.0 / (sigma_t * sigma_t),
                (true, false) => 4.0 / (sigma_r * sigma_r),
                _ => 0.0,
            };
            let _ = write!(out, " {v}");
        }
    }
    out
}

/// The text of a one-turn loop of `n` poses on a circle of radius 10, the robot turning by
/// `2π / n` about the vertical at each step: every measurement written with `w ≥ 0`, the loop
/// closure from vertex `n` back to vertex `0` near the identity, each perturbed by `noise` (in
/// radians and in units of the radius over a hundred); and every vertex at the pose that composing
/// the measured odometry from vertex 0 gives, in the sign the composition gives, so that the last
/// vertex's quaternion is near `-1` and the loop closure is on the far side of its prediction.
pub fn one_turn_loop(n: usize, noise: f64, seed: u64) -> String {
    let mut draw = Noise(seed);
    let mut text = String::new();
    let step = 2.0 * PI / n as f64;
    let truth = |k: usize| {
        let heading = step * k as f64;
        let q = host::exp([0.0, 0.0, heading]);
        [
            q[0],
            q[1],
            q[2],
            q[3],
            10.0 * heading.sin(),
            10.0 * (1.0 - heading.cos()),
            0.0,
        ]
    };
    let (sigma_r, sigma_t) = (noise.max(1e-3), (0.1 * noise).max(1e-3));
    let mut measure = |a: [f64; 7], b: [f64; 7]| {
        let mut sample = || [draw.normal(), draw.normal(), draw.normal()];
        let (phi, rho) = (sample(), sample());
        let z = perturbed(
            between(a, b),
            phi.map(|v| noise * v),
            rho.map(|v| 0.1 * noise * v),
        );
        let q = host::canonical([z[0], z[1], z[2], z[3]]);
        [q[0], q[1], q[2], q[3], z[4], z[5], z[6]]
    };
    let odometry: Vec<[f64; 7]> = (0..n).map(|k| measure(truth(k), truth(k + 1))).collect();
    let closure = measure(truth(n), truth(0));

    // The vertices, composed from vertex 0 with no choice of sign.
    let mut at = truth(0);
    let _ = writeln!(text, "VERTEX_SE3:QUAT 0 {}", written(at));
    for (k, z) in odometry.iter().enumerate() {
        at = host::pose_compose(at, *z);
        let _ = writeln!(text, "VERTEX_SE3:QUAT {} {}", k + 1, written(at));
    }
    let info = information(sigma_t, sigma_r);
    for (k, z) in odometry.iter().enumerate() {
        let _ = writeln!(text, "EDGE_SE3:QUAT {k} {} {}{info}", k + 1, written(*z));
    }
    let _ = writeln!(text, "EDGE_SE3:QUAT {n} 0 {}{info}", written(closure));
    text
}
