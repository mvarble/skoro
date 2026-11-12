#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)] // a test crate
//! Measured rotations in whatever sign they are given: a pose graph read from g2o's format, its
//! information converted to skoro's coordinates.

#[path = "../examples/common/g2o.rs"]
#[allow(dead_code)]
mod g2o;

use skoro_spatial::host;

/// `(γ(φ), ρ)`: the Cayley chart's step at the identity, in storage order.
fn cayley_step(tau: [f64; 6]) -> [f64; 7] {
    let s = (1.0 + (tau[0] * tau[0] + tau[1] * tau[1] + tau[2] * tau[2]) / 4.0).sqrt();
    [
        1.0 / s,
        tau[0] / 2.0 / s,
        tau[1] / 2.0 / s,
        tau[2] / 2.0 / s,
        tau[3],
        tau[4],
        tau[5],
    ]
}

/// g2o's error of the measurement `z` of the prediction `h`: the translation of `Δ = z⁻¹ h` and the
/// vector part of its quaternion, read off its rotation matrix with `w ≥ 0`.
fn g2o_error(z: [f64; 7], h: [f64; 7]) -> [f64; 6] {
    let delta = g2o::between(z, h);
    let q = host::from_matrix(host::matrix([delta[0], delta[1], delta[2], delta[3]]));
    let q = host::canonical(q);
    [delta[4], delta[5], delta[6], q[1], q[2], q[3]]
}

fn quadratic(m: &[f64; 36], v: &[f64; 6]) -> f64 {
    (0..6)
        .flat_map(|i| (0..6).map(move |j| (i, j)))
        .map(|(i, j)| v[i] * m[i * 6 + j] * v[j])
        .sum()
}

/// An edge line of g2o's format between two vertices, with the measurement `z` and the upper
/// triangle of `information`.
fn edge_line(z: [f64; 7], information: &[f64; 36]) -> String {
    let mut line = format!(
        "EDGE_SE3:QUAT 0 1 {} {} {} {} {} {} {}",
        z[4], z[5], z[6], z[1], z[2], z[3], z[0]
    );
    for i in 0..6 {
        for j in i..6 {
            line.push_str(&format!(" {}", information[i * 6 + j]));
        }
    }
    line
}

#[test]
fn g2o_information_is_converted() {
    // A diagonal information and one with a block coupling the translation to the rotation, both
    // symmetric positive definite, g2o's order: translation, then rotation.
    let mut diagonal = [0.0; 36];
    for (i, v) in [100.0, 200.0, 300.0, 400.0, 500.0, 600.0]
        .into_iter()
        .enumerate()
    {
        diagonal[i * 6 + i] = v;
    }
    let mut coupled = diagonal;
    for (i, j, v) in [(0, 4, 30.0), (1, 3, -20.0), (2, 5, 40.0), (3, 4, 10.0)] {
        coupled[i * 6 + j] = v;
        coupled[j * 6 + i] = v;
    }
    let z = [0.9, 0.1, -0.3, 0.3, 1.0, 2.0, 3.0];
    let norm = (0.81f64 + 0.01 + 0.09 + 0.09).sqrt();
    let z = [
        z[0] / norm,
        z[1] / norm,
        z[2] / norm,
        z[3] / norm,
        z[4],
        z[5],
        z[6],
    ];
    for information in [diagonal, coupled] {
        let graph = g2o::read(&edge_line(z, &information)).unwrap();
        let edge = &graph.edges[0];
        // The measurement, reordered to (w, x, y, z) and normalized, its sign kept.
        for (got, want) in edge.measurement.iter().zip(&z) {
            assert!((got - want).abs() < 1e-12, "{:?}", edge.measurement);
        }
        // The precision: a quarter of the rotation block, half the coupling block, the
        // translation block, rotation first.
        for i in 0..6 {
            for j in 0..6 {
                let (gi, gj) = ((i + 3) % 6, (j + 3) % 6);
                let scale = [i, j]
                    .iter()
                    .map(|&k| if k < 3 { 0.5 } else { 1.0 })
                    .product::<f64>();
                let want = scale * information[gi * 6 + gj];
                assert_eq!(edge.precision[i * 6 + j], want, "({i}, {j})");
            }
        }
    }

    // The two quadratic forms differ at third order in the residual: halving it divides their
    // difference by about eight. The measurement is the prediction moved by τ in skoro's chart,
    // so that skoro's residual is τ exactly.
    let h = [0.8, -0.2, 0.5, 0.26, 0.4, -1.1, 2.0];
    let n = h[..4].iter().map(|v| v * v).sum::<f64>().sqrt();
    let h = [h[0] / n, h[1] / n, h[2] / n, h[3] / n, h[4], h[5], h[6]];
    let direction = [0.5, -0.3, 0.6, 0.2, -0.4, 0.3];
    let length = direction.iter().map(|v| v * v).sum::<f64>().sqrt();
    let lambda = g2o::precision(&coupled);
    let difference = |size: f64| {
        let tau = direction.map(|v| size * v / length);
        let z = host::pose_compose(h, cayley_step(tau));
        let e = g2o_error(z, h);
        quadratic(&lambda, &tau) - quadratic(&coupled, &e)
    };
    let (large, small) = (difference(0.1), difference(0.05));
    let ratio = large / small;
    assert!(
        (6.0..10.0).contains(&ratio),
        "{large:e} then {small:e}: a ratio of {ratio}"
    );
}
