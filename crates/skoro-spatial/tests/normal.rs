#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)] // a test crate
//! The normal on these types: its density through the chart, against the Gaussian of the chart
//! coordinate, and its draws through `π`.

mod common;

use std::f64::consts::PI;

use common::host::{dot, mv, rot};
use common::{Rows, close, tolerance, vectors};
use skoro_core::scope::{constant, quantity};
use skoro_core::{Declares, Key, Observations, Space, Writer};
use skoro_normal::{Normal, Scale};
use skoro_spatial::{Pose, Rotation};
use skoro_tensor::{Target, Value};

/// The nodes and weights of Gauss–Legendre quadrature on `[-1, 1]`, by Newton's method on the
/// Legendre polynomial of degree `n`.
fn gauss_legendre(n: usize) -> Vec<(f64, f64)> {
    (0..n)
        .map(|i| {
            let mut x = (PI * (i as f64 + 0.75) / (n as f64 + 0.5)).cos();
            loop {
                let (mut p0, mut p1) = (1.0, x);
                for k in 2..=n {
                    let p2 = ((2 * k - 1) as f64 * x * p1 - (k - 1) as f64 * p0) / k as f64;
                    (p0, p1) = (p1, p2);
                }
                let derivative = n as f64 * (x * p1 - p0) / (x * x - 1.0);
                let step = p1 / derivative;
                x -= step;
                if step.abs() < 1e-15 {
                    return (x, 2.0 / ((1.0 - x * x) * derivative * derivative));
                }
            }
        })
        .collect()
}

/// Unit directions and their weights on the sphere: Gauss–Legendre in `cos θ`, uniform in `φ`.
fn directions(n_theta: usize, n_phi: usize) -> Vec<([f64; 3], f64)> {
    let mut out = Vec::new();
    for (c, w) in gauss_legendre(n_theta) {
        let s = (1.0 - c * c).sqrt();
        for k in 0..n_phi {
            let phi = 2.0 * PI * (k as f64 + 0.5) / n_phi as f64;
            out.push((
                [s * phi.cos(), s * phi.sin(), c],
                w * 2.0 * PI / n_phi as f64,
            ));
        }
    }
    out
}

fn continuous(value: &Value) -> Vec<f64> {
    match value {
        Value::Continuous(points) => points.to_host(),
        Value::Discrete(_) => panic!("a continuous value"),
    }
}

/// `log term(μ ⊕ t) + λ_μ(t)` of the normal on the rotations at `mean` with `scale`, at the
/// tangents `t`, on the host.
fn term_through_the_chart(mean: [f64; 4], scale: impl Fn() -> Scale, t: &[[f64; 3]]) -> Vec<f64> {
    common::on_host(
        |q| {
            let mu = constant(Rotation, mean);
            let x = mu.retract(q[0]);
            let parts = Normal::on(Rotation, mu, scale()).log_density(x);
            (parts.log_density + mu.log_volume(q[0])).untyped()
        },
        &[Rows::vectors(t)],
    )
}

/// A scale, made in the scope where it is used.
type ScaleOf = Box<dyn Fn() -> Scale>;

#[test]
fn normal_density_on_rotations() {
    // ∫ exp(term(μ ⊕ t) + λ_μ(t)) dt over ℝ³, in polar coordinates with r = 4 tan(u/4), u the
    // angle of the rotation 1 ⊕ t, in (0, 2π).
    let mean = common::unit([0.4, -0.3, 0.8, 0.2]);
    let (n_u, sphere) = (160, directions(20, 40));
    let du = 2.0 * PI / n_u as f64;
    let mut points = Vec::new();
    let mut weights = Vec::new();
    for i in 0..n_u {
        let u = (i as f64 + 0.5) * du;
        let r = 4.0 * (u / 4.0).tan();
        let dr = 1.0 / (u / 4.0).cos().powi(2);
        for (n, w) in &sphere {
            points.push(n.map(|v| v * r));
            weights.push(w * r * r * dr * du);
        }
    }
    let scales: [(&str, ScaleOf); 2] = [
        ("isotropic", Box::new(|| Scale::sd(0.8))),
        ("anisotropic", Box::new(|| Scale::sds(&[0.6, 1.2, 0.9]))),
    ];
    for (name, scale) in scales {
        let logs = term_through_the_chart(mean, scale, &points);
        let mass: f64 = logs.iter().zip(&weights).map(|(l, w)| l.exp() * w).sum();
        assert!((mass - 1.0).abs() < 1e-6, "{name}: {mass}");
    }
}

/// `log 𝒩(t; 0, Σ)` on the host, `Σ` row by row.
fn log_gaussian(t: &[f64], sigma: &[f64]) -> f64 {
    let d = t.len();
    let mut l = vec![0.0f64; d * d];
    for j in 0..d {
        let pivot = sigma[j * d + j] - (0..j).map(|k| l[j * d + k].powi(2)).sum::<f64>();
        l[j * d + j] = pivot.sqrt();
        for i in j + 1..d {
            let dot: f64 = (0..j).map(|k| l[i * d + k] * l[j * d + k]).sum();
            l[i * d + j] = (sigma[i * d + j] - dot) / l[j * d + j];
        }
    }
    let mut z = vec![0.0; d];
    for i in 0..d {
        let rest: f64 = (0..i).map(|j| l[i * d + j] * z[j]).sum();
        z[i] = (t[i] - rest) / l[i * d + i];
    }
    let log_det: f64 = (0..d).map(|i| l[i * d + i].ln()).sum();
    -0.5 * z.iter().map(|z| z * z).sum::<f64>() - log_det - 0.5 * d as f64 * (2.0 * PI).ln()
}

skoro_core::precision_tests! {
    fn normal_density_on_poses(target) {
        // With a global, full chart and a checked volume, term(μ ⊕ t) + λ_μ(t) = log 𝒩(t; 0, Σ)
        // is the density integrating to one.
        let mean = ([0.3, 0.5, -0.7, 0.4], [1.0, -2.0, 0.5]);
        let a = vectors::<6>(6, 701, -0.5, 0.5);
        let mut sigma = vec![0.0; 36];
        for i in 0..6 {
            for j in 0..6 {
                sigma[i * 6 + j] = (0..6).map(|k| a[i][k] * a[j][k]).sum::<f64>()
                    + if i == j { 0.3 } else { 0.0 };
            }
        }
        let t = vectors::<6>(8, 702, -2.0, 2.0);
        let covariance = sigma.clone();
        let out = common::evaluate(
            |q| {
                let mu = constant(Pose, mean);
                let x = mu.retract(q[0]);
                let parts = Normal::on(Pose, mu, Scale::covariance(&covariance, 6)).log_density(x);
                (parts.log_density + mu.log_volume(q[0])).untyped()
            },
            &[Rows::new(Space::Vector(6), t.concat())],
            &target,
        );
        let want: Vec<f64> = t.iter().map(|t| log_gaussian(t, &sigma)).collect();
        close(&out, &want, tolerance(&target));
    }
}

/// Three bounded functions of a rotation, through its matrix.
fn functions(q: [f64; 4]) -> [f64; 3] {
    let m = rot(q);
    let e = mv(m, [0.0, 0.0, 1.0]);
    [m[0][0], m[1][2], dot(&e, &[0.6, 0.0, 0.8]).powi(2)]
}

#[test]
fn normal_draws_on_rotations() {
    let mean = common::unit([-0.2, 0.7, 0.1, -0.6]);
    let sds = [0.3, 0.5, 0.4];
    let rows = 100_000;
    let target = Target::host();

    // The draws, through π.
    let mut writer = Writer::new();
    let q = writer
        .build(|| {
            let q = quantity("q", Rotation);
            Normal::on(Rotation, constant(Rotation, mean), Scale::sds(&sds)).score(q);
            q
        })
        .unwrap();
    let generated = writer
        .snapshot()
        .generate(&Observations::new(), &Key::from_seed(703), rows, &target)
        .unwrap();
    let drawn = continuous(generated.values.get(q).unwrap());
    let values: Vec<[f64; 3]> = drawn
        .chunks(4)
        .map(|q| functions([q[0], q[1], q[2], q[3]]))
        .collect();

    // The quadratures against p(q) + p(-q) over the hemisphere w > 0, through the chart at the
    // identity, which reaches it at |t| < 4, against ν, whose density there is e^{λ(t)}.
    let (radial, sphere) = (gauss_legendre(48), directions(32, 64));
    let mut points = Vec::new();
    let mut weights = Vec::new();
    for (x, w) in &radial {
        let r = 2.0 * (x + 1.0);
        for (n, v) in &sphere {
            points.push(n.map(|c| c * r));
            weights.push(w * 2.0 * v * r * r * (-3.0 * (1.0 + r * r / 16.0).ln()).exp());
        }
    }
    let at_identity = |t: &[f64; 3]| {
        let s = dot(t, t) / 16.0;
        [
            (1.0 - s) / (1.0 + s),
            t[0] / 2.0 / (1.0 + s),
            t[1] / 2.0 / (1.0 + s),
            t[2] / 2.0 / (1.0 + s),
        ]
    };
    let qs: Vec<[f64; 4]> = points.iter().map(at_identity).collect();
    let minus: Vec<[f64; 4]> = qs.iter().map(|q| q.map(|v| -v)).collect();
    let density = |qs: &[[f64; 4]]| {
        common::on_host(
            |h| {
                let x = h[0].narrow::<Rotation>();
                let mu = constant(Rotation, mean);
                Normal::on(Rotation, mu, Scale::sds(&sds))
                    .log_density(x)
                    .log_density
                    .untyped()
            },
            &[Rows::new(Rotation.space(), qs.concat())],
        )
    };
    let (plus, negated) = (density(&qs), density(&minus));
    let law: Vec<f64> = plus
        .iter()
        .zip(&negated)
        .map(|(a, b)| a.exp() + b.exp())
        .collect();
    let mass: f64 = law.iter().zip(&weights).map(|(p, w)| p * w).sum();
    assert!(
        (mass - 1.0).abs() < 1e-6,
        "the law of the rotation has mass {mass}"
    );

    for f in 0..3 {
        let expected: f64 = qs
            .iter()
            .zip(&law)
            .zip(&weights)
            .map(|((q, p), w)| functions(*q)[f] * p * w)
            .sum();
        let samples: Vec<f64> = values.iter().map(|v| v[f]).collect();
        let mean = samples.iter().sum::<f64>() / rows as f64;
        let variance = samples.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / (rows - 1) as f64;
        let error = (variance / rows as f64).sqrt();
        assert!(
            (mean - expected).abs() < 5.0 * error,
            "function {f}: {mean} drawn against {expected}, standard error {error}"
        );
    }
}
