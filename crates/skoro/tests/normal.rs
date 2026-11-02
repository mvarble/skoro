#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)] // a test crate
//! The normal on curved types: on the acceptance-test sphere, whose stereographic chart is global
//! and full, its density, term, sampler and information.

mod common;

use std::f64::consts::PI;

use common::sphere::{Sphere, basis};
use skoro::scope::{constant, quantity, quantity_in};
use skoro::tensor::{Frame, HostData, HostValue, Points, Value};
use skoro::{
    Frames, Key, NodeId, Observations, Positive, Precision, QDyn, Query, Rows, Snapshot, Target,
    Values, Writer,
};
use skoro_normal::{Normal, Scale, roles, triangular};

fn host(values: &Values, id: NodeId) -> Vec<f64> {
    match values.get(id).unwrap() {
        Value::Continuous(points) => points.to_host(),
        Value::Discrete(_) => panic!("a continuous value"),
    }
}

/// The values of `nodes`, given `given`.
fn evaluate(
    snapshot: &Snapshot,
    nodes: &[QDyn],
    given: Observations,
    target: &Target,
) -> Vec<Vec<f64>> {
    let program = snapshot
        .compile(&Query::of(nodes.iter().copied()).given(given), target)
        .unwrap();
    let out = program.forward(&Values::new()).unwrap();
    nodes.iter().map(|q| host(&out, q.id())).collect()
}

fn normalize(v: [f64; 3]) -> [f64; 3] {
    let n = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
    [v[0] / n, v[1] / n, v[2] / n]
}

/// `μ ⊕ t` on the sphere, by the stereographic chart.
fn retract(mu: [f64; 3], t: [f64; 2]) -> [f64; 3] {
    let b = basis(mu);
    let s = (t[0] * t[0] + t[1] * t[1]) / 4.0;
    let mut y = [0.0; 3];
    for (i, yi) in y.iter_mut().enumerate() {
        *yi = ((1.0 - s) * mu[i] + b[i][0] * t[0] + b[i][1] * t[1]) / (1.0 + s);
    }
    y
}

/// `y ⊖ μ` on the sphere: `2 B(μ)ᵀ y / (1 + μᵀ y)`.
fn local(mu: [f64; 3], y: [f64; 3]) -> [f64; 2] {
    let b = basis(mu);
    let dot = mu[0] * y[0] + mu[1] * y[1] + mu[2] * y[2];
    let bty = |c: usize| (0..3).map(|i| b[i][c] * y[i]).sum::<f64>();
    [2.0 * bty(0) / (1.0 + dot), 2.0 * bty(1) / (1.0 + dot)]
}

/// The stereographic chart's log-volume, `-2 log(1 + |t|²/4)`.
fn log_volume(t: [f64; 2]) -> f64 {
    -2.0 * (1.0 + (t[0] * t[0] + t[1] * t[1]) / 4.0).ln()
}

/// `log N(t; 0, Σ)` for a `Σ` of side 2, row by row.
fn log_gaussian(t: [f64; 2], sigma: [f64; 4]) -> f64 {
    let det = sigma[0] * sigma[3] - sigma[1] * sigma[2];
    let q = (sigma[3] * t[0] * t[0] - 2.0 * sigma[1] * t[0] * t[1] + sigma[0] * t[1] * t[1]) / det;
    -0.5 * q - 0.5 * det.ln() - (2.0 * PI).ln()
}

/// The nodes of a quadrature over the chart at `μ`, in polar coordinates `r = 2 tan u`: each
/// point `μ ⊕ t`, its tangent, and its weight `r dr dφ` with the midpoint rule in `u` and `φ`.
fn quadrature(mu: [f64; 3], n_u: usize, n_phi: usize) -> Vec<([f64; 3], [f64; 2], f64)> {
    let (du, dphi) = (PI / 2.0 / n_u as f64, 2.0 * PI / n_phi as f64);
    let mut nodes = Vec::with_capacity(n_u * n_phi);
    for i in 0..n_u {
        let u = (i as f64 + 0.5) * du;
        let r = 2.0 * u.tan();
        let dr = 2.0 / u.cos().powi(2) * du;
        for j in 0..n_phi {
            let phi = (j as f64 + 0.5) * dphi;
            let t = [r * phi.cos(), r * phi.sin()];
            nodes.push((retract(mu, t), t, r * dr * dphi));
        }
    }
    nodes
}

/// A mean near the south pole, where the basis turns fastest, and one far from it.
fn means() -> [[f64; 3]; 2] {
    [normalize([0.15, -0.1, -0.95]), normalize([0.3, 0.5, 0.8])]
}

/// An isotropic and an anisotropic covariance, of side 2.
const ISOTROPIC: [f64; 4] = [0.09, 0.0, 0.0, 0.09];
const ANISOTROPIC: [f64; 4] = [0.09, 0.03, 0.03, 0.16];

fn scale_of(sigma: [f64; 4]) -> Scale {
    if sigma == ISOTROPIC {
        Scale::sd(0.3)
    } else {
        Scale::covariance(&sigma, 2)
    }
}

/// The term of a normal on the sphere at `mu` with `sigma`, at each of `points`.
fn term_at(mu: [f64; 3], sigma: [f64; 4], points: &[[f64; 3]], target: &Target) -> Vec<f64> {
    let mut writer = Writer::new();
    let (x, term) = writer
        .build(|| {
            let x = quantity("x", Sphere);
            let mean = constant(Sphere, mu);
            (
                x,
                Normal::on(Sphere, mean, scale_of(sigma))
                    .log_density(x)
                    .log_density,
            )
        })
        .unwrap();
    let given = Observations::new().observe(x, Rows(points.to_vec()));
    evaluate(&writer.snapshot(), &[term.untyped()], given, target).remove(0)
}

skoro::precision_tests! {
    fn sphere_density_integrates_to_one(target) {
        // The midpoint rule in u is of second order: about 1e-6 at 2000 nodes.
        let relative = match target.precision() {
            Precision::Double => 1e-5,
            Precision::Single => 1e-4,
        };
        for mu in means() {
            for sigma in [ISOTROPIC, ANISOTROPIC] {
                let nodes = quadrature(mu, 2000, 64);
                let points: Vec<[f64; 3]> = nodes.iter().map(|(y, ..)| *y).collect();
                let terms = term_at(mu, sigma, &points, &target);
                let integral: f64 = nodes
                    .iter()
                    .zip(&terms)
                    .map(|((_, t, w), l)| (l + log_volume(*t)).exp() * w)
                    .sum();
                assert!((integral - 1.0).abs() < relative, "{mu:?} {sigma:?}: {integral}");
            }
        }
    }

    fn sphere_term_reads_the_declared_volume(target) {
        let relative = match target.precision() {
            Precision::Double => 1e-11,
            Precision::Single => 5e-5,
        };
        let mu = means()[1];
        let points: Vec<[f64; 3]> = [[0.2, -0.4], [1.5, 0.3], [-3.0, 2.0], [0.0, 0.0]]
            .iter()
            .map(|t| retract(mu, *t))
            .collect();
        let mut writer = Writer::new();
        let (x, term) = writer
            .build(|| {
                let x = quantity("x", Sphere);
                let mean = constant(Sphere, mu);
                (x, Normal::on(Sphere, mean, Scale::covariance(&ANISOTROPIC, 2)).score(x))
            })
            .unwrap();
        let snapshot = writer.snapshot();
        let lambda = snapshot.bound(term.id(), &roles::LOG_VOLUME).unwrap().unwrap();
        let given = Observations::new().observe(x, Rows(points.clone()));
        let values = evaluate(&snapshot, &[QDyn::from(term), lambda], given, &target);
        for (k, y) in points.iter().enumerate() {
            let t = local(mu, *y);
            let want = log_gaussian(t, ANISOTROPIC) - log_volume(t);
            assert!((values[0][k] - want).abs() < relative * want.abs().max(1.0), "{} against {want}", values[0][k]);
            let want = log_volume(t);
            assert!((values[1][k] - want).abs() < relative * want.abs().max(1.0), "{} against {want}", values[1][k]);
        }
    }

    fn sphere_sampler_matches_the_density(target) {
        let rows = 100_000;
        let mu = means()[0];
        let mut writer = Writer::new();
        let x = writer
            .build(|| {
                let x = quantity("x", Sphere);
                Normal::on(Sphere, constant(Sphere, mu), Scale::covariance(&ANISOTROPIC, 2)).score(x);
                x
            })
            .unwrap();
        let generated = writer.snapshot().generate(&Observations::new(), &Key::from_seed(21), rows, &target).unwrap();
        let draws = host(&generated.values, x.id());
        // Three bounded functions of a point: a coordinate, a square, and a bump at the mean.
        let functions: [Function; 3] = [
            |_, y| y[0],
            |_, y| y[1] * y[1],
            |mu, y| (-((y[0] - mu[0]).powi(2) + (y[1] - mu[1]).powi(2) + (y[2] - mu[2]).powi(2)) * 4.0).exp(),
        ];
        let nodes = quadrature(mu, 2000, 64);
        let points: Vec<[f64; 3]> = nodes.iter().map(|(y, ..)| *y).collect();
        let terms = term_at(mu, ANISOTROPIC, &points, &Target::host());
        for f in functions {
            let exact: f64 = nodes
                .iter()
                .zip(&terms)
                .map(|((y, t, w), l)| f(mu, *y) * (l + log_volume(*t)).exp() * w)
                .sum();
            let values: Vec<f64> = draws.chunks(3).map(|y| f(mu, [y[0], y[1], y[2]])).collect();
            let n = values.len() as f64;
            let mean = values.iter().sum::<f64>() / n;
            let se = (values.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / n / n).sqrt();
            assert!((mean - exact).abs() < 5.0 * se + 1e-9, "{mean} against {exact}, se {se}");
        }
    }
}

/// A bounded function of a point of the sphere, given the mean.
type Function = fn([f64; 3], [f64; 3]) -> f64;

/// `rows` copies of one row.
fn repeated(row: &[f64], rows: usize, target: &Target) -> Value {
    let data: Vec<f64> = row.iter().copied().cycle().take(rows * row.len()).collect();
    Value::Continuous(Points::from_host(&data, rows, row.len(), target))
}

/// The mean over draws of the negative chart Hessian of a sphere's normal over `(x, μ, θ)`, its
/// standard errors, and `CᵀC`, for a free isotropic scale, or a free covariance factor, at
/// `scale`.
fn information_at(scale: &[f64], rows: usize) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    let target = Target::host();
    let isotropic = scale.len() == 1;
    let mu_value = means()[1];
    let mut writer = Writer::new();
    let (x, mu, s, term) = writer
        .build(|| {
            let mu = quantity("mu", Sphere);
            let (s, form) = if isotropic {
                let s = quantity("scale", Positive::<1>);
                (s.untyped(), Scale::isotropic(s))
            } else {
                let s = quantity_in("scale", triangular(2));
                (s, Scale::covariance_factor(s))
            };
            let x = quantity("x", Sphere);
            let term = Normal::on(Sphere, mu, form).score(x);
            (x, mu, s, term.id())
        })
        .unwrap();
    let snapshot = writer.snapshot();
    let shared = |data: &[f64]| {
        HostValue::new(None, data.len(), HostData::Continuous(data.to_vec())).unwrap()
    };
    let given = Observations::new()
        .observe_dyn(mu.untyped(), shared(&mu_value))
        .observe_dyn(s, shared(scale));
    let draws = snapshot
        .generate(&given, &Key::from_seed(8), rows, &target)
        .unwrap();

    let root = snapshot
        .bound(term, &roles::INFORMATION_ROOT)
        .unwrap()
        .unwrap();
    let c = evaluate(&snapshot, &[root], given, &target).remove(0);
    let k = scale.len();
    let columns = 4 + k;
    let height = c.len() / columns;
    let ctc: Vec<f64> = (0..columns * columns)
        .map(|e| {
            (0..height)
                .map(|r| c[(e / columns) * height + r] * c[(e % columns) * height + r])
                .sum()
        })
        .collect();

    let program = snapshot
        .compile(&Query::log_density_of([term]), &target)
        .unwrap();
    let at = Values::new()
        .with(x, draws.values.get(x).unwrap().clone())
        .with(mu, repeated(&mu_value, rows, &target))
        .with(s, repeated(scale, rows, &target));
    let blocks = [(x.untyped(), 0, 2), (mu.untyped(), 2, 2), (s, 4, k)];
    let mut velocities = Frames::new();
    for &(q, offset, dim) in &blocks {
        let mut data = vec![0.0; rows * dim * columns];
        for row in 0..rows {
            for i in 0..dim {
                data[(row * dim + i) * columns + offset + i] = 1.0;
            }
        }
        velocities.insert(q, Frame::from_host(&data, rows, dim, columns, &target));
    }
    let second = program.hessian_products(&at, &velocities).unwrap();
    let (mut sum, mut squares) = (vec![0.0; columns * columns], vec![0.0; columns * columns]);
    for &(q, offset, dim) in &blocks {
        let beta = second.beta.get(q).unwrap().to_host();
        for row in 0..rows {
            for i in 0..dim {
                for j in 0..columns {
                    let h = -beta[(row * dim + i) * columns + j];
                    sum[(offset + i) * columns + j] += h;
                    squares[(offset + i) * columns + j] += h * h;
                }
            }
        }
    }
    let n = rows as f64;
    let mean: Vec<f64> = sum.iter().map(|s| s / n).collect();
    let se = squares
        .iter()
        .zip(&mean)
        .map(|(q, m)| ((q / n - m * m).max(0.0) / n).sqrt())
        .collect();
    (mean, se, ctc)
}

#[test]
fn sphere_information() {
    for (scale, sd) in [
        (vec![0.3], 0.3),
        (vec![0.1], 0.1),
        (vec![0.2, 0.3, 0.05], 0.3),
    ] {
        let k = scale.len();
        let columns = 4 + k;
        let (expected, se, ctc) = information_at(&scale, 50_000);
        // The scale's block is exact: within the sampling error of CᵀC.
        for i in 4..columns {
            for j in 4..columns {
                let e = i * columns + j;
                assert!(
                    (expected[e] - ctc[e]).abs() <= 5.0 * se[e] + 1e-9,
                    "θ block {expected:?} against {ctc:?}"
                );
            }
        }
        // The blocks of x and μ are ±Λ to second order in the dispersion.
        let lambda = ctc[0].abs().max(ctc[columns + 1].abs());
        for i in 0..4 {
            for j in 0..4 {
                let e = i * columns + j;
                let error = (expected[e] - ctc[e]).abs() - 5.0 * se[e];
                assert!(
                    error <= 2.0 * sd * sd * lambda,
                    "σ = {sd}: {expected:?} against {ctc:?}"
                );
            }
        }
    }
}
