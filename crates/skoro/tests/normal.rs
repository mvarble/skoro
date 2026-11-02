#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)] // a test crate
//! The normal on curved types: on the acceptance-test sphere, whose stereographic chart is global
//! and full, its density, term, sampler and information.

mod common;

use std::f64::consts::PI;

use common::sphere::{Sphere, basis};
use skoro::scope::{constant, quantity, quantity_in};
use skoro::tensor::{Frame, HostData, HostValue, Points, Value};
use skoro::{
    Declares, Frames, Key, NodeId, Observations, Positive, Precision, QDyn, Query, Rows, Snapshot,
    Target, Values, Writer,
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

/// One Gauss–Newton step's ingredients at `x`: the stacked residuals, their Jacobian (rows of
/// residual entries, two columns), and the gradient of the log-volumes.
struct Linearized {
    residual: Vec<f64>,
    jacobian: Vec<[f64; 2]>,
    log_volume_gradient: [f64; 2],
}

/// Gauss–Newton on `x`, from `start`, with or without the log-volumes' gradient: the iterates.
fn gauss_newton(
    program: &skoro::Program,
    x: QDyn,
    residuals: &[QDyn],
    log_volumes: &[QDyn],
    start: [f64; 3],
    with_log_volume: bool,
) -> Vec<[f64; 3]> {
    let target = Target::host();
    let linearize = |at: [f64; 3]| -> Linearized {
        let values =
            Values::new().with(x, Value::Continuous(Points::from_host(&at, 1, 3, &target)));
        let identity = Frame::from_host(&[1.0, 0.0, 0.0, 1.0], 1, 2, 2, &target);
        let (out, frames) = program
            .tangent(&values, &Frames::new().with(x, identity))
            .unwrap();
        let mut residual = Vec::new();
        let mut jacobian = Vec::new();
        for r in residuals {
            residual.extend(host(&out, r.id()));
            let frame = frames.get(r.id()).unwrap().to_host();
            jacobian.extend(frame.chunks(2).map(|row| [row[0], row[1]]));
        }
        let mut log_volume_gradient = [0.0; 2];
        for l in log_volumes {
            let frame = frames.get(l.id()).unwrap().to_host();
            log_volume_gradient[0] += frame[0];
            log_volume_gradient[1] += frame[1];
        }
        Linearized {
            residual,
            jacobian,
            log_volume_gradient,
        }
    };
    let mut iterates = vec![start];
    let mut at = start;
    for _ in 0..12 {
        let step = linearize(at);
        // f = ½|r|² + Σ λ: the gradient Jᵀr + ∇λ, and the matrix JᵀJ.
        let mut g = [0.0; 2];
        let mut h = [0.0; 4];
        for (row, r) in step.jacobian.iter().zip(&step.residual) {
            g[0] += row[0] * r;
            g[1] += row[1] * r;
            h[0] += row[0] * row[0];
            h[1] += row[0] * row[1];
            h[3] += row[1] * row[1];
        }
        h[2] = h[1];
        if with_log_volume {
            g[0] += step.log_volume_gradient[0];
            g[1] += step.log_volume_gradient[1];
        }
        let det = h[0] * h[3] - h[1] * h[2];
        let delta = [
            -(h[3] * g[0] - h[1] * g[1]) / det,
            -(h[0] * g[1] - h[2] * g[0]) / det,
        ];
        at = retract(at, delta);
        iterates.push(at);
    }
    iterates
}

#[test]
fn gauss_newton_reaches_the_model_mode() {
    let target = Target::host();
    let (sd1, sd2) = (0.2, 0.4);
    let z1 = normalize([0.1, 0.2, 0.97]);
    let z2 = retract(z1, [0.5, 0.1]);
    let mut writer = Writer::new();
    let (x, z, terms) = writer
        .build(|| {
            let x = quantity("x", Sphere);
            let z = [quantity("z1", Sphere), quantity("z2", Sphere)];
            let terms = [
                Normal::on(Sphere, x, Scale::sd(sd1)).score(z[0]).id(),
                Normal::on(Sphere, x, Scale::sd(sd2)).score(z[1]).id(),
            ];
            (x, z, terms)
        })
        .unwrap();
    let snapshot = writer.snapshot();
    let given = Observations::new().observe(z[0], z1).observe(z[1], z2);
    let residuals: Vec<QDyn> = terms
        .iter()
        .map(|t| snapshot.bound(*t, &roles::RESIDUAL).unwrap())
        .collect();
    let log_volumes: Vec<QDyn> = terms
        .iter()
        .map(|t| snapshot.bound(*t, &roles::LOG_VOLUME).unwrap().unwrap())
        .collect();
    let nodes: Vec<QDyn> = residuals.iter().chain(&log_volumes).copied().collect();
    let program = snapshot
        .compile(&Query::of(nodes).given(given.clone()), &target)
        .unwrap();

    let model = gauss_newton(&program, x.untyped(), &residuals, &log_volumes, z1, true);
    let squares = gauss_newton(&program, x.untyped(), &residuals, &log_volumes, z1, false);
    let (mode, residual_mode) = (model[model.len() - 1], squares[squares.len() - 1]);

    // The model's mode by a dense search of the full log-density, in the chart at the iterate.
    let density = snapshot
        .compile(&Query::log_density().given(given), &target)
        .unwrap();
    let (half, n) = (0.01, 201);
    let spacing = 2.0 * half / (n - 1) as f64;
    let grid: Vec<[f64; 2]> = (0..n * n)
        .map(|k| {
            [
                -half + spacing * (k / n) as f64,
                -half + spacing * (k % n) as f64,
            ]
        })
        .collect();
    let points: Vec<f64> = grid.iter().flat_map(|t| retract(mode, *t)).collect();
    let values = Values::new().with(
        x,
        Value::Continuous(Points::from_host(&points, grid.len(), 3, &target)),
    );
    let densities = host(&density.forward(&values).unwrap(), NodeId::DENSITY);
    let best = (0..grid.len())
        .max_by(|&a, &b| densities[a].total_cmp(&densities[b]))
        .unwrap();
    let norm = |t: [f64; 2]| (t[0] * t[0] + t[1] * t[1]).sqrt();
    assert!(
        norm(grid[best]) <= spacing,
        "the dense search's mode is {:?} from the iterate",
        grid[best]
    );

    // The residuals' own mode is another point.
    let apart = norm(local(mode, residual_mode));
    assert!(apart > 10.0 * spacing, "the modes are {apart} apart");

    // The rates: the log-volume adds at most κ = σ² to the residuals' own.
    let rate = |iterates: &[[f64; 3]], limit: [f64; 3]| {
        let errors: Vec<f64> = iterates.iter().map(|x| norm(local(limit, *x))).collect();
        (1..4)
            .map(|k| errors[k + 1] / errors[k])
            .fold(0.0, f64::max)
    };
    let (with, without) = (rate(&model, mode), rate(&squares, residual_mode));
    let kappa = sd2 * sd2;
    assert!(
        with <= without + kappa,
        "rates {with} with the log-volume and {without} without"
    );
}

#[test]
fn refuses_a_chart_that_is_not_global() {
    use common::sphere::Exponential;
    let failure = Writer::new()
        .build(|| {
            let x = quantity("x", Exponential);
            let mean = constant(Exponential, means()[1]);
            Normal::on(Exponential, mean, Scale::sd(0.3)).score(x);
        })
        .unwrap_err();
    assert_eq!(
        failure.error,
        skoro::Error::SpaceMismatch {
            operation: "skoro.normal",
            inputs: vec![Exponential.space()]
        }
    );
    // The Gaussian part at a measured value is a likelihood factor, and is accepted.
    let mut writer = Writer::new();
    writer
        .build(|| {
            let x = quantity("x", Exponential);
            let z = constant(Exponential, means()[1]);
            Normal::on(Exponential, x, Scale::sd(0.3)).factor_at(z);
        })
        .unwrap();
}

#[test]
fn refuses_a_type_without_a_volume() {
    use common::sphere::Unmeasured;
    let failure = Writer::new()
        .build(|| {
            let x = quantity("x", Unmeasured);
            let mean = constant(Unmeasured, means()[1]);
            Normal::on(Unmeasured, mean, Scale::sd(0.3)).score(x);
        })
        .unwrap_err();
    assert!(
        matches!(&failure.error, skoro::Error::NoVolume { space, .. } if *space == Unmeasured.space()),
        "{failure:?}"
    );

    // A measurement over data: the factor's gradient in the mean is that of -½|W (z ⊖ μ)|².
    let (z, sd) = (means()[1], 0.3);
    let mut writer = Writer::new();
    let (mu, factor) = writer
        .build(|| {
            let mu = quantity("mu", Unmeasured);
            let factor =
                Normal::on(Unmeasured, mu, Scale::sd(sd)).factor_at(constant(Unmeasured, z));
            (mu, factor.id())
        })
        .unwrap();
    let target = Target::host();
    let program = writer
        .snapshot()
        .compile(&Query::log_density_of([factor]), &target)
        .unwrap();
    let at_point = retract(z, [0.2, -0.15]);
    let at = Values::new().with(
        mu,
        Value::Continuous(Points::from_host(&at_point, 1, 3, &target)),
    );
    let (_, frames) = program.gradient(&at).unwrap();
    let gradient = frames.get(mu).unwrap().to_host();
    let f = |m: [f64; 3]| {
        let t = local(m, z);
        -0.5 * (t[0] * t[0] + t[1] * t[1]) / (sd * sd)
    };
    let h = 1e-6;
    for i in 0..2 {
        let mut e = [0.0; 2];
        e[i] = h;
        let plus = f(retract(at_point, e));
        e[i] = -h;
        let minus = f(retract(at_point, e));
        let want = (plus - minus) / (2.0 * h);
        assert!(
            (gradient[i] - want).abs() < 1e-7 * want.abs().max(1.0),
            "{gradient:?} against {want}"
        );
    }
}

#[test]
fn stacked_tangent_on_the_sphere() {
    use skoro_normal::Stacked;
    let target = Target::host();
    let anchor = means()[1];
    let (point, real) = (retract(anchor, [0.3, -0.2]), 1.1);
    const SIGMA: [f64; 9] = [0.5, 0.1, 0.0, 0.1, 0.4, 0.05, 0.0, 0.05, 0.3];
    let m = [0.05, -0.1, 0.2];
    let mut writer = Writer::new();
    let (s, r, delta, factor) = writer
        .build(|| {
            let s = quantity("s", Sphere);
            let r = quantity("r", skoro::Real);
            let stacked = Stacked::new()
                .block(s, Sphere, anchor)
                .block(r, skoro::Real, 0.4);
            let factor = stacked.factor(&m, Scale::covariance(&SIGMA, 3), 0.0);
            (s, r, stacked.tangent(), factor.id())
        })
        .unwrap();
    let snapshot = writer.snapshot();
    let residual = snapshot.bound(factor, &roles::RESIDUAL).unwrap();
    let given = Observations::new().observe(s, point).observe(r, real);
    let values = evaluate(&snapshot, &[delta.untyped(), residual], given, &target);

    // δ is the tangents laid end to end.
    let t = local(anchor, point);
    let want = [t[0], t[1], real - 0.4];
    for (got, want) in values[0].iter().zip(want) {
        assert!(
            (got - want).abs() < 1e-12,
            "{:?} against {want:?}",
            values[0]
        );
    }
    // Its factor's residual is W (δ - m), W = L⁻¹ for the Cholesky factor L of Σ.
    let l = [
        SIGMA[0].sqrt(),
        SIGMA[3] / SIGMA[0].sqrt(),
        (SIGMA[4] - SIGMA[3] * SIGMA[3] / SIGMA[0]).sqrt(),
    ];
    let l20 = SIGMA[6] / l[0];
    let l21 = (SIGMA[7] - l20 * l[1]) / l[2];
    let l22 = (SIGMA[8] - l20 * l20 - l21 * l21).sqrt();
    let d = [want[0] - m[0], want[1] - m[1], want[2] - m[2]];
    let z0 = d[0] / l[0];
    let z1 = (d[1] - l[1] * z0) / l[2];
    let z2 = (d[2] - l20 * z0 - l21 * z1) / l22;
    for (got, want) in values[1].iter().zip([z0, z1, z2]) {
        assert!(
            (got - want).abs() < 1e-12,
            "{:?} against {want:?}",
            values[1]
        );
    }

    // A discrete block is refused.
    let failure = Writer::new()
        .build(|| {
            let k = quantity("k", skoro::Categorical::<3>);
            Stacked::new()
                .block(k, skoro::Categorical::<3>, 1usize)
                .tangent();
        })
        .unwrap_err();
    assert!(
        matches!(failure.error, skoro::Error::NotContinuous { .. }),
        "{failure:?}"
    );
}
