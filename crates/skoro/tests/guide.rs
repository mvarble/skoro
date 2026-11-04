#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)] // a test crate
//! The guide's first model: a hierarchical normal, a point `x` of the plane and its squared range
//! `ρ`, written with `sample` and with `quantity` and `score`, generated, and queried with `ρ`
//! observed.

use std::f64::consts::PI;

use skoro::scope::{constant, quantity, sample};
use skoro::tensor::Value;
use skoro::testing::Tolerance;
use skoro::{
    IntoHostValue, Key, NodeId, Observations, Q, Query, Real, Rows, Snapshot, Target, Values,
    Vector, Writer,
};
use skoro_normal::{Normal, Scale};

/// The standard deviations of `x` and of `ρ` given `x`.
const SIGMA0: f64 = 1.5;
const SIGMA1: f64 = 0.5;

/// `x ~ N(0, σ₀² I)` on the plane, and `ρ ~ N(|x|², σ₁²)`.
fn model() -> (Snapshot, Q<Vector<2>>, Q<Real>) {
    let mut writer = Writer::new();
    let (x, rho) = writer
        .build(|| {
            let origin = constant(Vector::<2>, [0.0, 0.0]);
            let x = sample(
                "x",
                Vector::<2>,
                Normal::on(Vector::<2>, origin, Scale::sd(SIGMA0)),
            );
            let rho = sample(
                "rho",
                Real,
                Normal::on(Real, (x * x).sum(), Scale::sd(SIGMA1)),
            );
            (x, rho)
        })
        .unwrap();
    (writer.snapshot(), x, rho)
}

/// The same model, each quantity made and then scored.
fn model_by_quantity_and_score() -> (Snapshot, Q<Vector<2>>, Q<Real>) {
    let mut writer = Writer::new();
    let (x, rho) = writer
        .build(|| {
            let origin = constant(Vector::<2>, [0.0, 0.0]);
            let x = quantity("x", Vector::<2>);
            Normal::on(Vector::<2>, origin, Scale::sd(SIGMA0)).score(x);
            let rho = quantity("rho", Real);
            Normal::on(Real, (x * x).sum(), Scale::sd(SIGMA1)).score(rho);
            (x, rho)
        })
        .unwrap();
    (writer.snapshot(), x, rho)
}

fn host(values: &Values, q: impl Into<NodeId>) -> Vec<f64> {
    match values.get(q).unwrap() {
        Value::Continuous(points) => points.to_host(),
        Value::Discrete(_) => panic!("a continuous value"),
    }
}

/// `x` at `points`, one row each.
fn at(x: Q<Vector<2>>, points: &[[f64; 2]], target: &Target) -> Values {
    let value = IntoHostValue::<Vector<2>>::into_host_value(Rows(points.to_vec()))
        .unwrap()
        .load(points.len(), target)
        .unwrap()
        .unwrap();
    Values::new().with(x, value)
}

/// Each coordinate of `data`, rows of `d`, of mean zero and variance `sigma²`, within five
/// standard errors.
fn centred(data: &[f64], d: usize, sigma: f64) {
    let n = (data.len() / d) as f64;
    for i in 0..d {
        let column: Vec<f64> = data.iter().skip(i).step_by(d).copied().collect();
        let mean = column.iter().sum::<f64>() / n;
        let variance = column.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / n;
        assert!(mean.abs() < 5.0 * sigma / n.sqrt(), "mean {mean}");
        let se = sigma * sigma * (2.0 / n).sqrt();
        assert!(
            (variance - sigma * sigma).abs() < 5.0 * se,
            "variance {variance}"
        );
    }
}

fn close(got: &[f64], want: &[f64], target: &Target) {
    let relative = Tolerance::of(target.precision()).relative;
    assert_eq!(got.len(), want.len());
    for (g, w) in got.iter().zip(want) {
        assert!(
            (g - w).abs() <= relative * w.abs().max(1.0),
            "{got:?} against {want:?}"
        );
    }
}

const POINTS: [[f64; 2]; 3] = [[0.3, -0.8], [1.2, 0.5], [-0.4, 0.1]];
const RHO: f64 = 0.9;

/// `log N(x; 0, σ₀² I) + log N(ρ; |x|², σ₁²)`.
fn log_density(x: [f64; 2]) -> f64 {
    let r = x[0] * x[0] + x[1] * x[1];
    let prior = -r / (2.0 * SIGMA0 * SIGMA0) - 2.0 * SIGMA0.ln() - (2.0 * PI).ln();
    let range = -(RHO - r).powi(2) / (2.0 * SIGMA1 * SIGMA1) - SIGMA1.ln() - 0.5 * (2.0 * PI).ln();
    prior + range
}

/// Its gradient in `x`: `-x / σ₀² + 2x (ρ - |x|²) / σ₁²`.
fn gradient(x: [f64; 2]) -> [f64; 2] {
    let r = x[0] * x[0] + x[1] * x[1];
    x.map(|xi| -xi / (SIGMA0 * SIGMA0) + 2.0 * xi * (RHO - r) / (SIGMA1 * SIGMA1))
}

skoro::precision_tests! {
    fn hierarchical_normal_generates_the_joint(target) {
        let (snapshot, x, rho) = model();
        let generated = snapshot
            .generate(&Observations::new(), &Key::from_seed(11), 4000, &target)
            .unwrap();
        let (xs, rhos) = (host(&generated.values, x), host(&generated.values, rho));
        // Nothing is observed and both samplers are exact, so every draw weighs the same.
        assert!(generated.log_weight.to_host().iter().all(|w| *w == 0.0));

        let noise: Vec<f64> = xs
            .chunks(2)
            .zip(&rhos)
            .map(|(x, rho)| rho - (x[0] * x[0] + x[1] * x[1]))
            .collect();
        centred(&xs, 2, SIGMA0);
        centred(&noise, 1, SIGMA1);
    }

    fn hierarchical_normal_density_and_gradient(target) {
        let (snapshot, x, rho) = model();
        let query = Query::log_density().given(Observations::new().observe(rho, RHO));
        let program = snapshot.compile(&query, &target).unwrap();
        let (density, grad) = program.gradient(&at(x, &POINTS, &target)).unwrap();

        let want: Vec<f64> = POINTS.iter().map(|&p| log_density(p)).collect();
        close(&host(&density, NodeId::DENSITY), &want, &target);
        let want: Vec<f64> = POINTS.iter().flat_map(|&p| gradient(p)).collect();
        close(&grad.get(x).unwrap().to_host(), &want, &target);
    }

    fn hierarchical_normal_by_quantity_and_score(target) {
        let density = |(snapshot, x, rho): (Snapshot, Q<Vector<2>>, Q<Real>)| {
            let query = Query::log_density().given(Observations::new().observe(rho, RHO));
            let program = snapshot.compile(&query, &target).unwrap();
            host(&program.forward(&at(x, &POINTS, &target)).unwrap(), NodeId::DENSITY)
        };
        close(&density(model_by_quantity_and_score()), &density(model()), &target);
    }
}
