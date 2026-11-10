#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)] // a test crate
//! The Levenberg–Marquardt method, apart from the examples, on the crate's own models.

mod common;

use common::{close, gradient, gram, host, points, rotations, solve};
use skoro_core::scope::{constant, quantity};
use skoro_core::{NodeId, Observations, Query, Real, Values, Writer};
use skoro_elimination::Damping;
use skoro_linearize::{Assembly, Error, LevenbergMarquardt, Linearizer, Status, decrement};
use skoro_normal::{Normal, Scale};
use skoro_tensor::Target;

#[test]
fn the_objective_is_the_negative_log_density() {
    let target = Target::host();
    let (writer, query, point) = rotations(&target);
    let snapshot = writer.snapshot();
    let linearizer = Linearizer::new(&snapshot, &query, &target, Assembly::Residual).unwrap();
    let objective = linearizer.objective(&point).unwrap();
    assert_eq!(objective.len(), 2);
    let program = snapshot.compile(&query, &target).unwrap();
    let density: Vec<f64> = host(&program.forward(&point).unwrap(), NodeId::DENSITY)
        .into_iter()
        .map(|l| -l)
        .collect();
    close(&objective, &density, 1e-14, "the query's own program");
    let zero = vec![0.0; linearizer.layout().dimension()];
    let at_zero: Vec<f64> = linearizer
        .at(&point)
        .unwrap()
        .problems()
        .iter()
        .map(|p| p.value(&zero))
        .collect();
    close(&objective, &at_zero, 1e-12, "the linearization at zero");
}

#[test]
fn the_fisher_assembly_is_refused() {
    let target = Target::host();
    let (writer, query, point) = rotations(&target);
    let linearizer =
        Linearizer::new(&writer.snapshot(), &query, &target, Assembly::Fisher).unwrap();
    let schedule = linearizer.schedule().unwrap();
    let refused = LevenbergMarquardt::default()
        .minimize(&linearizer, &schedule, &point)
        .unwrap_err();
    assert_eq!(refused, Error::NotAnIntegral);
}

#[test]
fn non_convergence_is_an_outcome() {
    let target = Target::host();
    let (writer, query, point) = rotations(&target);
    let linearizer =
        Linearizer::new(&writer.snapshot(), &query, &target, Assembly::Residual).unwrap();
    let schedule = linearizer.schedule().unwrap();
    let one = LevenbergMarquardt {
        max_iterations: 1,
        ..LevenbergMarquardt::default()
    }
    .minimize(&linearizer, &schedule, &point)
    .unwrap();
    for row in &one.rows {
        assert_eq!(row.status, Status::IterationsExceeded);
        assert_eq!(row.iterations, 1);
    }
    let stiff = LevenbergMarquardt {
        max_damping: 1e-4,
        ..LevenbergMarquardt::default()
    }
    .minimize(&linearizer, &schedule, &point)
    .unwrap();
    for row in &stiff.rows {
        assert_eq!(row.status, Status::DampingExceeded);
        assert_eq!(row.iterations, 0);
    }
}

#[test]
fn levenberg_marquardt_finds_the_mode_on_rotations() {
    let target = Target::host();
    let (writer, query, point) = rotations(&target);
    let snapshot = writer.snapshot();
    let linearizer = Linearizer::new(&snapshot, &query, &target, Assembly::Residual).unwrap();
    let layout = linearizer.layout().clone();
    let d = layout.dimension();
    let schedule = linearizer.schedule().unwrap();
    let method = LevenbergMarquardt::default();
    let bound = 10.0 * method.tolerance;
    let minimized = method.minimize(&linearizer, &schedule, &point).unwrap();
    for row in &minimized.rows {
        assert_eq!(row.status, Status::Converged, "{row:?}");
    }
    let decrements = decrement(&linearizer, &schedule, &minimized.point).unwrap();
    for &delta in &decrements {
        assert!(delta <= bound, "decrement {delta}");
    }
    // The gradient a decrement of 10 ε allows: |g|² ≤ 2 · 10ε · tr(JᵀJ), since the decrement is
    // at least |g|² / (2 h_max) and h_max is at most the trace.
    let linearized = linearizer.at(&minimized.point).unwrap();
    let gradients = gradient(&snapshot, &query, &layout, &minimized.point, &target);
    for (g, problem) in gradients.iter().zip(linearized.problems()) {
        let jtj = gram(problem);
        let trace: f64 = (0..d).map(|i| jtj[i][i]).sum();
        let squared: f64 = g.iter().map(|g| g * g).sum();
        assert!(
            squared <= 2.0 * bound * trace,
            "|g|² = {squared} against {}",
            2.0 * bound * trace
        );
    }
    // The undamped loop of `steps_converge_on_rotations` reaches a mode of its own decrement at
    // most 10 ε, and the two are within 40 ε of each other in the metric of J.
    let mut at = point;
    for _ in 0..60 {
        let step = linearizer
            .at(&at)
            .unwrap()
            .eliminate(&schedule, None)
            .unwrap()
            .minimizer(&target)
            .unwrap();
        at = layout.retract(&at, &step, &target).unwrap();
    }
    for delta in decrement(&linearizer, &schedule, &at).unwrap() {
        assert!(delta <= bound, "the loop's decrement {delta}");
    }
    let tau = layout
        .local(&at, &minimized.point, &target)
        .unwrap()
        .to_host();
    let zero = Damping {
        lambda: 0.0,
        diagonal: vec![0.0; d],
    };
    for (r, problem) in linearized.problems().iter().enumerate() {
        let apart = problem.promised(&tau[r * d..(r + 1) * d], &zero);
        assert!(apart <= 4.0 * bound, "row {r}: ½|Jτ|² = {apart}");
    }
}

#[test]
fn levenberg_marquardt_finds_the_linear_gaussian_mean() {
    let target = Target::host();
    // x₀ ~ N(0.3, 1), x₁ ~ N(x₀, 0.5²), x₂ ~ N(x₁, 0.4²), and z = 1.2 observed, z ~ N(x₂, 0.2²).
    let mut writer = Writer::new();
    let (x, z) = writer
        .build(|| {
            let x0 = quantity("x0", Real);
            Normal::on(Real, constant(Real, 0.3), Scale::sd(1.0)).score(x0);
            let x1 = quantity("x1", Real);
            Normal::on(Real, x0, Scale::sd(0.5)).score(x1);
            let x2 = quantity("x2", Real);
            Normal::on(Real, x1, Scale::sd(0.4)).score(x2);
            let z = quantity("z", Real);
            Normal::on(Real, x2, Scale::sd(0.2)).score(z);
            ([x0, x1, x2], z)
        })
        .unwrap();
    let query = Query::log_density().given(Observations::new().observe(z, 1.2));
    let linearizer =
        Linearizer::new(&writer.snapshot(), &query, &target, Assembly::Residual).unwrap();
    let schedule = linearizer.schedule().unwrap();
    let start = Values::new()
        .with(x[0], points(&[-2.0], 1, 1, &target))
        .with(x[1], points(&[3.0], 1, 1, &target))
        .with(x[2], points(&[0.0], 1, 1, &target));
    let method = LevenbergMarquardt::default();
    let minimized = method.minimize(&linearizer, &schedule, &start).unwrap();
    assert_eq!(minimized.rows[0].status, Status::Converged);
    // The posterior's precision Λ = JᵀJ and mean μ, from the whitened residuals.
    let j = [
        [1.0, 0.0, 0.0],
        [-2.0, 2.0, 0.0],
        [0.0, -2.5, 2.5],
        [0.0, 0.0, 5.0],
    ];
    let b = [0.3, 0.0, 0.0, 6.0];
    let precision: Vec<Vec<f64>> = (0..3)
        .map(|r| {
            (0..3)
                .map(|c| (0..4).map(|i| j[i][r] * j[i][c]).sum())
                .collect()
        })
        .collect();
    let jtb: Vec<f64> = (0..3)
        .map(|r| (0..4).map(|i| j[i][r] * b[i]).sum())
        .collect();
    let mean = solve(&precision, &jtb);
    // f(x) - f(μ) = ½ (x - μ)ᵀ Λ (x - μ) exactly in a linear model: the tolerance itself, in nats.
    let mode: Vec<f64> = x.iter().flat_map(|&q| host(&minimized.point, q)).collect();
    let e: Vec<f64> = mode.iter().zip(&mean).map(|(x, m)| x - m).collect();
    let excess: f64 = 0.5
        * (0..3)
            .map(|r| (0..3).map(|c| e[r] * precision[r][c] * e[c]).sum::<f64>())
            .sum::<f64>();
    assert!(excess <= 10.0 * method.tolerance, "excess {excess}");
}
