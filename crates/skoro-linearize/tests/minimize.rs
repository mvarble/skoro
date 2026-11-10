#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)] // a test crate
//! The Levenberg–Marquardt method, apart from the examples, on the crate's own models.

mod common;

use common::{close, gradient, gram, host, points, rotations, row, solve};
use skoro_core::scope::{constant, quantity};
use skoro_core::{NodeId, Observations, Query, Real, Values, Vector, Writer};
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

/// Two iterates equal to rounding.
fn same(a: &skoro_linearize::Iterate, b: &skoro_linearize::Iterate) -> bool {
    let near = |x: f64, y: f64| (x - y).abs() <= 1e-9 * x.abs().max(y.abs()).max(1e-12);
    a.accepted == b.accepted
        && near(a.objective, b.objective)
        && near(a.damping, b.damping)
        && near(a.promised, b.promised)
}

#[test]
fn rows_are_minimized_independently() {
    let target = Target::host();
    let (writer, query, point) = rotations(&target);
    let linearizer =
        Linearizer::new(&writer.snapshot(), &query, &target, Assembly::Residual).unwrap();
    let layout = linearizer.layout().clone();
    let schedule = linearizer.schedule().unwrap();
    let method = LevenbergMarquardt::default();
    let batch = method.minimize(&linearizer, &schedule, &point).unwrap();
    assert_eq!(batch.rows.len(), 2);
    assert_ne!(batch.rows[0].iterations, 0);
    for r in 0..2 {
        let mut alone = Values::new();
        for &q in layout.quantities() {
            let data = row(point.get(q).unwrap(), r);
            alone.insert(q, points(&data, 1, data.len(), &target));
        }
        let single = method.minimize(&linearizer, &schedule, &alone).unwrap();
        let (a, b) = (&batch.rows[r], &single.rows[0]);
        assert_eq!(a.status, b.status, "row {r}");
        assert_eq!(a.history.len(), b.history.len(), "row {r}");
        for (x, y) in a.history.iter().zip(&b.history) {
            assert!(same(x, y), "row {r}: {x:?} against {y:?}");
        }
        for &q in layout.quantities() {
            close(
                &row(batch.point.get(q).unwrap(), r),
                &host(&single.point, q),
                1e-12,
                "the row's mode",
            );
        }
    }
}

#[test]
fn a_rejected_step_keeps_the_point() {
    let target = Target::host();
    // x ~ N(0, 1) and z ~ N(x³, 0.1²) observed at 1: from x = 0.1, where x³ is flat, an undamped
    // step overshoots far past the mode.
    let mut writer = Writer::new();
    let (x, z) = writer
        .build(|| {
            let x = quantity("x", Real);
            Normal::on(Real, constant(Real, 0.0), Scale::sd(1.0)).score(x);
            let z = quantity("z", Real);
            Normal::on(Real, x * x * x, Scale::sd(0.1)).score(z);
            (x, z)
        })
        .unwrap();
    let query = Query::log_density().given(Observations::new().observe(z, 1.0));
    let linearizer =
        Linearizer::new(&writer.snapshot(), &query, &target, Assembly::Residual).unwrap();
    let schedule = linearizer.schedule().unwrap();
    let start = Values::new().with(x, points(&[0.1], 1, 1, &target));
    let minimized = LevenbergMarquardt {
        initial_damping: 1e-8,
        ..LevenbergMarquardt::default()
    }
    .minimize(&linearizer, &schedule, &start)
    .unwrap();
    let row = &minimized.rows[0];
    assert_eq!(row.status, Status::Converged, "{row:?}");
    let history = &row.history;
    assert!(history.iter().any(|it| !it.accepted), "{history:?}");
    let mut beta = 2.0;
    for pair in history.windows(2) {
        let (now, next) = (&pair[0], &pair[1]);
        if now.accepted {
            beta = 2.0;
        } else {
            // The point is kept, and the damping grows by β, which doubles.
            assert_eq!(next.objective, now.objective, "{history:?}");
            assert!(
                (next.damping - beta * now.damping).abs() <= 1e-15 * next.damping,
                "{history:?}"
            );
            beta *= 2.0;
        }
    }
}

#[test]
fn rows_at_rounding_converge() {
    let target = Target::host();
    // x ~ N(0, I) on R³ and 2 000 observations z_i ~ N(x + 0.01 x ⊙ x, I): |f| in the thousands, and
    // nearly linear, so that Gauss–Newton contracts fast and the step after one promising just over
    // ε promises little more than the rounding of f.
    let n = 2000;
    let truth = [0.4, -0.7, 1.1];
    let mut state: u64 = 0x2026_1110;
    let mut uniform = || {
        state = state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        ((state >> 11) as f64 + 0.5) / (1u64 << 53) as f64
    };
    let mut normal = || {
        let (u, v) = (uniform(), uniform());
        (-2.0 * u.ln()).sqrt() * (2.0 * std::f64::consts::PI * v).cos()
    };
    let observed: Vec<[f64; 3]> = (0..n)
        .map(|_| std::array::from_fn(|i| truth[i] + 0.01 * truth[i] * truth[i] + normal()))
        .collect();
    let mut writer = Writer::new();
    let (x, zs) = writer
        .build(|| {
            let x = quantity("x", Vector::<3>);
            Normal::on(Vector::<3>, constant(Vector::<3>, [0.0; 3]), Scale::sd(1.0)).score(x);
            let mean = x + x * x * 0.01;
            let zs: Vec<_> = (0..n)
                .map(|i| {
                    let z = quantity(format!("z/{i}").as_str(), Vector::<3>);
                    Normal::on(Vector::<3>, mean, Scale::sd(1.0)).score(z);
                    z
                })
                .collect();
            (x, zs)
        })
        .unwrap();
    let given = zs
        .iter()
        .zip(&observed)
        .fold(Observations::new(), |o, (&z, v)| o.observe(z, *v));
    let query = Query::log_density().given(given);
    let linearizer =
        Linearizer::new(&writer.snapshot(), &query, &target, Assembly::Residual).unwrap();
    let schedule = linearizer.schedule().unwrap();
    let method = LevenbergMarquardt::default();
    // The mode, from one row; then 1 000 rows of starts near it.
    let one = method
        .minimize(
            &linearizer,
            &schedule,
            &Values::new().with(x, points(&[0.0; 3], 1, 3, &target)),
        )
        .unwrap();
    let mode = host(&one.point, x);
    assert!(linearizer.objective(&one.point).unwrap()[0].abs() > 1000.0);
    let rows = 1000;
    let starts: Vec<f64> = (0..rows)
        .flat_map(|_| {
            let offsets: [f64; 3] = std::array::from_fn(|_| 1e-3 * normal());
            [
                mode[0] + offsets[0],
                mode[1] + offsets[1],
                mode[2] + offsets[2],
            ]
        })
        .collect();
    let minimized = method
        .minimize(
            &linearizer,
            &schedule,
            &Values::new().with(x, points(&starts, rows, 3, &target)),
        )
        .unwrap();
    for (r, outcome) in minimized.rows.iter().enumerate() {
        assert_eq!(outcome.status, Status::Converged, "row {r}: {outcome:?}");
        let last = outcome.history.last().unwrap();
        assert!(last.promised <= method.tolerance, "row {r}: {last:?}");
    }
}
