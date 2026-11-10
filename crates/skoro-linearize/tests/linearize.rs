#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)] // a test crate
#![allow(clippy::needless_range_loop)] // dense linear algebra, by index
//! The linearization of a query, against the query itself and the statements of its mathematics.

mod common;

use common::{close, f_at, gradient, gram, points, row, solve, tolerance};
use skoro_core::scope::{constant, quantity};
use skoro_core::{Matrix, Observations, Positive, Query, Real, Values, Vector, Writer};
use skoro_elimination::{Damping, Problem, Schedule};
use skoro_linearize::{Assembly, Linearizer};
use skoro_normal::{Normal, Scale};
use skoro_tensor::Target;

/// A linear-Gaussian model of vectors: `x ~ N(x₀, diag(1, 2)²)`, `y ~ N(M x + c, Σ)`, and an
/// observed `z ~ N(y, 0.5²)`, with its query and a batch of two points that are not its mode.
fn vectors(target: &Target) -> (Writer, Query, Values) {
    let mut writer = Writer::new();
    let (x, y, z) = writer
        .build(|| {
            let x = quantity("x", Vector::<2>);
            Normal::on(
                Vector::<2>,
                constant(Vector::<2>, [0.2, -0.1]),
                Scale::sds(&[1.0, 2.0]),
            )
            .score(x);
            let y = quantity("y", Vector::<2>);
            let m = constant(Matrix::<2, 2>, [[1.0, 0.5], [-0.3, 2.0]]);
            let mean = m.matmul(x) + constant(Vector::<2>, [0.1, 0.4]);
            Normal::on(
                Vector::<2>,
                mean,
                Scale::covariance(&[1.0, 0.3, 0.3, 2.0], 2),
            )
            .score(y);
            let z = quantity("z", Vector::<2>);
            Normal::on(Vector::<2>, y, Scale::sd(0.5)).score(z);
            (x, y, z)
        })
        .unwrap();
    let query = Query::log_density().given(Observations::new().observe(z, [0.7, -1.2]));
    let point = Values::new()
        .with(x, points(&[0.5, 0.3, -0.4, 1.1], 2, 2, target))
        .with(y, points(&[1.0, -0.6, 0.2, 0.9], 2, 2, target));
    (writer, query, point)
}

/// A log-normal chain: `p ~ LN(1.5, 0.4²)`, `q ~ LN(p, 0.3²)`, and an observed `w ~ LN(q, 0.2²)`,
/// each a normal on the positive type, with a batch of two points.
fn positives(target: &Target) -> (Writer, Query, Values) {
    let mut writer = Writer::new();
    let (p, q, w) = writer
        .build(|| {
            let p = quantity("p", Positive::<1>);
            Normal::on(
                Positive::<1>,
                constant(Positive::<1>, [1.5]),
                Scale::sd(0.4),
            )
            .score(p);
            let q = quantity("q", Positive::<1>);
            Normal::on(Positive::<1>, p, Scale::sd(0.3)).score(q);
            let w = quantity("w", Positive::<1>);
            Normal::on(Positive::<1>, q, Scale::sd(0.2)).score(w);
            (p, q, w)
        })
        .unwrap();
    let query = Query::log_density().given(Observations::new().observe(w, [2.2]));
    let point = Values::new()
        .with(p, points(&[1.1, 2.0], 2, 1, target))
        .with(q, points(&[0.8, 2.5], 2, 1, target));
    (writer, query, point)
}

skoro_core::precision_tests! {
    fn linear_gaussian_terms_are_exact(target) {
        for model in [vectors as fn(&Target) -> (Writer, Query, Values), positives] {
            let (writer, query, point) = model(&target);
            let snapshot = writer.snapshot();
            let linearizer =
                Linearizer::new(&snapshot, &query, &target, Assembly::Residual).unwrap();
            let linearized = linearizer.at(&point).unwrap();
            let d = linearizer.layout().dimension();
            let steps: Vec<Vec<f64>> = (0..4)
                .map(|i| (0..d).map(|c| 0.3 * (i as f64) - 0.2 * (c as f64) + 0.1).collect())
                .collect();
            for step in &steps {
                let f = f_at(&snapshot, &query, linearizer.layout(), &point, &[step.clone(), step.clone()], &target);
                for (row, problem) in linearized.problems().iter().enumerate() {
                    close(&[problem.value(step)], &[f[row]], tolerance(&target), "m(τ) = f(p̂ ⊕ τ)");
                }
            }
        }
    }
}

#[test]
fn the_linear_term_is_zero_without_log_volumes() {
    let target = Target::host();
    let (writer, query, point) = vectors(&target);
    let linearizer =
        Linearizer::new(&writer.snapshot(), &query, &target, Assembly::Residual).unwrap();
    let schedule = Schedule::all(linearizer.structure()).unwrap();
    for problem in linearizer.at(&point).unwrap().problems() {
        assert!(
            problem.linear().is_none(),
            "no log-volume is bound on a vector"
        );
        // The step is the least-squares solution of the rows alone: the same, bit for bit, as
        // that of a problem assembled from them by hand.
        let by_hand = Problem::new(
            problem.structure().clone(),
            problem.factors().to_vec(),
            None,
            problem.constant(),
            problem.roundoff(),
        )
        .unwrap();
        let step = problem
            .eliminate(&schedule, None)
            .unwrap()
            .minimizer()
            .unwrap();
        let expected = by_hand
            .eliminate(&schedule, None)
            .unwrap()
            .minimizer()
            .unwrap();
        let bits = |v: &[f64]| v.iter().map(|x| x.to_bits()).collect::<Vec<_>>();
        assert_eq!(bits(&step), bits(&expected));
    }
}

skoro_core::precision_tests! {
    fn positive_log_volumes_are_a_linear_term(target) {
        let (writer, query, point) = positives(&target);
        let linearizer =
            Linearizer::new(&writer.snapshot(), &query, &target, Assembly::Residual).unwrap();
        for problem in linearizer.at(&point).unwrap().problems() {
            // The gradients of the log-volumes Σ log p and Σ log q in the logarithmic charts: one
            // per scored positive coordinate. The observed w's log-volume is constant.
            close(problem.linear().unwrap(), &[1.0, 1.0], tolerance(&target), "η");
        }
    }
}

/// A nonlinear model of vectors: `x ~ N(0, I)`, an observed `z ~ N(x ⊙ x, 0.3²)` and an observed
/// `u ~ N(M x, 0.5²)`, with a batch of `rows` points.
fn squares(rows: usize, target: &Target) -> (Writer, Query, Values) {
    let mut writer = Writer::new();
    let (x, z, u) = writer
        .build(|| {
            let x = quantity("x", Vector::<2>);
            Normal::on(
                Vector::<2>,
                constant(Vector::<2>, [0.0, 0.0]),
                Scale::sd(1.0),
            )
            .score(x);
            let z = quantity("z", Vector::<2>);
            Normal::on(Vector::<2>, x * x, Scale::sd(0.3)).score(z);
            let u = quantity("u", Vector::<2>);
            let m = constant(Matrix::<2, 2>, [[0.8, -0.2], [0.4, 1.1]]);
            Normal::on(Vector::<2>, m.matmul(x), Scale::sd(0.5)).score(u);
            (x, z, u)
        })
        .unwrap();
    let given = Observations::new()
        .observe(z, [0.6, 1.4])
        .observe(u, [0.3, -0.5]);
    let data: Vec<f64> = (0..rows)
        .flat_map(|r| [0.4 + 0.1 * r as f64, -0.7 + 0.2 * r as f64])
        .collect();
    let point = Values::new().with(x, points(&data, rows, 2, target));
    (writer, Query::log_density().given(given), point)
}

#[test]
fn the_step_is_gauss_newton() {
    let target = Target::host();
    let (writer, query, point) = squares(2, &target);
    let snapshot = writer.snapshot();
    let linearizer = Linearizer::new(&snapshot, &query, &target, Assembly::Residual).unwrap();
    let schedule = linearizer.schedule().unwrap();
    let linearized = linearizer.at(&point).unwrap();
    let solved = linearized.eliminate(&schedule, None).unwrap();
    let g = gradient(&snapshot, &query, linearizer.layout(), &point, &target);
    for (r, problem) in linearized.problems().iter().enumerate() {
        let minus: Vec<f64> = g[r].iter().map(|g| -g).collect();
        let expected = solve(&gram(problem), &minus);
        close(
            &solved.eliminations()[r].minimizer().unwrap(),
            &expected,
            1e-10,
            "(JᵀJ) τ = -g",
        );
    }
}

#[test]
fn the_step_keeps_its_digits() {
    // Free reals x₁ and x₂, measured as x₁ + x₂ = 2, ε x₁ = ε and ε x₂ = ε with unit noise: the
    // rows of the note's example, whose normal equations are singular as computed.
    let e = 1e-8;
    let target = Target::host();
    let mut writer = Writer::new();
    let (x1, x2, z) = writer
        .build(|| {
            let x1 = quantity("x1", Real);
            let x2 = quantity("x2", Real);
            let z: Vec<_> = (0..3).map(|i| quantity(format!("z/{i}"), Real)).collect();
            Normal::on(Real, x1 + x2, Scale::sd(1.0)).score(z[0]);
            Normal::on(Real, x1 * e, Scale::sd(1.0)).score(z[1]);
            Normal::on(Real, x2 * e, Scale::sd(1.0)).score(z[2]);
            (x1, x2, z)
        })
        .unwrap();
    let given = Observations::new()
        .observe(z[0], 2.0)
        .observe(z[1], e)
        .observe(z[2], e);
    let query = Query::log_density().given(given);
    let linearizer =
        Linearizer::new(&writer.snapshot(), &query, &target, Assembly::Residual).unwrap();
    let point = Values::new()
        .with(x1, points(&[0.0], 1, 1, &target))
        .with(x2, points(&[0.0], 1, 1, &target));
    let schedule = linearizer.schedule().unwrap();
    let solved = linearizer
        .at(&point)
        .unwrap()
        .eliminate(&schedule, None)
        .unwrap();
    close(
        &solved.eliminations()[0].minimizer().unwrap(),
        &[1.0, 1.0],
        1e-14,
        "the step",
    );
}

#[test]
fn batch_rows_share_the_schedule() {
    let target = Target::host();
    let (writer, query, point) = squares(5, &target);
    let snapshot = writer.snapshot();
    let linearizer = Linearizer::new(&snapshot, &query, &target, Assembly::Residual).unwrap();
    let schedule = linearizer.schedule().unwrap();
    let batch = linearizer
        .at(&point)
        .unwrap()
        .eliminate(&schedule, None)
        .unwrap();
    let x = linearizer.layout().quantities()[0];
    for r in 0..5 {
        let alone = Values::new().with(x, points(&row(point.get(x).unwrap(), r), 1, 2, &target));
        let one = linearizer
            .at(&alone)
            .unwrap()
            .eliminate(&schedule, None)
            .unwrap();
        close(
            &batch.eliminations()[r].minimizer().unwrap(),
            &one.eliminations()[0].minimizer().unwrap(),
            1e-12,
            "the row alone",
        );
    }
}

#[test]
fn damping_promises_the_decrease() {
    let target = Target::host();
    let (writer, query, point) = squares(1, &target);
    let snapshot = writer.snapshot();
    let linearizer = Linearizer::new(&snapshot, &query, &target, Assembly::Residual).unwrap();
    let schedule = linearizer.schedule().unwrap();
    let linearized = linearizer.at(&point).unwrap();
    let damping = Damping {
        lambda: 0.7,
        diagonal: vec![1.5, 0.5],
    };
    let solved = linearized.eliminate(&schedule, Some(&damping)).unwrap();
    let step = solved.eliminations()[0].minimizer().unwrap();
    let problem = &linearized.problems()[0];
    let mut damped = gram(problem);
    for i in 0..2 {
        damped[i][i] += damping.lambda * damping.diagonal[i];
    }
    let g = gradient(&snapshot, &query, linearizer.layout(), &point, &target);
    let minus: Vec<f64> = g[0].iter().map(|g| -g).collect();
    close(&step, &solve(&damped, &minus), 1e-10, "(JᵀJ + λD) τ = -g");
    let decrease = problem.value(&[0.0, 0.0]) - problem.value(&step);
    close(
        &[problem.promised(&step, &damping)],
        &[decrease],
        1e-10,
        "the promised decrease",
    );
}
