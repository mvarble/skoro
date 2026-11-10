#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)] // a test crate
#![allow(clippy::needless_range_loop)] // dense linear algebra, by index
//! The linearization of a query, against the query itself and the statements of its mathematics.

mod common;

use common::{
    close, f_at, factor_rows, gradient, gram, host, node_gradient, nodes_at, points, row, solve,
    tolerance,
};
use skoro_core::scope::{constant, quantity};
use skoro_core::{Key, Matrix, Observations, Positive, Query, Real, Values, Vector, Writer};
use skoro_elimination::{Damping, Problem, Schedule};
use skoro_linearize::{Assembly, Linearizer};
use skoro_normal::{Normal, Scale, roles};
use skoro_spatial::Rotation;
use skoro_tensor::{Points, Precision, Target};

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

/// A chain of rotations: `r₀ ~ N(μ₀, 0.3²)`, `r₁ ~ N(r₀, diag(0.2, 0.25, 0.15)²)`, and an observed
/// measurement `z ~ N(r₁, 0.1²)`, each a normal on the rotations, whose log-volume is folded; with
/// a batch of two points.
fn rotations(target: &Target) -> (Writer, Query, Values) {
    let unit = |q: [f64; 4]| {
        let n = q.iter().map(|x| x * x).sum::<f64>().sqrt();
        q.map(|x| x / n)
    };
    let mut writer = Writer::new();
    let (r0, r1, z) = writer
        .build(|| {
            let r0 = quantity("r0", Rotation);
            Normal::on(
                Rotation,
                constant(Rotation, unit([0.9, 0.1, -0.3, 0.2])),
                Scale::sd(0.3),
            )
            .score(r0);
            let r1 = quantity("r1", Rotation);
            Normal::on(Rotation, r0, Scale::sds(&[0.2, 0.25, 0.15])).score(r1);
            let z = quantity("z", Rotation);
            Normal::on(Rotation, r1, Scale::sd(0.1)).score(z);
            (r0, r1, z)
        })
        .unwrap();
    let query =
        Query::log_density().given(Observations::new().observe(z, unit([0.7, 0.3, -0.4, 0.5])));
    let a = [unit([0.8, 0.2, -0.3, 0.1]), unit([0.95, 0.0, -0.2, 0.25])].concat();
    let b = [
        unit([0.75, 0.25, -0.35, 0.45]),
        unit([0.6, 0.35, -0.3, 0.6]),
    ]
    .concat();
    let point = Values::new()
        .with(r0, points(&a, 2, 4, target))
        .with(r1, points(&b, 2, 4, target));
    (writer, query, point)
}

type Model = fn(&Target) -> (Writer, Query, Values);

skoro_core::precision_tests! {
    fn the_linearization_agrees_with_the_query(target) {
        let models: [(&str, Model); 3] = [
            ("vectors", |t| squares(2, t)),
            ("positives", positives),
            ("rotations", rotations),
        ];
        let (h, relative) = match target.precision() {
            Precision::Double => (1e-6, 1e-7),
            Precision::Single => (1e-2, 2e-3),
        };
        for (name, model) in models {
            let (writer, query, point) = model(&target);
            let snapshot = writer.snapshot();
            let linearizer = Linearizer::new(&snapshot, &query, &target, Assembly::Residual).unwrap();
            let layout = linearizer.layout();
            let d = layout.dimension();
            let linearized = linearizer.at(&point).unwrap();
            let rows = linearized.problems().len();
            // m(0) = f(p̂), and ∇m(0) = -Jᵀb + η = g.
            let f = f_at(&snapshot, &query, layout, &point, &vec![vec![0.0; d]; rows], &target);
            let g = gradient(&snapshot, &query, layout, &point, &target);
            for (r, problem) in linearized.problems().iter().enumerate() {
                close(&[problem.value(&vec![0.0; d])], &[f[r]], tolerance(&target), name);
                let (a, b) = common::dense_rows(problem);
                let eta = problem.linear().map_or(vec![0.0; d], <[f64]>::to_vec);
                let at_zero: Vec<f64> = (0..d)
                    .map(|c| eta[c] - a.iter().zip(&b).map(|(row, b)| row[c] * b).sum::<f64>())
                    .collect();
                close(&at_zero, &g[r], tolerance(&target), name);
            }
            // Each J_t against central differences of r_t along p̂ ⊕ h e_c, and against the dense
            // Jacobian of an identity frame, zero outside the factor's variables.
            let residuals: Vec<_> = linearizer
                .terms()
                .iter()
                .map(|&t| snapshot.term(t).unwrap().bound(&roles::RESIDUAL).unwrap())
                .collect();
            let program = snapshot
                .compile(&Query::of(residuals.clone()).given(query.observations().clone()), &target)
                .unwrap();
            assert_eq!(program.layout().quantities(), layout.quantities());
            let (_, identity) = program.tangent(&point, &skoro_core::Frames::identity(program.layout())).unwrap();
            for (f, &residual) in residuals.iter().enumerate() {
                let components = snapshot.space(residual).unwrap().dimension().unwrap();
                let dense = identity.get(residual).unwrap().to_host();
                for c in 0..d {
                    let step = |s: f64| (0..rows).map(|_| (0..d).map(|j| if j == c { s } else { 0.0 }).collect()).collect::<Vec<Vec<f64>>>();
                    let plus = host(&nodes_at(&snapshot, &query, &[residual], layout, &point, Some(&step(h)), &target), residual);
                    let minus = host(&nodes_at(&snapshot, &query, &[residual], layout, &point, Some(&step(-h)), &target), residual);
                    for (r, problem) in linearized.problems().iter().enumerate() {
                        let rows_of = factor_rows(problem, f);
                        let column: Vec<f64> = rows_of.iter().map(|row| row[c]).collect();
                        let differences: Vec<f64> = (0..components)
                            .map(|i| (plus[r * components + i] - minus[r * components + i]) / (2.0 * h))
                            .collect();
                        close(&column, &differences, relative, name);
                        let identity_column: Vec<f64> =
                            (0..components).map(|i| dense[(r * components + i) * d + c]).collect();
                        close(&column, &identity_column, tolerance(&target), name);
                    }
                }
            }
        }
    }
}

#[test]
fn folded_log_volumes_are_in_the_right_hand_sides() {
    let target = Target::host();
    let (writer, query, point) = rotations(&target);
    let snapshot = writer.snapshot();
    let linearizer = Linearizer::new(&snapshot, &query, &target, Assembly::Residual).unwrap();
    let layout = linearizer.layout().clone();
    let d = layout.dimension();
    let linearized = linearizer.at(&point).unwrap();
    let terms = linearizer.terms();
    let bound = |t, role| snapshot.term(t).unwrap().bound(role).unwrap();
    let residuals: Vec<_> = terms.iter().map(|&t| bound(t, &roles::RESIDUAL)).collect();
    let volumes: Vec<_> = terms
        .iter()
        .map(|&t| {
            snapshot
                .term(t)
                .unwrap()
                .bound(&roles::LOG_VOLUME)
                .unwrap()
                .unwrap()
        })
        .collect();
    let values = nodes_at(
        &snapshot, &query, &residuals, &layout, &point, None, &target,
    );
    for (r, problem) in linearized.problems().iter().enumerate() {
        assert!(
            problem.linear().is_none(),
            "every log-volume folded, the observed z's too"
        );
        for (f, (&residual, &volume)) in residuals.iter().zip(&volumes).enumerate() {
            // ĉ_t = -b_t - r̂_t, and J_tᵀ ĉ_t is the log-volume's gradient.
            let rows_of = factor_rows(problem, f);
            let n = rows_of.len();
            let r_hat = &host(&values, residual)[r * n..(r + 1) * n];
            let c: Vec<f64> = problem.factors()[f]
                .b
                .iter()
                .zip(r_hat)
                .map(|(b, r)| -b - r)
                .collect();
            let jtc: Vec<f64> = (0..d)
                .map(|j| rows_of.iter().zip(&c).map(|(row, c)| row[j] * c).sum())
                .collect();
            let expected = &node_gradient(&snapshot, &query, volume, &layout, &point, &target)[r];
            close(&jtc, expected, 1e-10, "J_tᵀ ĉ_t");
        }
    }
    // Iterated from one start, the folded steps and steps that carry the same gradients as a
    // linear term reach the same mode.
    let schedule = linearizer.schedule().unwrap();
    let mut folded = point.clone();
    let mut unfolded = point.clone();
    for _ in 0..40 {
        let step = linearizer
            .at(&folded)
            .unwrap()
            .eliminate(&schedule, None)
            .unwrap()
            .minimizer(&target)
            .unwrap();
        folded = layout.retract(&folded, &step, &target).unwrap();
        let at = linearizer.at(&unfolded).unwrap();
        let values = nodes_at(
            &snapshot, &query, &residuals, &layout, &unfolded, None, &target,
        );
        let mut data = Vec::new();
        for (r, problem) in at.problems().iter().enumerate() {
            let mut eta = vec![0.0; d];
            for &volume in &volumes {
                let g = &node_gradient(&snapshot, &query, volume, &layout, &unfolded, &target)[r];
                for (e, g) in eta.iter_mut().zip(g) {
                    *e += g;
                }
            }
            let factors: Vec<_> = problem
                .factors()
                .iter()
                .zip(&residuals)
                .map(|(rows, &residual)| {
                    let n = rows.b.len();
                    let r_hat = &host(&values, residual)[r * n..(r + 1) * n];
                    skoro_elimination::Rows {
                        a: rows.a.clone(),
                        b: r_hat.iter().map(|x| -x).collect(),
                    }
                })
                .collect();
            let plain = Problem::new(
                problem.structure().clone(),
                factors,
                Some(eta),
                0.0,
                problem.roundoff(),
            )
            .unwrap();
            data.extend(
                plain
                    .eliminate(&schedule, None)
                    .unwrap()
                    .minimizer()
                    .unwrap(),
            );
        }
        let step = Points::from_host(&data, 2, d, &target);
        unfolded = layout.retract(&unfolded, &step, &target).unwrap();
    }
    for &q in layout.quantities() {
        let (a, b) = (
            row(folded.get(q).unwrap(), 0),
            row(unfolded.get(q).unwrap(), 0),
        );
        let sign = if a.iter().zip(&b).map(|(x, y)| x * y).sum::<f64>() < 0.0 {
            -1.0
        } else {
            1.0
        };
        let b: Vec<f64> = b.iter().map(|x| sign * x).collect();
        close(&a, &b, 1e-9, "the same mode");
    }
}

#[test]
fn steps_converge_on_rotations() {
    let target = Target::host();
    let (writer, query, point) = rotations(&target);
    let snapshot = writer.snapshot();
    let linearizer = Linearizer::new(&snapshot, &query, &target, Assembly::Residual).unwrap();
    let layout = linearizer.layout().clone();
    let schedule = linearizer.schedule().unwrap();
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
    // A stationary point of f itself, log-volumes included: the query's gradient vanishes.
    for g in gradient(&snapshot, &query, &layout, &at, &target) {
        let norm = g.iter().map(|g| g * g).sum::<f64>().sqrt();
        assert!(norm < 1e-9, "|g| = {norm}");
    }
}

/// `log 𝒩(x; m, s²)`.
fn log_normal(x: f64, m: f64, s: f64) -> f64 {
    -0.5 * ((x - m) / s).powi(2) - s.ln() - 0.5 * (2.0 * std::f64::consts::PI).ln()
}

#[test]
fn the_laplace_approximation_is_centred_in_the_chart() {
    let target = Target::host();
    // A quantity scored by a log-normal of median μ: the chart linearization at μ has a zero
    // minimizer and the variance of the logarithm.
    let (mu, sigma) = (1.7, 0.35);
    let mut writer = Writer::new();
    let p = writer
        .build(|| {
            let p = quantity("p", Positive::<1>);
            Normal::on(
                Positive::<1>,
                constant(Positive::<1>, [mu]),
                Scale::sd(sigma),
            )
            .score(p);
            p
        })
        .unwrap();
    let query = Query::log_density();
    let linearizer =
        Linearizer::new(&writer.snapshot(), &query, &target, Assembly::Residual).unwrap();
    let point = Values::new().with(p, points(&[mu], 1, 1, &target));
    let chart = linearizer.at_in_chart(&point, &[p.id()]).unwrap();
    let schedule = linearizer.schedule().unwrap();
    let solved = chart.eliminate(&schedule, None).unwrap();
    close(
        &solved.eliminations()[0].minimizer().unwrap(),
        &[0.0],
        1e-14,
        "centred at the median",
    );
    let keeping = linearizer.schedule_keeping(&[p.id()]).unwrap();
    let marginal = &chart.eliminate(&keeping, None).unwrap().marginal().unwrap()[0];
    close(
        marginal.marginal.covariance().data(),
        &[sigma * sigma],
        1e-12,
        "the variance of log p",
    );
    // The evidence of a linear-Gaussian model of positives: p ~ LN(μ₀, s²), w ~ LN(p, r²) with w
    // observed, is log 𝒩(log w; log μ₀, s² + r²) - log w, at any point.
    let (mu0, s, r, w_hat) = (1.3, 0.4, 0.25, 2.1);
    let mut writer = Writer::new();
    let (p, w) = writer
        .build(|| {
            let p = quantity("p", Positive::<1>);
            Normal::on(Positive::<1>, constant(Positive::<1>, [mu0]), Scale::sd(s)).score(p);
            let w = quantity("w", Positive::<1>);
            Normal::on(Positive::<1>, p, Scale::sd(r)).score(w);
            (p, w)
        })
        .unwrap();
    let snapshot = writer.snapshot();
    let query = Query::log_density().given(Observations::new().observe(w, [w_hat]));
    let linearizer = Linearizer::new(&snapshot, &query, &target, Assembly::Residual).unwrap();
    let schedule = linearizer.schedule().unwrap();
    let closed = log_normal(w_hat.ln(), mu0.ln(), (s * s + r * r).sqrt()) - w_hat.ln();
    for at in [0.6, 1.5, 2.8] {
        let point = Values::new().with(p, points(&[at], 1, 1, &target));
        let chart = linearizer.at_in_chart(&point, &[p.id()]).unwrap();
        let evidence = chart
            .eliminate(&schedule, None)
            .unwrap()
            .log_integral()
            .unwrap();
        close(&evidence, &[closed], 1e-12, "the evidence");
    }
    // And the closed form against quadrature of the model's own density over u = log p.
    let n = 4000;
    let (low, high) = (-6.0, 6.0);
    let du = (high - low) / n as f64;
    let us: Vec<f64> = (0..n).map(|i| low + (i as f64 + 0.5) * du).collect();
    let ps: Vec<f64> = us.iter().map(|u: &f64| u.exp()).collect();
    let program = snapshot.compile(&query, &target).unwrap();
    let at = Values::new().with(p, points(&ps, n, 1, &target));
    let logs = host(&program.forward(&at).unwrap(), skoro_core::NodeId::DENSITY);
    let integral: f64 = logs.iter().zip(&ps).map(|(l, p)| l.exp() * p * du).sum();
    close(&[integral.ln()], &[closed], 1e-8, "quadrature");
}

#[test]
fn draws_have_the_marginal_covariance() {
    let target = Target::host();
    let n = 6000;
    let (writer, query, _) = squares(1, &target);
    let snapshot = writer.snapshot();
    let linearizer = Linearizer::new(&snapshot, &query, &target, Assembly::Residual).unwrap();
    let x = linearizer.layout().quantities()[0];
    let data: Vec<f64> = (0..n).flat_map(|_| [0.4, -0.7]).collect();
    let point = Values::new().with(x, points(&data, n, 2, &target));
    let schedule = linearizer.schedule().unwrap();
    let solved = linearizer
        .at(&point)
        .unwrap()
        .eliminate(&schedule, None)
        .unwrap();
    let draws = solved.draw(&point, &Key::from_seed(7), &target).unwrap();
    let again = solved.draw(&point, &Key::from_seed(7), &target).unwrap();
    assert_eq!(
        host(&draws, x),
        host(&again, x),
        "a draw is a function of its key"
    );
    // On a vector the chart is a translation: the draws less the point and the step are the
    // Gaussian's, of the marginal covariance.
    let step = solved.eliminations()[0].minimizer().unwrap();
    let samples = host(&draws, x);
    let centred: Vec<[f64; 2]> = (0..n)
        .map(|r| {
            [
                samples[2 * r] - 0.4 - step[0],
                samples[2 * r + 1] + 0.7 - step[1],
            ]
        })
        .collect();
    let covariance: Vec<f64> = (0..2)
        .flat_map(|i| (0..2).map(move |j| (i, j)))
        .map(|(i, j)| centred.iter().map(|c| c[i] * c[j]).sum::<f64>() / n as f64)
        .collect();
    let keeping = linearizer.schedule_keeping(&[x]).unwrap();
    let alone = Values::new().with(x, points(&[0.4, -0.7], 1, 2, &target));
    let marginal = &linearizer
        .at(&alone)
        .unwrap()
        .eliminate(&keeping, None)
        .unwrap()
        .marginal()
        .unwrap()[0];
    let expected = marginal.marginal.covariance();
    for (c, e) in covariance.iter().zip(expected.data()) {
        // Five standard errors of a sample covariance, about √(2/n) of the variances.
        let scale = expected.get(0, 0).max(expected.get(1, 1));
        assert!(
            (c - e).abs() < 5.0 * (2.0 / n as f64).sqrt() * scale,
            "{covariance:?} against {expected:?}"
        );
    }
}

#[test]
fn marginals_name_their_curved_quantities() {
    let target = Target::host();
    let mut writer = Writer::new();
    let (v, p, r) = writer
        .build(|| {
            let v = quantity("v", Vector::<2>);
            Normal::on(
                Vector::<2>,
                constant(Vector::<2>, [0.0, 0.0]),
                Scale::sd(1.0),
            )
            .score(v);
            let p = quantity("p", Positive::<1>);
            Normal::on(
                Positive::<1>,
                constant(Positive::<1>, [1.0]),
                Scale::sd(0.5),
            )
            .score(p);
            let r = quantity("r", Rotation);
            Normal::on(
                Rotation,
                constant(Rotation, [1.0, 0.0, 0.0, 0.0]),
                Scale::sd(0.3),
            )
            .score(r);
            (v, p, r)
        })
        .unwrap();
    let linearizer = Linearizer::new(
        &writer.snapshot(),
        &Query::log_density(),
        &target,
        Assembly::Residual,
    )
    .unwrap();
    let point = Values::new()
        .with(v, points(&[0.1, 0.2], 1, 2, &target))
        .with(p, points(&[1.2], 1, 1, &target))
        .with(
            r,
            points(
                &[0.9, 0.1, 0.2, 0.3].map(|x: f64| x / 0.95_f64.sqrt()),
                1,
                4,
                &target,
            ),
        );
    let keep = [v.id(), p.id(), r.id()];
    let schedule = linearizer.schedule_keeping(&keep).unwrap();
    let marginals = linearizer
        .at(&point)
        .unwrap()
        .eliminate(&schedule, None)
        .unwrap()
        .marginal()
        .unwrap();
    assert_eq!(marginals[0].quantities, keep);
    assert_eq!(marginals[0].curved, [r.id()]);
}
