#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)] // a test crate
//! The linearization of a query, against the query itself and the statements of its mathematics.

mod common;

use common::{close, f_at, points, tolerance};
use skoro_core::scope::{constant, quantity};
use skoro_core::{Matrix, Observations, Positive, Query, Values, Vector, Writer};
use skoro_elimination::{Problem, Schedule};
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
