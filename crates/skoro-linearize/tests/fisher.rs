#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)] // a test crate
#![allow(clippy::needless_range_loop)] // dense linear algebra, by index
//! The Fisher linearization: a root of the expected information, with the query's gradient.

mod common;

use common::{close, dense_rows, gradient, gram, host, points};
use skoro_core::scope::{constant, quantity, quantity_in};
use skoro_core::{Frames, Key, Observations, Positive, QDyn, Query, Space, Values, Vector, Writer};
use skoro_linearize::{Assembly, Error, Linearizer};
use skoro_normal::{Form, Normal, Scale, triangular};
use skoro_tensor::{HostData, HostValue, Target};

const D: usize = 2;
const MU: [f64; 2] = [0.4, 1.3];

/// The scale's space and its value, and the factor `L` with `x = μ ⊕ L z`, row by row.
fn scale_of(form: Form) -> (Space, Vec<f64>, [[f64; 2]; 2]) {
    match form {
        Form::Isotropic => (Space::Positive(1), vec![0.7], [[0.7, 0.0], [0.0, 0.7]]),
        Form::Diagonal => (Space::Positive(2), vec![0.5, 1.2], [[0.5, 0.0], [0.0, 1.2]]),
        // L itself: its diagonal, then the entry below it.
        Form::CovarianceFactor => (triangular(2), vec![0.6, 0.9, 0.3], [[0.6, 0.0], [0.3, 0.9]]),
        // W, with Σ = (WᵀW)⁻¹, so that L = W⁻¹.
        Form::Whitening => {
            let (w00, w11, w10) = (1.5, 0.8, -0.4);
            (
                triangular(2),
                vec![w00, w11, w10],
                [[1.0 / w00, 0.0], [-w10 / (w00 * w11), 1.0 / w11]],
            )
        }
    }
}

fn scale(form: Form, theta: QDyn) -> Scale {
    match form {
        Form::Isotropic => Scale::isotropic(theta.narrow()),
        Form::Diagonal => Scale::diagonal(theta),
        Form::CovarianceFactor => Scale::covariance_factor(theta),
        Form::Whitening => Scale::whitening(theta),
    }
}

/// `x`, observed per row, scored by the normal on `space` with the free mean `μ` and the free
/// scale `θ` of `form`: the query, the point `(μ, θ)` for `rows` rows, and `μ` and `θ`.
fn free_scale(
    space: &Space,
    form: Form,
    xs: &[[f64; 2]],
    target: &Target,
) -> (Writer, Query, Values, QDyn, QDyn) {
    let (scale_space, theta_value, _) = scale_of(form);
    let mut writer = Writer::new();
    let (mu, theta, x) = writer
        .build(|| {
            let mu = quantity_in("mu", space.clone());
            let theta = quantity_in("theta", scale_space.clone());
            let x = quantity_in("x", space.clone());
            Normal::in_space(space.clone(), mu, scale(form, theta)).score(x);
            (mu, theta, x)
        })
        .unwrap();
    let rows = xs.len();
    let observed = HostValue::new(Some(rows), D, HostData::Continuous(xs.concat())).unwrap();
    let query = Query::log_density().given(Observations::new().observe_dyn(x, observed));
    let k = theta_value.len();
    let point = Values::new()
        .with(mu, points(&MU.repeat(rows), rows, D, target))
        .with(theta, points(&theta_value.repeat(rows), rows, k, target));
    (writer, query, point, mu, theta)
}

/// `n` draws of `x = μ ⊕ L z` on `space`, from `seed`.
fn draws(space: &Space, form: Form, n: usize, seed: u64) -> Vec<[f64; 2]> {
    let (_, _, l) = scale_of(form);
    let keys = Key::from_seed(seed).split(n);
    keys.iter()
        .map(|key| {
            let z: Vec<f64> = key
                .uniforms(D)
                .into_iter()
                .map(skoro_normal::quantile)
                .collect();
            let t = [l[0][0] * z[0], l[1][0] * z[0] + l[1][1] * z[1]];
            match space {
                Space::Positive(_) => [MU[0] * t[0].exp(), MU[1] * t[1].exp()],
                _ => [MU[0] + t[0], MU[1] + t[1]],
            }
        })
        .collect()
}

const FORMS: [Form; 4] = [
    Form::Isotropic,
    Form::Diagonal,
    Form::CovarianceFactor,
    Form::Whitening,
];

#[test]
fn fisher_rows_are_a_root_of_the_expected_information() {
    let target = Target::host();
    for space in [Space::Vector(D), Space::Positive(D)] {
        for form in FORMS {
            let n = 40_000;
            let xs = draws(&space, form, n, 1);
            let (writer, query, point, _, _) = free_scale(&space, form, &xs, &target);
            let snapshot = writer.snapshot();
            let linearizer = Linearizer::new(&snapshot, &query, &target, Assembly::Fisher).unwrap();
            let information = gram(&linearizer.at(&point).unwrap().problems()[0]);
            // E[∇²f] over the draws, by the core's Hessian passes, f = -log p.
            let program = snapshot.compile(&query, &target).unwrap();
            let layout = program.layout();
            let second = program
                .hessian_products(&point, &Frames::identity(layout))
                .unwrap();
            let d = layout.dimension();
            let mut sum = vec![vec![0.0; d]; d];
            let mut squares = vec![vec![0.0; d]; d];
            let mut offset = 0;
            for (&q, block) in layout.quantities().iter().zip(layout.blocks()) {
                let beta = second.beta.get(q).unwrap().to_host();
                let dq = block.tangent.len();
                for r in 0..n {
                    for i in 0..dq {
                        for j in 0..d {
                            let h = -beta[(r * dq + i) * d + j];
                            sum[offset + i][j] += h;
                            squares[offset + i][j] += h * h;
                        }
                    }
                }
                offset += dq;
            }
            for i in 0..d {
                for j in 0..d {
                    let mean = sum[i][j] / n as f64;
                    let se = ((squares[i][j] / n as f64 - mean * mean).max(0.0) / n as f64).sqrt();
                    assert!(
                        (information[i][j] - mean).abs() <= 5.0 * se + 1e-9,
                        "{space:?} {form:?} ({i}, {j}): AᵀA {} against E[H] {mean} ± {se}",
                        information[i][j]
                    );
                }
            }
        }
    }
}

#[test]
fn fisher_gradient_is_the_query_gradient() {
    let target = Target::host();
    for space in [Space::Vector(D), Space::Positive(D)] {
        for form in FORMS {
            let xs = draws(&space, form, 3, 2);
            let (writer, query, point, _, _) = free_scale(&space, form, &xs, &target);
            let snapshot = writer.snapshot();
            let linearizer = Linearizer::new(&snapshot, &query, &target, Assembly::Fisher).unwrap();
            let layout = linearizer.layout().clone();
            let g = gradient(&snapshot, &query, &layout, &point, &target);
            for (r, problem) in linearizer.at(&point).unwrap().problems().iter().enumerate() {
                let (a, b) = dense_rows(problem);
                let d = layout.dimension();
                let eta = problem.linear().map_or(vec![0.0; d], <[f64]>::to_vec);
                let at_zero: Vec<f64> = (0..d)
                    .map(|c| eta[c] - a.iter().zip(&b).map(|(row, b)| row[c] * b).sum::<f64>())
                    .collect();
                close(&at_zero, &g[r], 1e-10, &format!("{space:?} {form:?}"));
            }
        }
    }
}

#[test]
fn fisher_with_fixed_scales_is_the_residual_linearization() {
    let target = Target::host();
    for space in [Space::Vector(D), Space::Positive(D)] {
        for form in FORMS {
            // The same model with θ observed: every scale is fixed, and μ alone is free.
            let xs = draws(&space, form, 2, 3);
            let (writer, query, point, _, theta) = free_scale(&space, form, &xs, &target);
            let (_, theta_value, _) = scale_of(form);
            let fixed =
                HostValue::new(None, theta_value.len(), HostData::Continuous(theta_value)).unwrap();
            let given = Observations::new()
                .observe_dyn(theta, fixed)
                .union(query.observations());
            let query = Query::log_density().given(given);
            let snapshot = writer.snapshot();
            let residual = Linearizer::new(&snapshot, &query, &target, Assembly::Residual).unwrap();
            let fisher = Linearizer::new(&snapshot, &query, &target, Assembly::Fisher).unwrap();
            let (a, b) = (residual.at(&point).unwrap(), fisher.at(&point).unwrap());
            for (p, q) in a.problems().iter().zip(b.problems()) {
                assert_eq!(p.factors(), q.factors(), "{space:?} {form:?}: row for row");
                assert_eq!(p.linear(), q.linear());
                assert_eq!(p.constant(), q.constant());
            }
        }
    }
}

/// `n` observed vectors `x_i ~ N(μ, Σ)`, with the free mean `μ` and the free scale of `form`,
/// isotropic or diagonal, and no priors: the query and its point at `μ = 0`, `σ = 1`.
fn sample(form: Form, xs: &[[f64; 2]], target: &Target) -> (Writer, Query, Values) {
    let (scale_space, _, _) = scale_of(form);
    let mut writer = Writer::new();
    let (mu, theta, x) = writer
        .build(|| {
            let mu = quantity("mu", Vector::<2>);
            let theta = quantity_in("theta", scale_space.clone());
            let x: Vec<_> = (0..xs.len())
                .map(|i| {
                    let x = quantity(format!("x/{i}"), Vector::<2>);
                    Normal::on(Vector::<2>, mu, scale(form, theta)).score(x);
                    x
                })
                .collect();
            (mu, theta, x)
        })
        .unwrap();
    let given = x
        .iter()
        .zip(xs)
        .fold(Observations::new(), |o, (&x, v)| o.observe(x, *v));
    let k = scale_space.dimension().unwrap();
    let point = Values::new()
        .with(mu, points(&[0.0, 0.0], 1, D, target))
        .with(theta, points(&vec![1.0; k], 1, k, target));
    (writer, Query::log_density().given(given), point)
}

#[test]
fn fisher_scoring_converges_to_the_estimate() {
    let target = Target::host();
    let n = 40;
    let xs = draws(&Space::Vector(D), Form::Diagonal, n, 4);
    let mean = [0, 1].map(|j| xs.iter().map(|x| x[j]).sum::<f64>() / n as f64);
    let variance =
        [0, 1].map(|j| xs.iter().map(|x| (x[j] - mean[j]).powi(2)).sum::<f64>() / n as f64);
    for form in [Form::Isotropic, Form::Diagonal] {
        let (writer, query, mut point) = sample(form, &xs, &target);
        let linearizer =
            Linearizer::new(&writer.snapshot(), &query, &target, Assembly::Fisher).unwrap();
        let layout = linearizer.layout().clone();
        let schedule = linearizer.schedule().unwrap();
        for _ in 0..60 {
            let step = linearizer
                .at(&point)
                .unwrap()
                .eliminate(&schedule, None)
                .unwrap()
                .minimizer(&target)
                .unwrap();
            point = layout.retract(&point, &step, &target).unwrap();
        }
        let [mu, theta] = [0, 1].map(|i| host(&point, layout.quantities()[i]));
        close(&mu, &mean, 1e-9, "the mean");
        let expected: Vec<f64> = match form {
            Form::Isotropic => vec![((variance[0] + variance[1]) / 2.0).sqrt()],
            _ => variance.iter().map(|v| v.sqrt()).collect(),
        };
        close(&theta, &expected, 1e-9, &format!("{form:?}: the scale"));
    }
    // An isotropic scale alone, with the mean fixed: the step in u = log σ is (|r|² - d) / 2d.
    let mut writer = Writer::new();
    let (sigma, x) = writer
        .build(|| {
            let sigma = quantity("sigma", Positive::<1>);
            let x = quantity("x", Vector::<2>);
            Normal::on(
                Vector::<2>,
                constant(Vector::<2>, [0.0, 0.0]),
                Scale::isotropic(sigma),
            )
            .score(x);
            (sigma, x)
        })
        .unwrap();
    let (s0, x0) = (0.8, [0.9, -1.3]);
    let query = Query::log_density().given(Observations::new().observe(x, x0));
    let linearizer =
        Linearizer::new(&writer.snapshot(), &query, &target, Assembly::Fisher).unwrap();
    let point = Values::new().with(sigma, points(&[s0], 1, 1, &target));
    let schedule = linearizer.schedule().unwrap();
    let step = linearizer
        .at(&point)
        .unwrap()
        .eliminate(&schedule, None)
        .unwrap()
        .eliminations()[0]
        .minimizer()
        .unwrap();
    let r2 = (x0[0] * x0[0] + x0[1] * x0[1]) / (s0 * s0);
    close(
        &step,
        &[(r2 - D as f64) / (2.0 * D as f64)],
        1e-12,
        "the scale's step",
    );
}

#[test]
fn fisher_marginals_are_the_inverse_information() {
    // n draws with an isotropic scale σ: the information is n I / σ² on μ and 2nd on log σ, so the
    // marginal covariances are σ² I / n and 1 / 2nd, at any point.
    let target = Target::host();
    let n = 25;
    let xs = draws(&Space::Vector(D), Form::Isotropic, n, 5);
    let (writer, query, _) = sample(Form::Isotropic, &xs, &target);
    let linearizer =
        Linearizer::new(&writer.snapshot(), &query, &target, Assembly::Fisher).unwrap();
    let [mu, theta] = [0, 1].map(|i| linearizer.layout().quantities()[i]);
    let sigma = 0.6;
    let point = Values::new()
        .with(mu, points(&[0.2, -0.1], 1, D, &target))
        .with(theta, points(&[sigma], 1, 1, &target));
    let linearized = linearizer.at(&point).unwrap();
    for (keep, expected) in [
        (
            mu,
            vec![sigma * sigma / n as f64, 0.0, 0.0, sigma * sigma / n as f64],
        ),
        (theta, vec![1.0 / (2.0 * n as f64 * D as f64)]),
    ] {
        let schedule = linearizer.schedule_keeping(&[keep]).unwrap();
        let marginal = &linearized
            .eliminate(&schedule, None)
            .unwrap()
            .marginal()
            .unwrap()[0];
        close(
            marginal.marginal.covariance().data(),
            &expected,
            1e-12,
            "the inverse information",
        );
    }
}

#[test]
fn fisher_shares_the_schedule() {
    let target = Target::host();
    let xs = draws(&Space::Vector(D), Form::Diagonal, 2, 6);
    let (writer, query, point, _, theta) =
        free_scale(&Space::Vector(D), Form::Diagonal, &xs, &target);
    let snapshot = writer.snapshot();
    let fixed = HostValue::new(None, 2, HostData::Continuous(vec![0.5, 1.2])).unwrap();
    let fixed_query = Query::log_density().given(
        Observations::new()
            .observe_dyn(theta, fixed)
            .union(query.observations()),
    );
    let residual = Linearizer::new(&snapshot, &fixed_query, &target, Assembly::Residual).unwrap();
    let fisher = Linearizer::new(&snapshot, &fixed_query, &target, Assembly::Fisher).unwrap();
    assert_eq!(**residual.structure(), **fisher.structure());
    let schedule = residual.schedule().unwrap();
    let a = residual
        .at(&point)
        .unwrap()
        .eliminate(&schedule, None)
        .unwrap();
    let b = fisher
        .at(&point)
        .unwrap()
        .eliminate(&schedule, None)
        .unwrap();
    for (a, b) in a.eliminations().iter().zip(b.eliminations()) {
        close(
            &a.minimizer().unwrap(),
            &b.minimizer().unwrap(),
            1e-14,
            "one schedule, both assemblies",
        );
    }
    // With the scale free, the Fisher structure involves it, and one schedule serves every row.
    let free = Linearizer::new(&snapshot, &query, &target, Assembly::Fisher).unwrap();
    let schedule = free.schedule().unwrap();
    assert!(free.at(&point).unwrap().eliminate(&schedule, None).is_ok());
}

#[test]
fn fisher_gives_no_integral() {
    let target = Target::host();
    let xs = draws(&Space::Vector(D), Form::Isotropic, 1, 7);
    let (writer, query, point, mu, _) =
        free_scale(&Space::Vector(D), Form::Isotropic, &xs, &target);
    let linearizer =
        Linearizer::new(&writer.snapshot(), &query, &target, Assembly::Fisher).unwrap();
    assert_eq!(
        linearizer.at_in_chart(&point, &[mu.id()]).unwrap_err(),
        Error::NotAnIntegral
    );
    let solved = linearizer
        .at(&point)
        .unwrap()
        .eliminate(&linearizer.schedule().unwrap(), None)
        .unwrap();
    assert_eq!(solved.log_integral().unwrap_err(), Error::NotAnIntegral);
    assert_eq!(
        solved
            .draw(&point, &Key::from_seed(1), &target)
            .unwrap_err(),
        Error::NotAnIntegral
    );
}
