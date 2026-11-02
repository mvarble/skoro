#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)] // a test crate
//! The normal's term on the flat-in-disguise spaces: the closed form, and a density.

mod common;

use common::{close, evaluate, inverse, log_gaussian, tolerance};
use skoro_core::scope::{constant, product, quantity, quantity_in};
use skoro_core::{Matrix, Observations, Positive, Real, Rows, Space, Vector, Writer};
use skoro_normal::{Normal, Scale};
use skoro_tensor::{HostData, HostValue};

/// A symmetric positive definite `Σ` of side 3, row by row.
const SIGMA: [f64; 9] = [0.8, 0.2, -0.1, 0.2, 0.5, 0.15, -0.1, 0.15, 0.6];

/// The four scales of a `Σ`, as `build` makes them: the matrix itself as a factor and a
/// precision, and, when `Σ` is diagonal, its standard deviations; the isotropic one for `σ`.
fn four_scales(sigma: &[f64], d: usize) -> Vec<(Scale, Vec<f64>)> {
    let mut scales = vec![
        (Scale::covariance(sigma, d), sigma.to_vec()),
        (Scale::precision(&inverse(sigma, d), d), sigma.to_vec()),
    ];
    let sd = 0.6;
    let mut isotropic = vec![0.0; d * d];
    for i in 0..d {
        isotropic[i * d + i] = sd * sd;
    }
    scales.push((Scale::sd(sd), isotropic));
    let sds: Vec<f64> = (0..d).map(|i| 0.3 + 0.2 * i as f64).collect();
    let mut diagonal = vec![0.0; d * d];
    for i in 0..d {
        diagonal[i * d + i] = sds[i] * sds[i];
    }
    scales.push((Scale::sds(&sds), diagonal));
    scales
}

skoro_core::precision_tests! {
    fn term_is_the_closed_form(target) {
        let relative = tolerance(&target);

        // A real line, its mean a quantity.
        let mut writer = Writer::new();
        let (x, mu, term) = writer
            .build(|| {
                let (x, mu) = (quantity("x", Real), quantity("mu", Real));
                let term = Normal::on(Real, mu, Scale::sd(0.7)).log_density(x).log_density;
                (x, mu, term)
            })
            .unwrap();
        let xs = vec![-1.3, 0.2, 0.9, 2.5];
        let given = Observations::new().observe(x, Rows(xs.clone())).observe(mu, 0.4);
        let got = &evaluate(&writer.snapshot(), &[term.untyped()], given, &target)[0];
        let want: Vec<f64> = xs.iter().map(|x| log_gaussian(&[x - 0.4], &[0.49])).collect();
        close(got, &want, relative);

        // A vector, in each form of the scale.
        let mean = [0.1, -0.3, 0.7];
        let points = [[0.5, 0.2, 0.1], [-1.0, 0.4, 1.9], [0.1, -0.3, 0.7]];
        let mut writer = Writer::new();
        let (x, terms) = writer
            .build(|| {
                let x = quantity("x", Vector::<3>);
                let mu = constant(Vector::<3>, mean);
                let terms: Vec<_> = four_scales(&SIGMA, 3)
                    .into_iter()
                    .map(|(scale, sigma)| {
                        (Normal::on(Vector::<3>, mu, scale).log_density(x).log_density, sigma)
                    })
                    .collect();
                (x, terms)
            })
            .unwrap();
        let given = Observations::new().observe(x, Rows(points.to_vec()));
        let nodes: Vec<_> = terms.iter().map(|(t, _)| t.untyped()).collect();
        let got = evaluate(&writer.snapshot(), &nodes, given, &target);
        for ((_, sigma), got) in terms.iter().zip(&got) {
            let want: Vec<f64> = points
                .iter()
                .map(|p| log_gaussian(&[p[0] - mean[0], p[1] - mean[1], p[2] - mean[2]], sigma))
                .collect();
            close(got, &want, relative);
        }

        // A matrix, read column by column, with an isotropic scale.
        let mut writer = Writer::new();
        let (x, term) = writer
            .build(|| {
                let x = quantity("x", Matrix::<2, 2>);
                let mu = constant(Matrix::<2, 2>, [[0.0, 1.0], [2.0, 3.0]]);
                (x, Normal::on(Matrix::<2, 2>, mu, Scale::sd(1.5)).log_density(x).log_density)
            })
            .unwrap();
        let given = Observations::new().observe(x, [[0.5, 0.0], [1.0, 4.0]]);
        let got = &evaluate(&writer.snapshot(), &[term.untyped()], given, &target)[0];
        let want: f64 = [0.5, 1.0, 0.0, 4.0]
            .iter()
            .zip([0.0, 2.0, 1.0, 3.0])
            .map(|(x, m)| log_gaussian(&[x - m], &[2.25]))
            .sum();
        close(got, &[want], relative);

        // The positive orthant: a log-normal, less J(x) = Σ log x_i.
        let mut writer = Writer::new();
        let (x, term) = writer
            .build(|| {
                let x = quantity("x", Positive::<2>);
                let mu = constant(Positive::<2>, [1.5, 0.4]);
                (x, Normal::on(Positive::<2>, mu, Scale::sds(&[0.3, 0.8])).log_density(x).log_density)
            })
            .unwrap();
        let points = [[2.0, 0.1], [1.5, 0.4], [0.7, 3.0]];
        let given = Observations::new().observe(x, Rows(points.to_vec()));
        let got = &evaluate(&writer.snapshot(), &[term.untyped()], given, &target)[0];
        let want: Vec<f64> = points
            .iter()
            .map(|p| {
                let t = [(p[0] / 1.5).ln(), (p[1] / 0.4).ln()];
                log_gaussian(&t, &[0.09, 0.0, 0.0, 0.64]) - p[0].ln() - p[1].ln()
            })
            .collect();
        close(got, &want, relative);

        // A product of a vector and a positive factor, correlated through Σ.
        let space = Space::product([Space::Vector(2), Space::Positive(1)]).unwrap();
        let mut writer = Writer::new();
        let (x, term) = writer
            .build(|| {
                let x = quantity_in("x", space.clone());
                let mu = product(&[
                    constant(Vector::<2>, [0.1, 0.2]).untyped(),
                    constant(Positive::<1>, [2.0]).untyped(),
                ]);
                (x, Normal::in_space(space.clone(), mu, Scale::covariance(&SIGMA, 3)).log_density(x).log_density)
            })
            .unwrap();
        let points = [0.4, -0.1, 3.0, 0.0, 0.5, 1.2];
        let value = HostValue::new(Some(2), 3, HostData::Continuous(points.to_vec())).unwrap();
        let given = Observations::new().observe_dyn(x, value);
        let got = &evaluate(&writer.snapshot(), &[term.untyped()], given, &target)[0];
        let want: Vec<f64> = points
            .chunks(3)
            .map(|p| {
                let t = [p[0] - 0.1, p[1] - 0.2, (p[2] / 2.0).ln()];
                log_gaussian(&t, &SIGMA) - p[2].ln()
            })
            .collect();
        close(got, &want, relative);
    }

    fn density_integrates_to_one(target) {
        let relative = match target.precision() {
            skoro_tensor::Precision::Double => 1e-9,
            skoro_tensor::Precision::Single => 1e-4,
        };
        // On the real line, by the trapezoidal rule over ten standard deviations.
        let (mean, sd) = (0.3, 0.7);
        let mut writer = Writer::new();
        let (x, term) = writer
            .build(|| {
                let x = quantity("x", Real);
                (x, Normal::on(Real, constant(Real, mean), Scale::sd(sd)).log_density(x).log_density)
            })
            .unwrap();
        let n = 4001;
        let step = 20.0 * sd / (n - 1) as f64;
        let xs: Vec<f64> = (0..n).map(|i| mean - 10.0 * sd + step * i as f64).collect();
        let given = Observations::new().observe(x, Rows(xs.clone()));
        let densities = &evaluate(&writer.snapshot(), &[term.untyped()], given, &target)[0];
        let integral: f64 = densities.iter().map(|l| l.exp()).sum::<f64>() * step;
        close(&[integral], &[1.0], relative);

        // On the positive reals against Lebesgue measure, through x = e^s: ∫ p(x) x ds.
        let (median, sd) = (1.5, 0.4);
        let mut writer = Writer::new();
        let (x, term) = writer
            .build(|| {
                let x = quantity("x", Positive::<1>);
                let mu = constant(Positive::<1>, [median]);
                (x, Normal::on(Positive::<1>, mu, Scale::sd(sd)).log_density(x).log_density)
            })
            .unwrap();
        let center = f64::ln(median);
        let step = 20.0 * sd / (n - 1) as f64;
        let xs: Vec<[f64; 1]> =
            (0..n).map(|i| [(center - 10.0 * sd + step * i as f64).exp()]).collect();
        let given = Observations::new().observe(x, Rows(xs.clone()));
        let densities = &evaluate(&writer.snapshot(), &[term.untyped()], given, &target)[0];
        let integral: f64 =
            densities.iter().zip(&xs).map(|(l, x)| l.exp() * x[0]).sum::<f64>() * step;
        close(&[integral], &[1.0], relative);
    }
}
