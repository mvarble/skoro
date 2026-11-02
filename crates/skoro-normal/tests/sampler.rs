#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)] // a test crate
//! The normal's sampler: the quantile it draws through, its draws, and its weight.

mod common;

use common::{cholesky, evaluate, host, inverse};
use skoro_core::scope::{constant, quantity};
use skoro_core::{
    Address, Categorical, Error, Key, NodeId, Observations, Positive, Query, Real, Snapshot,
    Values, Vector, Writer,
};
use skoro_normal::{Normal, Scale, quantile};
use skoro_tensor::{Precision, Target};

#[test]
fn quantile_matches_reference() {
    // Φ⁻¹ by root-finding at 50 digits, independently of AS 241.
    let reference = [
        (0.5, 0.0),
        (0.3, -0.524_400_512_708_040_8),
        (0.975, 1.959_963_984_540_054_2),
        (0.025, -1.959_963_984_540_054_2),
        (0.999, 3.090_232_306_167_813_5),
        (1e-10, -6.361_340_902_404_056),
        (1e-300, -37.047_096_299_361_2),
    ];
    for (p, x) in reference {
        let error = (quantile(p) - x).abs() / f64::abs(x).max(1.0);
        assert!(error < 1e-15, "Φ⁻¹({p}) = {} against {x}", quantile(p));
    }
}

/// A covariance of side 2, row by row.
const SIGMA2: [f64; 4] = [0.5, 0.2, 0.2, 0.3];
/// A covariance of side 3, row by row.
const SIGMA3: [f64; 9] = [0.8, 0.2, -0.1, 0.2, 0.5, 0.15, -0.1, 0.15, 0.6];

/// The keys of the quantity at `address`, one per row.
fn keys(key: &Key, rows: usize, address: &str) -> Vec<Key> {
    let address = Address::from(address);
    key.split(rows).iter().map(|k| k.at(&address)).collect()
}

fn draws(
    snapshot: &Snapshot,
    q: impl Into<NodeId>,
    key: &Key,
    rows: usize,
    target: &Target,
) -> Vec<f64> {
    let generated = snapshot
        .generate(&Observations::new(), key, rows, target)
        .unwrap();
    host(&generated.values, q.into())
}

#[test]
fn draws_are_functions_of_key_and_address() {
    let mean = [0.4, -1.0];
    let mut writer = Writer::new();
    let (x, y) = writer
        .build(|| {
            let mu = constant(Vector::<2>, mean);
            let (x, y) = (quantity("x", Vector::<2>), quantity("y", Vector::<2>));
            Normal::on(Vector::<2>, mu, Scale::covariance(&SIGMA2, 2)).score(x);
            Normal::on(Vector::<2>, mu, Scale::covariance(&SIGMA2, 2)).score(y);
            (x, y)
        })
        .unwrap();
    let snapshot = writer.snapshot();
    let (key, rows, target) = (Key::from_seed(11), 6, Target::host());
    let first = draws(&snapshot, x, &key, rows, &target);
    let again = draws(&snapshot, x, &key, rows, &target);
    let bits = |v: &[f64]| v.iter().map(|x| x.to_bits()).collect::<Vec<_>>();
    assert_eq!(bits(&first), bits(&again));
    assert_ne!(first, draws(&snapshot, y, &key, rows, &target));

    // Each row is μ + L Φ⁻¹(u), u the variates of that row's key.
    let l = cholesky(&SIGMA2, 2);
    for (row, key) in keys(&key, rows, "x").iter().enumerate() {
        let zeta: Vec<f64> = key.uniforms(2).into_iter().map(quantile).collect();
        let want = [
            mean[0] + l[0] * zeta[0],
            mean[1] + l[2] * zeta[0] + l[3] * zeta[1],
        ];
        for i in 0..2 {
            assert!(
                (first[row * 2 + i] - want[i]).abs() < 1e-14,
                "{first:?} against {want:?}"
            );
        }
    }
}

/// The sample mean and covariance of `rows` draws of storage `d`, row by row.
fn moments(data: &[f64], d: usize) -> (Vec<f64>, Vec<f64>) {
    let n = data.len() / d;
    let mean: Vec<f64> = (0..d)
        .map(|i| data.iter().skip(i).step_by(d).sum::<f64>() / n as f64)
        .collect();
    let covariance = (0..d * d)
        .map(|k| {
            let (i, j) = (k / d, k % d);
            data.chunks(d)
                .map(|x| (x[i] - mean[i]) * (x[j] - mean[j]))
                .sum::<f64>()
                / n as f64
        })
        .collect();
    (mean, covariance)
}

/// The sample moments within five standard errors of `mean` and `sigma`.
fn within(data: &[f64], mean: &[f64], sigma: &[f64]) {
    let d = mean.len();
    let n = (data.len() / d) as f64;
    let (m, c) = moments(data, d);
    for i in 0..d {
        let se = (sigma[i * d + i] / n).sqrt();
        assert!(
            (m[i] - mean[i]).abs() < 5.0 * se,
            "mean {m:?} against {mean:?}"
        );
        for j in 0..d {
            let s = |a: usize, b: usize| sigma[a * d + b];
            let se = ((s(i, i) * s(j, j) + s(i, j).powi(2)) / n).sqrt();
            assert!(
                (c[i * d + j] - s(i, j)).abs() < 5.0 * se,
                "covariance {c:?} against {sigma:?}"
            );
        }
    }
}

skoro_core::precision_tests! {
    fn draws_have_the_normal_moments(target) {
        let rows = 100_000;
        let mean = [0.3, -0.2, 1.0];
        let sds: [f64; 3] = [0.4, 0.7, 0.2];
        let diagonal: Vec<f64> =
            (0..9).map(|k| if k % 4 == 0 { sds[k / 4].powi(2) } else { 0.0 }).collect();
        let isotropic: Vec<f64> = (0..9).map(|k| if k % 4 == 0 { 0.25 } else { 0.0 }).collect();
        let mut writer = Writer::new();
        let quantities = writer
            .build(|| {
                let mu = constant(Vector::<3>, mean);
                let scales = [
                    (Scale::sd(0.5), isotropic.clone()),
                    (Scale::sds(&sds), diagonal.clone()),
                    (Scale::covariance(&SIGMA3, 3), SIGMA3.to_vec()),
                    (Scale::precision(&inverse(&SIGMA3, 3), 3), SIGMA3.to_vec()),
                ];
                scales
                    .into_iter()
                    .enumerate()
                    .map(|(i, (scale, sigma))| {
                        let x = quantity(format!("x{i}"), Vector::<3>);
                        Normal::on(Vector::<3>, mu, scale).score(x);
                        (x, sigma)
                    })
                    .collect::<Vec<_>>()
            })
            .unwrap();
        let generated = writer
            .snapshot()
            .generate(&Observations::new(), &Key::from_seed(5), rows, &target)
            .unwrap();
        for (x, sigma) in &quantities {
            within(&host(&generated.values, x.id()), &mean, sigma);
        }

        // On the positive orthant, log x has mean log μ and covariance Σ.
        let mut writer = Writer::new();
        let x = writer
            .build(|| {
                let x = quantity("x", Positive::<2>);
                let mu = constant(Positive::<2>, [1.5, 0.4]);
                Normal::on(Positive::<2>, mu, Scale::covariance(&SIGMA2, 2)).score(x);
                x
            })
            .unwrap();
        let generated = writer
            .snapshot()
            .generate(&Observations::new(), &Key::from_seed(6), rows, &target)
            .unwrap();
        let logs: Vec<f64> = host(&generated.values, x.id()).iter().map(|x| x.ln()).collect();
        within(&logs, &[f64::ln(1.5), f64::ln(0.4)], &SIGMA2);
    }

    fn exact_sampler_leaves_the_weight_to_the_likelihoods(target) {
        let mut writer = Writer::new();
        let (x0, x1, z, tz) = writer
            .build(|| {
                let x0 = quantity("x0", Real);
                Normal::on(Real, constant(Real, 0.0), Scale::sd(1.0)).score(x0);
                let x1 = quantity("x1", Real);
                Normal::on(Real, x0, Scale::sd(0.5)).score(x1);
                let z = quantity("z", Real);
                let tz = Normal::on(Real, x1, Scale::sd(0.3)).score(z);
                (x0, x1, z, tz)
            })
            .unwrap();
        let snapshot = writer.snapshot();
        let given = Observations::new().observe(z, 0.7);
        let generated = snapshot.generate(&given, &Key::from_seed(7), 5, &target).unwrap();
        let program = snapshot
            .compile(&Query::log_density_of([tz.id()]).given(given), &target)
            .unwrap();
        let at = Values::new()
            .with(x0, generated.values.get(x0).unwrap().clone())
            .with(x1, generated.values.get(x1).unwrap().clone());
        let likelihood = host(&program.forward(&at).unwrap(), NodeId::DENSITY);
        let weight = generated.log_weight.to_host();
        let bits = |v: &[f64]| v.iter().map(|x| x.to_bits()).collect::<Vec<_>>();
        assert_eq!(bits(&weight), bits(&likelihood), "{weight:?} and {likelihood:?}");
    }
}

#[test]
fn same_key_same_draw_at_both_precisions() {
    let mut writer = Writer::new();
    let (x, p) = writer
        .build(|| {
            let x = quantity("x", Vector::<3>);
            Normal::on(
                Vector::<3>,
                constant(Vector::<3>, [0.1, 2.0, -3.0]),
                Scale::covariance(&SIGMA3, 3),
            )
            .score(x);
            let p = quantity("p", Positive::<2>);
            Normal::on(
                Positive::<2>,
                constant(Positive::<2>, [1.5, 0.4]),
                Scale::sds(&[0.3, 0.2]),
            )
            .score(p);
            (x, p)
        })
        .unwrap();
    let snapshot = writer.snapshot();
    let key = Key::from_seed(9);
    let single = skoro_core::testing::flex(Precision::Single).unwrap();
    let double = skoro_core::testing::flex(Precision::Double).unwrap();
    for q in [x.id(), p.id()] {
        let a = draws(&snapshot, q, &key, 50, &single);
        let b = draws(&snapshot, q, &key, 50, &double);
        for (a, b) in a.iter().zip(&b) {
            // The mean read at single precision and the final cast, and nothing else.
            assert!(
                (a - b).abs() <= 4.0 * f64::from(f32::EPSILON) * b.abs().max(1.0),
                "{a} and {b}"
            );
        }
    }
}

#[test]
fn refuses_spaces_it_does_not_take() {
    let line = line!() + 5;
    let failure = Writer::new()
        .build(|| {
            let k = quantity("k", Categorical::<3>);
            let mean = constant(Categorical::<3>, 1usize);
            Normal::on(Categorical::<3>, mean, Scale::sd(1.0)).score(k);
        })
        .unwrap_err();
    assert_eq!(
        failure.error,
        Error::NotContinuous {
            space: skoro_core::Space::Categorical(3)
        }
    );
    assert_eq!(failure.location.line(), line);
    let _ = evaluate; // shared with the other tests
}
