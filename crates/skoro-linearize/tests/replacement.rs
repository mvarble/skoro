#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)] // a test crate
//! A retirement's replacement, linearized: the Kalman filter and its extensions, and improper
//! replacements of what the retired terms do not determine.

mod common;

use common::{close, host, points};
use skoro_core::scope::{constant, quantity};
use skoro_core::{
    Addresses, NodeId, Observations, Q, QDyn, Query, Real, Replacement, Rows, Snapshot, TermHandle,
    Values, Writer,
};
use skoro_linearize::{Error, Replacing};
use skoro_normal::{Normal, Scale};
use skoro_tensor::Target;

const A: f64 = 0.3;
const S: f64 = 1.1;
const R: f64 = 0.5;
const QN: f64 = 0.7;
const Z0: f64 = 0.9;

/// The core's scalar chain: `x0 ~ N(a, s²)`, `z0 | x0 ~ N(x0, r²)`, `x1 | x0 ~ N(x0, q²)`,
/// `z1 | x1 ~ N(x1, r²)`, each a normal.
fn chain() -> (Writer, [Q<Real>; 4]) {
    let mut writer = Writer::new();
    let quantities = writer
        .build(|| {
            let x0 = quantity("tick/0/x", Real);
            Normal::on(Real, constant(Real, A), Scale::sd(S)).score(x0);
            let z0 = quantity("tick/0/z", Real);
            Normal::on(Real, x0, Scale::sd(R)).score(z0);
            let x1 = quantity("tick/1/x", Real);
            Normal::on(Real, x0, Scale::sd(QN)).score(x1);
            let z1 = quantity("tick/1/z", Real);
            Normal::on(Real, x1, Scale::sd(R)).score(z1);
            [x0, z0, x1, z1]
        })
        .unwrap();
    (writer, quantities)
}

#[test]
fn a_mismatched_blanket_is_refused() {
    let target = Target::host();
    let (mut writer, [x0, z0, x1, z1]) = chain();
    let observed = Observations::new().observe(z0, Z0);
    let retirement = writer
        .retire(Addresses::new().under("tick/0"), observed)
        .unwrap();
    // A kept quantity outside the blanket.
    let error =
        Replacing::new(&retirement, &[z1.untyped()], &Observations::new(), &target).unwrap_err();
    assert_eq!(error, Error::Blanket { quantity: z1.id() });
    // A blanket quantity neither kept nor given a value: another integral.
    let error = Replacing::new(&retirement, &[], &Observations::new(), &target).unwrap_err();
    assert_eq!(error, Error::Blanket { quantity: x1.id() });
    // A kept quantity that is also given a value.
    let fixed = Observations::new().observe(x1, 0.2);
    let error = Replacing::new(&retirement, &[x1.untyped()], &fixed, &target).unwrap_err();
    assert_eq!(error, Error::Blanket { quantity: x1.id() });
    // A value the caller gives again for z0, which the retirement observes, does not make it
    // free or change what is integrated.
    let again = Observations::new().observe(z0, 5.0);
    let replacing = Replacing::new(&retirement, &[x1.untyped()], &again, &target).unwrap();
    assert_eq!(replacing.integrated(), [x0.id()]);
}

/// `log 𝒩(x; m, s²)`.
fn log_normal(x: f64, m: f64, s: f64) -> f64 {
    -0.5 * ((x - m) / s).powi(2) - s.ln() - 0.5 * (2.0 * std::f64::consts::PI).ln()
}

/// The log-density of the terms `terms` of `snapshot`, with `given` observed: one value per row.
fn density_of(snapshot: &Snapshot, terms: &[NodeId], given: Observations) -> Vec<f64> {
    let query = Query::log_density_of(terms.iter().copied()).given(given);
    let program = snapshot.compile(&query, &Target::host()).unwrap();
    host(&program.forward(&Values::new()).unwrap(), NodeId::DENSITY)
}

/// Retires `addresses` with `observed`, replacing them by the linearized replacement at `at`,
/// keeping `keep`: the factor added.
fn retire(
    writer: &mut Writer,
    addresses: Addresses,
    observed: Observations,
    keep: &[QDyn],
    at: &Values,
) -> TermHandle {
    let target = Target::host();
    let retirement = writer.retire(addresses, observed).unwrap();
    let replacing = Replacing::new(&retirement, keep, &Observations::new(), &target).unwrap();
    let linearized = replacing.at(at).unwrap();
    retirement.replace(|_| linearized.build()).unwrap().unwrap()
}

const XS: [f64; 4] = [-1.0, 0.2, 0.9, 2.4];

#[test]
fn retiring_the_scalar_chain_is_the_kalman_filter() {
    let target = Target::host();
    // [](eq:archive/core/math/kalman): the evidence of ẑ0, and the predicted mean and variance.
    let gain = S * S / (S * S + R * R);
    let mean = A + gain * (Z0 - A);
    let variance = (1.0 - gain) * S * S + QN * QN;
    let evidence = log_normal(Z0, A, (S * S + R * R).sqrt());
    for linearized_at in [[0.0, 0.0], [1.3, -0.4]] {
        let (mut writer, [x0, z0, x1, _]) = chain();
        let at = Values::new()
            .with(x0, points(&[linearized_at[0]], 1, 1, &target))
            .with(x1, points(&[linearized_at[1]], 1, 1, &target));
        let observed = Observations::new().observe(z0, Z0);
        let factor = retire(
            &mut writer,
            Addresses::new().under("tick/0"),
            observed,
            &[x1.untyped()],
            &at,
        );
        let values = density_of(
            &writer.snapshot(),
            &[factor.id()],
            Observations::new().observe(x1, Rows(XS.to_vec())),
        );
        let expected: Vec<f64> = XS
            .iter()
            .map(|&x| evidence + log_normal(x, mean, variance.sqrt()))
            .collect();
        close(
            &values,
            &expected,
            1e-10,
            "the Kalman filter's update and prediction",
        );
    }
    // Retiring x0 alone keeps (z0, x1): the joint normal of [](eq:archive/core/math/joint-blanket),
    // of mean (a, a) and covariance [s² + r², s²; s², s² + q²].
    let (mut writer, [x0, z0, x1, _]) = chain();
    let at = Values::new()
        .with(x0, points(&[0.5], 1, 1, &target))
        .with(z0, points(&[0.1], 1, 1, &target))
        .with(x1, points(&[-0.2], 1, 1, &target));
    let factor = retire(
        &mut writer,
        Addresses::new().at("tick/0/x"),
        Observations::new(),
        &[z0.untyped(), x1.untyped()],
        &at,
    );
    let (zs, xs) = ([0.4, -0.3, 1.2], [0.1, 0.8, -0.9]);
    let values = density_of(
        &writer.snapshot(),
        &[factor.id()],
        Observations::new()
            .observe(z0, Rows(zs.to_vec()))
            .observe(x1, Rows(xs.to_vec())),
    );
    let (szz, sxx, szx) = (S * S + R * R, S * S + QN * QN, S * S);
    let det = szz * sxx - szx * szx;
    let expected: Vec<f64> = zs
        .iter()
        .zip(&xs)
        .map(|(&z, &x)| {
            let (u, v) = (z - A, x - A);
            -0.5 * (sxx * u * u - 2.0 * szx * u * v + szz * v * v) / det
                - 0.5 * det.ln()
                - (2.0 * std::f64::consts::PI).ln()
        })
        .collect();
    close(&values, &expected, 1e-10, "the joint normal");
    // The model after the retirement gives the log-density the model retired with the exact
    // replacement gives, the term of z1 included.
    let (mut linearized, [x0, z0, x1, z1]) = chain();
    let at = Values::new()
        .with(x0, points(&[0.0], 1, 1, &target))
        .with(x1, points(&[0.0], 1, 1, &target));
    retire(
        &mut linearized,
        Addresses::new().under("tick/0"),
        Observations::new().observe(z0, Z0),
        &[x1.untyped()],
        &at,
    );
    let (mut exact, [_, z0e, x1e, z1e]) = chain();
    exact
        .retire(
            Addresses::new().under("tick/0"),
            Observations::new().observe(z0e, Z0),
        )
        .unwrap()
        .replace(|_| {
            let parts =
                Normal::on(Real, constant(Real, mean), Scale::sd(variance.sqrt())).log_density(x1e);
            Replacement {
                log_density: (parts.log_density.untyped() + evidence).narrow::<Real>(),
                proposals: Vec::new(),
                bindings: Vec::new(),
            }
        })
        .unwrap();
    let z1s = [0.3, -0.7, 1.1, 0.0];
    let whole = |writer: &Writer, x1: Q<Real>, z1: Q<Real>| {
        let snapshot = writer.snapshot();
        let terms: Vec<NodeId> = snapshot.terms().iter().map(|t| t.log_density).collect();
        density_of(
            &snapshot,
            &terms,
            Observations::new()
                .observe(x1, Rows(XS.to_vec()))
                .observe(z1, Rows(z1s.to_vec())),
        )
    };
    close(
        &whole(&linearized, x1, z1),
        &whole(&exact, x1e, z1e),
        1e-10,
        "the retired models",
    );
}

#[test]
fn retiring_a_chain_one_state_at_a_time_is_the_kalman_filter() {
    let target = Target::host();
    let zs = [0.9, 0.4, 1.6, 1.1];
    let mut writer = Writer::new();
    let states = writer
        .build(|| {
            let mut states = Vec::new();
            let mut previous: Option<Q<Real>> = None;
            for (k, _) in zs.iter().enumerate() {
                let x = quantity(format!("tick/{k}/x"), Real);
                match previous {
                    None => Normal::on(Real, constant(Real, A), Scale::sd(S)).score(x),
                    Some(p) => Normal::on(Real, p, Scale::sd(QN)).score(x),
                };
                let z = quantity(format!("tick/{k}/z"), Real);
                Normal::on(Real, x, Scale::sd(R)).score(z);
                states.push((x, z));
                previous = Some(x);
            }
            states
        })
        .unwrap();
    // The Kalman filter, as the replacements should accumulate it.
    let (mut m, mut p, mut evidence) = (A, S * S, 0.0);
    for k in 0..zs.len() - 1 {
        evidence += log_normal(zs[k], m, (p + R * R).sqrt());
        let gain = p / (p + R * R);
        m += gain * (zs[k] - m);
        p = (1.0 - gain) * p + QN * QN;
        let (x, z) = states[k];
        let next = states[k + 1].0;
        let at = Values::new()
            .with(x, points(&[0.3 * k as f64], 1, 1, &target))
            .with(next, points(&[-0.5], 1, 1, &target));
        let observed = Observations::new().observe(z, zs[k]);
        let factor = retire(
            &mut writer,
            Addresses::new().under(format!("tick/{k}")),
            observed,
            &[next.untyped()],
            &at,
        );
        let values = density_of(
            &writer.snapshot(),
            &[factor.id()],
            Observations::new().observe(next, Rows(XS.to_vec())),
        );
        let expected: Vec<f64> = XS
            .iter()
            .map(|&x| evidence + log_normal(x, m, p.sqrt()))
            .collect();
        close(
            &values,
            &expected,
            1e-10,
            &format!("after retiring tick {k}"),
        );
    }
}

#[test]
fn an_undetermined_retired_quantity_is_refused() {
    use skoro_core::{Matrix, Vector};
    let target = Target::host();
    // x0 in the plane, seen along one direction alone by x1: its retirement integrates over a
    // direction no term determines.
    let mut writer = Writer::new();
    let (x0, x1) = writer
        .build(|| {
            let x0 = quantity("x0", Vector::<2>);
            let x1 = quantity("x1", Vector::<1>);
            let along = constant(Matrix::<1, 2>, [[0.6, 0.8]]).matmul(x0);
            Normal::on(Vector::<1>, along, Scale::sd(0.1)).score(x1);
            (x0, x1)
        })
        .unwrap();
    let retirement = writer
        .retire(Addresses::new().at("x0"), Observations::new())
        .unwrap();
    let replacing =
        Replacing::new(&retirement, &[x1.untyped()], &Observations::new(), &target).unwrap();
    let at = Values::new()
        .with(x0, points(&[0.1, 0.2], 1, 2, &target))
        .with(x1, points(&[0.3], 1, 1, &target));
    assert_eq!(
        replacing.at(&at).unwrap_err(),
        Error::RetiredUndetermined { quantity: x0.id() }
    );
}
