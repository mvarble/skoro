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
use skoro_linearize::{Assembly, Error, LinearizedReplacement, Linearizer, Replacing};
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

/// Retires `addresses` with `observed`, keeping `keep`: the linearized replacement at `at`, and the
/// factor it adds.
fn retire_with(
    writer: &mut Writer,
    addresses: Addresses,
    observed: Observations,
    keep: &[QDyn],
    at: &Values,
) -> (LinearizedReplacement, TermHandle) {
    let target = Target::host();
    let retirement = writer.retire(addresses, observed).unwrap();
    let replacing = Replacing::new(&retirement, keep, &Observations::new(), &target).unwrap();
    let linearized = replacing.at(at).unwrap();
    let factor = retirement.replace(|_| linearized.build()).unwrap().unwrap();
    (linearized, factor)
}

/// The Gaussian of a replacement of one row on one scalar: `-½(A δ - b)²` with `δ = x - x̂`, of
/// mean `x̂ + b/A` and variance `1/A²`.
fn gaussian(linearized: &LinearizedReplacement, anchor: f64) -> (f64, f64) {
    let reduced = &linearized.reduced()[0];
    assert!(reduced.linear.is_none());
    let (a, b) = (reduced.a.get(0, 0), reduced.b[0]);
    (anchor + b / a, 1.0 / (a * a))
}

#[test]
fn a_nonlinear_chain_is_the_extended_kalman_filter() {
    let target = Target::host();
    // x0 ~ N(a, s²), z0 ~ N(sin x0, r²), x1 ~ N(g(x0), q²) with g(x) = x + 0.3 x².
    let (g, dg) = (|x: f64| x + 0.3 * x * x, |x: f64| 1.0 + 0.6 * x);
    let build = || {
        let mut writer = Writer::new();
        let q = writer
            .build(|| {
                let x0 = quantity("tick/0/x", Real);
                Normal::on(Real, constant(Real, A), Scale::sd(S)).score(x0);
                let z0 = quantity("tick/0/z", Real);
                Normal::on(Real, x0.sin(), Scale::sd(R)).score(z0);
                let x1 = quantity("tick/1/x", Real);
                Normal::on(Real, x0 * x0 * 0.3 + x0, Scale::sd(QN)).score(x1);
                [x0, z0, x1]
            })
            .unwrap();
        (writer, q)
    };
    let observed = |z0: Q<Real>| Observations::new().observe(z0, Z0);
    // At the prediction: the measurement linearized at x̂0 = a, the motion at x̂0 = a too.
    let (mut writer, [x0, z0, x1]) = build();
    let at = Values::new()
        .with(x0, points(&[A], 1, 1, &target))
        .with(x1, points(&[g(A)], 1, 1, &target));
    let (linearized, _) = retire_with(
        &mut writer,
        Addresses::new().under("tick/0"),
        observed(z0),
        &[x1.untyped()],
        &at,
    );
    let (m, p) = gaussian(&linearized, g(A));
    let h = A.cos();
    let gain = S * S * h / (h * h * S * S + R * R);
    let update = A + gain * (Z0 - A.sin());
    let covariance = (1.0 - gain * h) * S * S;
    // The motion's first-order expansion about the prediction, evaluated at the update, and the
    // covariance propagated by its Jacobian at the prediction.
    close(
        &[m],
        &[g(A) + dg(A) * (update - A)],
        1e-10,
        "the predicted mean",
    );
    close(
        &[p],
        &[dg(A).powi(2) * covariance + QN * QN],
        1e-10,
        "the predicted variance",
    );
    // The classical filter's mean differs at second order in the update's step, its variance at
    // first.
    let step = update - A;
    close(
        &[g(update) - m],
        &[0.3 * step * step],
        1e-10,
        "second order",
    );
    let classical = dg(update).powi(2) * covariance + QN * QN;
    close(
        &[classical - p],
        &[(dg(update).powi(2) - dg(A).powi(2)) * covariance],
        1e-10,
        "first order",
    );
    // At the retired terms' mode, found by Gauss–Newton on them: the iterated filter's step.
    let (mut writer, [x0, z0, x1]) = build();
    let retirement = writer
        .retire(Addresses::new().under("tick/0"), observed(z0))
        .unwrap();
    let terms: Vec<NodeId> = retirement
        .plan()
        .terms
        .iter()
        .map(|t| t.log_density)
        .collect();
    let query = Query::log_density_of(terms).given(observed(z0));
    let linearizer =
        Linearizer::new(retirement.snapshot(), &query, &target, Assembly::Residual).unwrap();
    let schedule = linearizer.schedule().unwrap();
    let mut mode = Values::new()
        .with(x0, points(&[A], 1, 1, &target))
        .with(x1, points(&[g(A)], 1, 1, &target));
    for _ in 0..50 {
        let step = linearizer
            .at(&mode)
            .unwrap()
            .eliminate(&schedule, None)
            .unwrap()
            .minimizer(&target)
            .unwrap();
        mode = linearizer.layout().retract(&mode, &step, &target).unwrap();
    }
    let replacing =
        Replacing::new(&retirement, &[x1.untyped()], &Observations::new(), &target).unwrap();
    let linearized = replacing.at(&mode).unwrap();
    let anchor = host(&mode, x1)[0];
    let (m, p) = gaussian(&linearized, anchor);
    let mut x = A;
    for _ in 0..100 {
        let h = x.cos();
        let gain = S * S * h / (h * h * S * S + R * R);
        x = A + gain * (Z0 - x.sin() - h * (A - x));
    }
    let h = x.cos();
    let gain = S * S * h / (h * h * S * S + R * R);
    let covariance = (1.0 - gain * h) * S * S;
    close(&[host(&mode, x0)[0]], &[x], 1e-10, "the update's mode");
    close(&[m], &[g(x)], 1e-10, "the iterated filter's mean");
    close(
        &[p],
        &[dg(x).powi(2) * covariance + QN * QN],
        1e-10,
        "the iterated filter's variance",
    );
}

/// The planar stand-in for a bearing: a landmark `ℓ`, and for `k` in `ticks` a pose `p_k ~ N(m_k,
/// 0.5² I)` and a measurement `z_k ~ N(a_kᵀ(ℓ - p_k), σ²)` along a known unit direction.
/// A tick of the planar example: its pose and its measurement.
type Tick = (Q<skoro_core::Vector<2>>, Q<skoro_core::Vector<1>>);

fn planar() -> (Writer, Q<skoro_core::Vector<2>>, Vec<Tick>) {
    use skoro_core::{Matrix, Vector};
    let mut writer = Writer::new();
    let (landmark, ticks) = writer
        .build(|| {
            let landmark = quantity("landmark", Vector::<2>);
            let ticks = PLANAR
                .iter()
                .enumerate()
                .map(|(k, (m, a, _))| {
                    let p = quantity(format!("tick/{k}/p"), Vector::<2>);
                    Normal::on(Vector::<2>, constant(Vector::<2>, *m), Scale::sd(0.5)).score(p);
                    let z = quantity(format!("tick/{k}/z"), Vector::<1>);
                    let along = constant(Matrix::<1, 2>, [*a]).matmul(landmark - p);
                    Normal::on(Vector::<1>, along, Scale::sd(SIGMA)).score(z);
                    (p, z)
                })
                .collect();
            (landmark, ticks)
        })
        .unwrap();
    (writer, landmark, ticks)
}

const SIGMA: f64 = 0.2;
/// Each tick's prior mean, direction and measurement.
const PLANAR: [([f64; 2], [f64; 2], f64); 2] = [
    ([0.0, 0.0], [0.6, 0.8], 2.1),
    ([3.0, 0.5], [-0.8, 0.6], 0.7),
];

#[test]
fn improper_replacements_accumulate() {
    let target = Target::host();
    let ls = [[0.4, 1.5], [1.2, 1.9], [-0.3, 0.8]];
    let at_landmark = |landmark| Observations::new().observe(landmark, Rows(ls.to_vec()));
    // One tick at a time: each leaves one row on ℓ.
    let (mut writer, landmark, ticks) = planar();
    let mut factors = Vec::new();
    for (k, &(p, z)) in ticks.iter().enumerate() {
        let at = Values::new()
            .with(landmark, points(&[0.5, 0.5], 1, 2, &target))
            .with(p, points(&PLANAR[k].0, 1, 2, &target));
        let observed = Observations::new().observe(z, [PLANAR[k].2]);
        let (linearized, factor) = retire_with(
            &mut writer,
            Addresses::new().under(format!("tick/{k}")),
            observed,
            &[landmark.untyped()],
            &at,
        );
        assert_eq!(linearized.rows(), 1, "tick {k} leaves one row");
        factors.push(factor);
        if k == 0 {
            // Alone, the first replacement does not determine ℓ.
            let query = Query::log_density_of([factor.id()]);
            let linearizer =
                Linearizer::new(&writer.snapshot(), &query, &target, Assembly::Residual).unwrap();
            let point = Values::new().with(landmark, points(&[0.5, 0.5], 1, 2, &target));
            let error = linearizer
                .at(&point)
                .unwrap()
                .eliminate(&linearizer.schedule().unwrap(), None)
                .unwrap_err();
            assert_eq!(
                error,
                Error::Undetermined {
                    quantity: landmark.id()
                }
            );
        }
    }
    let ids: Vec<NodeId> = factors.iter().map(|f| f.id()).collect();
    let summed = density_of(&writer.snapshot(), &ids, at_landmark(landmark));
    // Both ticks at once.
    let (mut both, landmark2, ticks2) = planar();
    let mut at = Values::new().with(landmark2, points(&[0.5, 0.5], 1, 2, &target));
    let mut observed = Observations::new();
    for (k, &(p, z)) in ticks2.iter().enumerate() {
        at = at.with(p, points(&PLANAR[k].0, 1, 2, &target));
        observed = observed.observe(z, [PLANAR[k].2]);
    }
    let (_, joint) = retire_with(
        &mut both,
        Addresses::new().under("tick"),
        observed,
        &[landmark2.untyped()],
        &at,
    );
    let joint = density_of(&both.snapshot(), &[joint.id()], at_landmark(landmark2));
    close(&summed, &joint, 1e-10, "the sum is the joint replacement");
    // And the two together give the exact posterior of ℓ: z_k | ℓ ~ N(a_kᵀ(ℓ - m_k), v_k).
    let query = Query::log_density_of(ids);
    let linearizer =
        Linearizer::new(&writer.snapshot(), &query, &target, Assembly::Residual).unwrap();
    let point = Values::new().with(landmark, points(&[0.5, 0.5], 1, 2, &target));
    let schedule = linearizer.schedule_keeping(&[landmark.id()]).unwrap();
    let marginal = &linearizer
        .at(&point)
        .unwrap()
        .eliminate(&schedule, None)
        .unwrap()
        .marginal()
        .unwrap()[0];
    let mut precision = [[0.0; 2]; 2];
    let mut right = [0.0; 2];
    for (m, a, z) in PLANAR {
        let v = 0.25 + SIGMA * SIGMA;
        let am: f64 = a[0] * m[0] + a[1] * m[1];
        for i in 0..2 {
            right[i] += a[i] * (z + am) / v;
            for j in 0..2 {
                precision[i][j] += a[i] * a[j] / v;
            }
        }
    }
    let det = precision[0][0] * precision[1][1] - precision[0][1] * precision[1][0];
    let covariance = [
        precision[1][1] / det,
        -precision[0][1] / det,
        -precision[1][0] / det,
        precision[0][0] / det,
    ];
    let mean = [
        covariance[0] * right[0] + covariance[1] * right[1],
        covariance[2] * right[0] + covariance[3] * right[1],
    ];
    close(
        marginal.marginal.covariance().data(),
        &covariance,
        1e-10,
        "the posterior covariance",
    );
    let offset: Vec<f64> = marginal
        .marginal
        .mean
        .iter()
        .zip([0.5, 0.5])
        .map(|(m, p)| m + p)
        .collect();
    close(&offset, &mean, 1e-10, "the posterior mean");
}

#[test]
fn a_bearing_replacement_is_the_linearized_integral() {
    let target = Target::host();
    let (sp, sigma, z_hat) = (0.3, 0.05, 0.9);
    let mut writer = Writer::new();
    let [lx, ly, px, py, z] = writer
        .build(|| {
            let lx = quantity("lx", Real);
            let ly = quantity("ly", Real);
            let px = quantity("tick/px", Real);
            Normal::on(Real, constant(Real, 0.1), Scale::sd(sp)).score(px);
            let py = quantity("tick/py", Real);
            Normal::on(Real, constant(Real, -0.2), Scale::sd(sp)).score(py);
            let z = quantity("tick/z", Real);
            Normal::on(Real, (ly - py).atan2(lx - px), Scale::sd(sigma)).score(z);
            [lx, ly, px, py, z]
        })
        .unwrap();
    let (l, p) = ([2.0, 1.5], [0.15, -0.1]);
    let at = Values::new()
        .with(lx, points(&[l[0]], 1, 1, &target))
        .with(ly, points(&[l[1]], 1, 1, &target))
        .with(px, points(&[p[0]], 1, 1, &target))
        .with(py, points(&[p[1]], 1, 1, &target));
    let observed = Observations::new().observe(z, z_hat);
    let (_, factor) = retire_with(
        &mut writer,
        Addresses::new().under("tick"),
        observed,
        &[lx.untyped(), ly.untyped()],
        &at,
    );
    // The same linearized integral, densely: τ = (τ_px, τ_py, τ_lx, τ_ly).
    let (dx, dy) = (l[0] - p[0], l[1] - p[1]);
    let rho2 = dx * dx + dy * dy;
    let bearing = [dy / rho2, -dx / rho2, -dy / rho2, dx / rho2]; // ∂θ/∂τ
    let rows = [
        [1.0 / sp, 0.0, 0.0, 0.0],
        [0.0, 1.0 / sp, 0.0, 0.0],
        bearing.map(|b| -b / sigma),
    ];
    let residuals = [
        (p[0] - 0.1) / sp,
        (p[1] + 0.2) / sp,
        (z_hat - dy.atan2(dx)) / sigma,
    ];
    let normalizers = 2.0 * (-sp.ln()) - sigma.ln() - 1.5 * (2.0 * std::f64::consts::PI).ln();
    for tau_l in [[0.0, 0.0], [0.1, -0.05], [-0.2, 0.15]] {
        // min over τ_E of ½|J_E τ_E + J_K τ_K + r|², and log ∫ = log 2π - ½ log det(J_EᵀJ_E) - (that - Σ n_t).
        let c: Vec<f64> = (0..3)
            .map(|i| rows[i][2] * tau_l[0] + rows[i][3] * tau_l[1] + residuals[i])
            .collect();
        let g = [[0.0; 2]; 2];
        let mut g = g;
        let mut rhs = [0.0; 2];
        for i in 0..3 {
            for a in 0..2 {
                rhs[a] -= rows[i][a] * c[i];
                for b in 0..2 {
                    g[a][b] += rows[i][a] * rows[i][b];
                }
            }
        }
        let det = g[0][0] * g[1][1] - g[0][1] * g[1][0];
        let e = [
            (g[1][1] * rhs[0] - g[0][1] * rhs[1]) / det,
            (g[0][0] * rhs[1] - g[1][0] * rhs[0]) / det,
        ];
        let minimum: f64 = (0..3)
            .map(|i| 0.5 * (rows[i][0] * e[0] + rows[i][1] * e[1] + c[i]).powi(2))
            .sum();
        let expected = (2.0 * std::f64::consts::PI).ln() - 0.5 * det.ln() - (minimum - normalizers);
        let given = Observations::new()
            .observe(lx, l[0] + tau_l[0])
            .observe(ly, l[1] + tau_l[1]);
        let value = density_of(&writer.snapshot(), &[factor.id()], given);
        close(&value, &[expected], 1e-10, "the linearized integral");
    }
}

#[test]
fn the_positive_replacement_is_exact() {
    use skoro_core::Positive;
    let target = Target::host();
    let sigma = 0.3;
    let mut writer = Writer::new();
    let (t, f, x) = writer
        .build(|| {
            let t = quantity("theta", Real);
            Normal::on(Real, constant(Real, 0.0), Scale::sd(1.0)).score(t);
            let f = quantity("phi", Real);
            let x = quantity("x", Positive::<1>);
            Normal::on(Positive::<1>, (t + f).exp_positive(), Scale::sd(sigma)).score(x);
            (t, f, x)
        })
        .unwrap();
    let at = Values::new()
        .with(t, points(&[0.2], 1, 1, &target))
        .with(f, points(&[-0.4], 1, 1, &target))
        .with(x, points(&[1.3], 1, 1, &target));
    let (linearized, factor) = retire_with(
        &mut writer,
        Addresses::new().at("theta"),
        Observations::new(),
        &[f.untyped(), x.untyped()],
        &at,
    );
    assert_eq!(linearized.quantities(), [f.id(), x.id()]);
    assert_eq!(linearized.rows(), 1);
    let reduced = &linearized.reduced()[0];
    close(
        reduced.linear.as_deref().unwrap(),
        &[0.0, 1.0],
        1e-12,
        "the linear term",
    );
    close(
        &[reduced.a.get(0, 0)],
        &[-reduced.a.get(0, 1)],
        1e-12,
        "a row along τ - φ",
    );
    // The exact replacement: -log x - (log x - φ)² / 2(1 + σ²) - ½ log 2π(1 + σ²).
    let cases = [(-0.4, 1.3), (0.5, 0.7), (1.1, 2.6)];
    let given = Observations::new()
        .observe(f, Rows(cases.iter().map(|c| c.0).collect()))
        .observe(x, Rows(cases.iter().map(|c| [c.1]).collect()));
    let values = density_of(&writer.snapshot(), &[factor.id()], given);
    let v = 1.0 + sigma * sigma;
    let expected: Vec<f64> = cases
        .iter()
        .map(|&(phi, x)| {
            -x.ln()
                - (x.ln() - phi).powi(2) / (2.0 * v)
                - 0.5 * (2.0 * std::f64::consts::PI * v).ln()
        })
        .collect();
    close(&values, &expected, 1e-10, "the exact replacement");
}

#[test]
fn rotations_and_vectors_leave_no_linear_term() {
    use skoro_spatial::Rotation;
    let target = Target::host();
    let (mut writer, [x0, z0, x1, _]) = chain();
    let at = Values::new()
        .with(x0, points(&[0.1], 1, 1, &target))
        .with(x1, points(&[0.2], 1, 1, &target));
    let (linearized, _) = retire_with(
        &mut writer,
        Addresses::new().under("tick/0"),
        Observations::new().observe(z0, Z0),
        &[x1.untyped()],
        &at,
    );
    assert!(linearized.reduced()[0].linear.is_none(), "on the reals");
    let mut writer = Writer::new();
    let (r0, r1) = writer
        .build(|| {
            let r0 = quantity("r0", Rotation);
            Normal::on(
                Rotation,
                constant(Rotation, [1.0, 0.0, 0.0, 0.0]),
                Scale::sd(0.3),
            )
            .score(r0);
            let r1 = quantity("r1", Rotation);
            Normal::on(Rotation, r0, Scale::sd(0.2)).score(r1);
            (r0, r1)
        })
        .unwrap();
    let n = (0.81f64 + 0.01 + 0.04 + 0.09).sqrt();
    let at = Values::new()
        .with(
            r0,
            points(&[0.9 / n, 0.1 / n, 0.2 / n, 0.3 / n], 1, 4, &target),
        )
        .with(r1, points(&[1.0, 0.0, 0.0, 0.0], 1, 4, &target));
    let (linearized, _) = retire_with(
        &mut writer,
        Addresses::new().at("r0"),
        Observations::new(),
        &[r1.untyped()],
        &at,
    );
    assert!(linearized.reduced()[0].linear.is_none(), "on the rotations");
    assert_eq!(linearized.rows(), 3);
}

#[test]
fn a_batch_of_replacements_has_one_shape() {
    use skoro_core::Vector;
    let target = Target::host();
    // z ~ N(|ℓ - p|², 0.1²): its Jacobian vanishes where ℓ̂ = p̂, which the first row is at.
    let mut writer = Writer::new();
    let (landmark, p, z) = writer
        .build(|| {
            let landmark = quantity("landmark", Vector::<2>);
            let p = quantity("tick/p", Vector::<2>);
            Normal::on(
                Vector::<2>,
                constant(Vector::<2>, [0.0, 0.0]),
                Scale::sd(1.0),
            )
            .score(p);
            let z = quantity("tick/z", Real);
            let d = landmark - p;
            Normal::on(Real, (d * d).sum(), Scale::sd(0.1)).score(z);
            (landmark, p, z)
        })
        .unwrap();
    let at = Values::new()
        .with(landmark, points(&[0.3, 0.4, 1.0, 2.0], 2, 2, &target))
        .with(p, points(&[0.3, 0.4, 0.1, -0.2], 2, 2, &target));
    let retirement = writer
        .retire(
            Addresses::new().under("tick"),
            Observations::new().observe(z, 0.5),
        )
        .unwrap();
    let replacing = Replacing::new(
        &retirement,
        &[landmark.untyped()],
        &Observations::new(),
        &target,
    )
    .unwrap();
    let linearized = replacing.at(&at).unwrap();
    for reduced in linearized.reduced() {
        assert_eq!(reduced.a.rows(), 1, "min(m', D) = 1 in every row");
        assert!(reduced.linear.is_none());
    }
    let singular = &linearized.reduced()[0];
    assert!(
        singular.a.data().iter().all(|a| a.abs() < 1e-12),
        "the first row's is singular"
    );
    // The batch's factor is built with one value per row.
    let factor = retirement.replace(|_| linearized.build()).unwrap().unwrap();
    let values = density_of(
        &writer.snapshot(),
        &[factor.id()],
        Observations::new().observe(landmark, Rows(vec![[0.3, 0.4], [1.0, 2.0]])),
    );
    assert_eq!(values.len(), 2);
}
