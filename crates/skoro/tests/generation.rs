#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)] // a test crate
//! Generation: free quantities drawn ancestrally from samplers and proposals, keyed by address, and
//! weighed by the importance weight.

mod common;

use common::normal::{Normal, exact, inexact, log_density, log_normal, proposal};
use skoro::scope::{constant, factor, propose, quantity, score};
use skoro::tensor::Value;
use skoro::testing::flex;
use skoro::{
    Address, Error, Generated, Key, NodeId, Observations, Precision, Q, Query, Real, Replacement,
    Sampler, Snapshot, Target, Values, Writer,
};

fn host(values: &Values, q: impl Into<NodeId>) -> Vec<f64> {
    match values.get(q).unwrap() {
        Value::Continuous(points) => points.to_host(),
        Value::Discrete(_) => panic!("a continuous value"),
    }
}

fn weights(generated: &Generated) -> Vec<f64> {
    generated.log_weight.to_host()
}

/// The keys of the quantity at `address`, one per row.
fn keys(key: &Key, rows: usize, address: &str) -> Vec<Key> {
    let address = Address::from(address);
    key.split(rows).iter().map(|k| k.at(&address)).collect()
}

/// `x0 ~ N(0, 1)` and `x1 ~ N(x0, 0.5²)`, drawn exactly; `z ~ N(x1, 0.3²)`, observed; and a
/// likelihood factor `N(x1; x0, 2²)`.
struct Chain {
    snapshot: Snapshot,
    x0: Q<Real>,
    x1: Q<Real>,
    z: Q<Real>,
    likelihoods: [NodeId; 2],
}

fn chain() -> Chain {
    let mut writer = Writer::new();
    let (x0, x1, z, likelihoods) = writer
        .build(|| {
            let zero = constant(Real, 0.0);
            let x0 = quantity("x0", Real);
            score(x0, log_density(x0, zero, 1.0)).sampler(exact(zero, 1.0));
            let x1 = quantity("x1", Real);
            score(x1, log_density(x1, x0, 0.5)).sampler(exact(x0, 0.5));
            let z = quantity("z", Real);
            let tz = score(z, log_density(z, x1, 0.3));
            let tf = factor(log_density(x1, x0, 2.0));
            (x0, x1, z, [tz.id(), tf.id()])
        })
        .unwrap();
    Chain {
        snapshot: writer.snapshot(),
        x0,
        x1,
        z,
        likelihoods,
    }
}

#[test]
fn generation_refuses_a_quantity_it_cannot_draw() {
    let mut writer = Writer::new();
    let theta = writer
        .build(|| {
            let theta = quantity("theta", Real);
            let x = quantity("x", Real);
            score(x, log_density(x, theta, 1.0)).sampler(exact(theta, 1.0));
            theta
        })
        .unwrap();
    let refused =
        writer
            .snapshot()
            .generate(&Observations::new(), &Key::from_seed(1), 2, &Target::host());
    match refused {
        Err(Error::CannotGenerate { quantity, address }) => {
            assert_eq!(quantity, NodeId::from(theta));
            assert_eq!(address, Address::from("theta"));
        }
        other => panic!("{other:?}"),
    }

    // A scored quantity whose term has no sampler, likewise.
    let mut writer = Writer::new();
    let x = writer
        .build(|| {
            let x = quantity("x", Real);
            score(x, log_density(x, constant(Real, 0.0), 1.0));
            x
        })
        .unwrap();
    let refused =
        writer
            .snapshot()
            .generate(&Observations::new(), &Key::from_seed(1), 2, &Target::host());
    assert!(
        matches!(refused, Err(Error::CannotGenerate { quantity, .. }) if quantity == NodeId::from(x)),
        "{refused:?}"
    );
}

#[test]
fn exact_samplers_cancel_in_the_weight() {
    let chain = chain();
    let (key, rows, target) = (Key::from_seed(7), 5, Target::host());
    let given = Observations::new().observe(chain.z, 0.7);
    let generated = chain
        .snapshot
        .generate(&given, &key, rows, &target)
        .unwrap();
    // The likelihoods alone, at the values drawn.
    let program = chain
        .snapshot
        .compile(
            &Query::log_density_of(chain.likelihoods).given(given),
            &target,
        )
        .unwrap();
    let at = Values::new()
        .with(chain.x0, generated.values.get(chain.x0).unwrap().clone())
        .with(chain.x1, generated.values.get(chain.x1).unwrap().clone());
    let likelihoods = host(&program.forward(&at).unwrap(), NodeId::DENSITY);
    let weight = weights(&generated);
    assert_eq!(
        weight.iter().map(|w| w.to_bits()).collect::<Vec<_>>(),
        likelihoods.iter().map(|l| l.to_bits()).collect::<Vec<_>>(),
        "{weight:?} and {likelihoods:?}"
    );
}

#[test]
fn inexact_sampler_contributes_its_density() {
    let mut writer = Writer::new();
    let x = writer
        .build(|| {
            let x = quantity("x", Real);
            let (zero, half) = (constant(Real, 0.0), constant(Real, 0.5));
            score(x, log_density(x, zero, 1.0)).sampler(inexact(x, half, 2.0));
            x
        })
        .unwrap();
    let (key, rows) = (Key::from_seed(3), 4);
    let generated = writer
        .snapshot()
        .generate(&Observations::new(), &key, rows, &Target::host())
        .unwrap();
    let (draws, weight) = (host(&generated.values, x), weights(&generated));
    let sampler = Normal { sd: 2.0 };
    for (row, key) in keys(&key, rows, "x").iter().enumerate() {
        let draw = sampler.draw_one(key, 0.5);
        assert_eq!(draws[row], draw);
        let expected = log_normal(draw, 0.0, 1.0) - log_normal(draw, 0.5, 2.0);
        assert!(
            (weight[row] - expected).abs() < 1e-12,
            "{} and {expected}",
            weight[row]
        );
    }
}

#[test]
fn parameter_is_drawn_from_its_proposal() {
    let mut writer = Writer::new();
    let (theta, z) = writer
        .build(|| {
            let theta = quantity("theta", Real);
            propose(theta, proposal(theta, constant(Real, 1.0), 0.5));
            let z = quantity("z", Real);
            score(z, log_density(z, theta, 1.0));
            (theta, z)
        })
        .unwrap();
    let (key, rows) = (Key::from_seed(4), 4);
    let given = Observations::new().observe(z, 0.3);
    let generated = writer
        .snapshot()
        .generate(&given, &key, rows, &Target::host())
        .unwrap();
    let (draws, weight) = (host(&generated.values, theta), weights(&generated));
    let sampler = Normal { sd: 0.5 };
    for (row, key) in keys(&key, rows, "theta").iter().enumerate() {
        let draw = sampler.draw_one(key, 1.0);
        assert_eq!(draws[row], draw);
        let expected = log_normal(0.3, draw, 1.0) - log_normal(draw, 1.0, 0.5);
        assert!(
            (weight[row] - expected).abs() < 1e-12,
            "{} and {expected}",
            weight[row]
        );
    }
}

#[test]
fn sampler_must_stay_within_its_term() {
    let mut writer = Writer::new();
    // Its inputs read y, which the term is not conditioned on.
    let failure = writer
        .build(|| {
            let x0 = quantity("x0", Real);
            let y = quantity("y", Real);
            let x1 = quantity("x1", Real);
            score(x1, log_density(x1, x0, 1.0)).sampler(exact(y, 1.0));
        })
        .unwrap_err();
    assert!(
        matches!(failure.error, Error::SamplerOutsideTerm { .. }),
        "{failure:?}"
    );
    // Its log-density reads y, which the term does not involve.
    let failure = writer
        .build(|| {
            let x0 = quantity("x0", Real);
            let y = quantity("y", Real);
            let x1 = quantity("x1", Real);
            let sampler = Sampler {
                log_density: Some(log_density(x1, y, 1.0)),
                ..exact(x0, 1.0)
            };
            score(x1, log_density(x1, x0, 1.0)).sampler(sampler);
        })
        .unwrap_err();
    assert!(
        matches!(failure.error, Error::SamplerOutsideTerm { .. }),
        "{failure:?}"
    );
}

#[test]
fn weights_estimate_the_evidence() {
    let (a, s, r, observed) = (0.4, 1.2, 0.8, 1.5);
    let mut writer = Writer::new();
    let z = writer
        .build(|| {
            let mean = constant(Real, a);
            let x = quantity("x", Real);
            score(x, log_density(x, mean, s)).sampler(exact(mean, s));
            let z = quantity("z", Real);
            score(z, log_density(z, x, r));
            z
        })
        .unwrap();
    let rows = 100_000;
    let generated = writer
        .snapshot()
        .generate(
            &Observations::new().observe(z, observed),
            &Key::from_seed(11),
            rows,
            &Target::host(),
        )
        .unwrap();
    let w: Vec<f64> = weights(&generated).iter().map(|l| l.exp()).collect();
    let mean = w.iter().sum::<f64>() / rows as f64;
    let variance = w.iter().map(|w| (w - mean).powi(2)).sum::<f64>() / (rows - 1) as f64;
    let error = (variance / rows as f64).sqrt();
    let evidence = log_normal(observed, a, (s * s + r * r).sqrt()).exp();
    assert!(
        (mean - evidence).abs() < 4.0 * error,
        "{mean} against {evidence}, standard error {error}"
    );
}

/// `a/x ~ N(0.25, 1)` and `b/y ~ N(-1, 3²)`, made in either order: means exact in either
/// precision, so that a single-precision draw is the double-precision one, cast.
fn independent(x_first: bool) -> (Writer, Q<Real>) {
    let mut writer = Writer::new();
    let x = writer
        .build(|| {
            let make_x = || {
                let mean = constant(Real, 0.25);
                let x = quantity("a/x", Real);
                score(x, log_density(x, mean, 1.0)).sampler(exact(mean, 1.0));
                x
            };
            let make_y = || {
                let mean = constant(Real, -1.0);
                let y = quantity("b/y", Real);
                score(y, log_density(y, mean, 3.0)).sampler(exact(mean, 3.0));
            };
            if x_first {
                let x = make_x();
                make_y();
                x
            } else {
                make_y();
                make_x()
            }
        })
        .unwrap();
    (writer, x)
}

#[test]
fn draw_depends_on_key_and_address_alone() {
    let (key, rows, target) = (Key::from_seed(5), 3, Target::host());
    let draw = |snapshot: &Snapshot, x: Q<Real>| {
        let generated = snapshot
            .generate(&Observations::new(), &key, rows, &target)
            .unwrap();
        host(&generated.values, x)
    };
    let (first, x) = independent(true);
    let reference = draw(&first.snapshot(), x);
    // Built in the other order.
    let (second, x_second) = independent(false);
    assert_ne!(NodeId::from(x), NodeId::from(x_second));
    assert_eq!(draw(&second.snapshot(), x_second), reference);
    // On another branch, which has grown a quantity of its own.
    let mut branch = first.snapshot().fork();
    branch
        .build(|| {
            let mean = constant(Real, 3.0);
            let w = quantity("c/w", Real);
            score(w, log_density(w, mean, 1.0)).sampler(exact(mean, 1.0));
        })
        .unwrap();
    assert_eq!(draw(&branch.snapshot(), x), reference);
    // After a retirement of quantities made before it.
    let mut retiring = Writer::new();
    retiring
        .build(|| {
            let mean = constant(Real, 0.0);
            let u = quantity("old/u", Real);
            score(u, log_density(u, mean, 1.0)).sampler(exact(mean, 1.0));
        })
        .unwrap();
    let x_after = retiring
        .build(|| {
            let mean = constant(Real, 0.25);
            let x = quantity("a/x", Real);
            score(x, log_density(x, mean, 1.0)).sampler(exact(mean, 1.0));
            x
        })
        .unwrap();
    retiring
        .retire("old", Observations::new())
        .unwrap()
        .replace(|_| Replacement {
            log_density: log_density(constant(Real, 0.0), constant(Real, 0.0), 1.0),
            proposals: Vec::new(),
        })
        .unwrap();
    assert_ne!(NodeId::from(x), NodeId::from(x_after));
    assert_eq!(draw(&retiring.snapshot(), x_after), reference);
}

#[test]
fn rows_draw_independently() {
    let chain = chain();
    let key = Key::from_seed(9);
    let given = Observations::new().observe(chain.z, 0.7);
    let draw = |rows: usize| {
        let generated = chain
            .snapshot
            .generate(&given, &key, rows, &Target::host())
            .unwrap();
        host(&generated.values, chain.x0)
    };
    let four = draw(4);
    let sampler = Normal { sd: 1.0 };
    for (row, key) in keys(&key, 4, "x0").iter().enumerate() {
        assert_eq!(four[row], sampler.draw_one(key, 0.0));
    }
    for (i, a) in four.iter().enumerate() {
        for b in &four[i + 1..] {
            assert_ne!(a, b);
        }
    }
    // A row's draw does not depend on how many rows there are.
    assert_eq!(draw(2), four[..2]);
}

#[test]
fn draws_agree_across_precisions() {
    let (snapshot, x) = {
        let (writer, x) = independent(true);
        (writer.snapshot(), x)
    };
    let key = Key::from_seed(13);
    let generate = |precision: Precision| {
        let generated = snapshot
            .generate(&Observations::new(), &key, 6, &flex(precision).unwrap())
            .unwrap();
        (host(&generated.values, x), weights(&generated))
    };
    let (double, double_weight) = generate(Precision::Double);
    let (single, single_weight) = generate(Precision::Single);
    for (s, d) in single.iter().zip(&double) {
        assert_eq!(*s, f64::from(*d as f32), "{s} and {d}");
    }
    assert_eq!(
        single_weight, double_weight,
        "both weights are zero: every sampler is exact"
    );
}
