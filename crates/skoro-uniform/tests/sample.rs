#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)] // a test crate
//! The uniform's draws: functions of a run's key and a quantity's address alone.

mod common;

use common::{close, continuous, density, host};
use skoro_core::scope::{sample, sample_in};
use skoro_core::{Address, Boolean, Categorical, Key, Observations, Space, Writer};
use skoro_normal::{Normal, Scale};
use skoro_spatial::{ROTATION, Rotation};
use skoro_tensor::Target;
use skoro_uniform::Uniform;

#[test]
fn draws_are_functions_of_key_and_address() {
    let mut writer = Writer::new();
    let (a, b) = writer
        .build(|| {
            let a = sample("a", Rotation, Uniform::on(Rotation));
            let b = sample("frames/b", Rotation, Uniform::on(Rotation));
            (a, b)
        })
        .unwrap();
    let snapshot = writer.snapshot();
    let (key, rows, target) = (Key::from_seed(17), 5, Target::host());
    let generate = || {
        snapshot
            .generate(&Observations::new(), &key, rows, &target)
            .unwrap()
    };
    let generated = generate();
    let draw = ROTATION.mass().unwrap().draw();
    let expected = |address: &str| -> Vec<f64> {
        let address = Address::from(address);
        key.split(rows)
            .iter()
            .flat_map(|row| {
                let mut q = [0.0; 4];
                draw(&row.at(&address).uniforms(3), &mut q);
                q
            })
            .collect()
    };
    let (at_a, at_b) = (
        host(&generated.values, a.id()),
        host(&generated.values, b.id()),
    );
    assert_eq!(at_a, expected("a"));
    assert_eq!(at_b, expected("frames/b"));
    assert_ne!(at_a, at_b);
    for q in at_a.chunks(4) {
        let norm = q.iter().map(|v| v * v).sum::<f64>().sqrt();
        assert!((norm - 1.0).abs() < 1e-15, "{q:?}");
    }
    // The same on every run, and another under another key.
    let again = generate();
    assert_eq!(host(&again.values, a.id()), at_a);
    assert_eq!(host(&again.values, b.id()), at_b);
    let other = snapshot
        .generate(&Observations::new(), &Key::from_seed(18), rows, &target)
        .unwrap();
    assert_ne!(host(&other.values, a.id()), at_a);
    // An exact sampler: each term cancels against its draw, and nothing is left to weigh.
    assert_eq!(generated.log_weight.to_host(), vec![0.0; rows]);
}

#[test]
fn categorical_draws_are_floor_n_u() {
    let (key, rows, target) = (Key::from_seed(23), 64, Target::host());
    for n in [2usize, 3, 7, 1000] {
        let mut writer = Writer::new();
        let k = writer
            .build(|| {
                let space = Space::Categorical(n);
                sample_in("k", space.clone(), Uniform::in_space(space))
            })
            .unwrap();
        let generated = writer
            .snapshot()
            .generate(&Observations::new(), &key, rows, &target)
            .unwrap();
        let address = Address::from("k");
        let expected: Vec<f64> = key
            .split(rows)
            .iter()
            .map(|row| (n as f64 * row.at(&address).uniforms(1)[0]).floor())
            .collect();
        let drawn = host(&generated.values, k.id());
        assert_eq!(drawn, expected, "n = {n}");
        assert!(drawn.iter().all(|&c| c >= 0.0 && c < n as f64));
        assert_eq!(generated.log_weight.to_host(), vec![0.0; rows]);
    }
}

#[test]
fn categorical_counts_are_uniform() {
    let (draws, n) = (60_000, 6);
    let mut writer = Writer::new();
    let k = writer
        .build(|| sample("k", Categorical::<6>, Uniform::on(Categorical::<6>)))
        .unwrap();
    let generated = writer
        .snapshot()
        .generate(
            &Observations::new(),
            &Key::from_seed(29),
            draws,
            &Target::host(),
        )
        .unwrap();
    let mut counts = vec![0.0; n];
    for category in host(&generated.values, k.id()) {
        counts[category as usize] += 1.0;
    }
    let expected = draws as f64 / n as f64;
    let chi_square: f64 = counts
        .iter()
        .map(|c| (c - expected).powi(2) / expected)
        .sum();
    // The 0.999 quantile of chi-square at five degrees of freedom.
    assert!(chi_square < 20.5, "{chi_square} from the counts {counts:?}");
}

/// The overview's model, with the compass observed: the uniform terms of the drawn quantities
/// cancel against their exact draws, and the weight is the likelihood of the compass alone.
#[test]
fn generation_weighs_the_likelihood_alone() {
    let mut writer = Writer::new();
    let (heading, compass) = writer
        .build(|| {
            let heading = sample("heading", Rotation, Uniform::on(Rotation));
            sample("outlier", Boolean, Uniform::on(Boolean));
            let noise = Normal::on(Rotation, heading, Scale::sd(0.05));
            let compass = sample("compass", Rotation, noise);
            (heading, compass)
        })
        .unwrap();
    let snapshot = writer.snapshot();
    let (rows, target) = (16, Target::host());
    let reading = [0.5, 0.1, -0.3, 0.7];
    let given = Observations::new().observe(compass, reading);
    let generated = snapshot
        .generate(&given, &Key::from_seed(41), rows, &target)
        .unwrap();
    // The normal's log-density of the reading, at each drawn heading.
    let drawn = host(&generated.values, heading.id());
    let at = given.observe_dyn(heading.untyped(), continuous(&drawn, 4));
    let likelihood = snapshot.scored_by(compass).unwrap().log_density;
    let expected = density(&snapshot, likelihood, at, &target);
    assert_eq!(expected.len(), rows);
    let weights = generated.log_weight.to_host();
    close(&weights, &expected, 1e-12);
    // Neither log 8π² nor log 2 is in it: either would move every weight by more than this.
    let spread = weights.iter().fold(f64::NEG_INFINITY, |m, w| m.max(*w))
        - weights.iter().fold(f64::INFINITY, |m, w| m.min(*w));
    assert!(
        spread > 1.0,
        "the headings drawn give one weight: {weights:?}"
    );
}
