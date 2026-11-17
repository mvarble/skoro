#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)] // a test crate
//! The uniform's draws: functions of a run's key and a quantity's address alone.

mod common;

use common::host;
use skoro_core::scope::sample;
use skoro_core::{Address, Key, Observations, Writer};
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
