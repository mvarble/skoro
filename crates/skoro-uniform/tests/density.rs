#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)] // a test crate
//! The uniform's term: the constant `-log ν(𝕏)`, whose every derivative is zero.

mod common;

use std::f64::consts::PI;

use common::{close, continuous, density, discrete, rotations, tolerance};
use skoro_core::scope::{quantity, quantity_in};
use skoro_core::{Boolean, Categorical, Frames, Observations, Query, Space, Values, Writer};
use skoro_spatial::{ROTATION, Rotation};
use skoro_tensor::{Frame, Target};
use skoro_uniform::Uniform;

skoro_core::precision_tests! {
    fn term_is_minus_the_log_mass(target) {
        let rotation = Space::Embedded(ROTATION.clone());
        let pair = Space::product([rotation.clone(), rotation]).unwrap();
        let mut writer = Writer::new();
        let (r, k, b, p, terms) = writer
            .build(|| {
                let r = quantity("r", Rotation);
                let k = quantity("k", Categorical::<5>);
                let b = quantity("b", Boolean);
                let p = quantity_in("p", pair.clone());
                let terms = [
                    Uniform::on(Rotation).score(r).id(),
                    Uniform::on(Categorical::<5>).score(k).id(),
                    Uniform::on(Boolean).score(b).id(),
                    Uniform::in_space(pair.clone()).score(p).id(),
                ];
                (r, k, b, p, terms)
            })
            .unwrap();
        let snapshot = writer.snapshot();
        let rotations = rotations();
        let pairs: Vec<f64> = rotations
            .chunks(4)
            .zip(rotations.chunks(4).rev())
            .flat_map(|(a, b)| [a, b].concat())
            .collect();
        let at = [
            Observations::new().observe_dyn(r.untyped(), continuous(&rotations, 4)),
            Observations::new().observe_dyn(k.untyped(), discrete(&[0, 3, 4])),
            Observations::new().observe_dyn(b.untyped(), discrete(&[0, 1, 1])),
            Observations::new().observe_dyn(p, continuous(&pairs, 8)),
        ];
        let log_mass = (8.0 * PI * PI).ln();
        let expected = [-log_mass, -5f64.ln(), -2f64.ln(), -2.0 * log_mass];
        for ((term, given), expected) in terms.into_iter().zip(at).zip(expected) {
            // A constant of the rows: one value, whatever the three points are.
            let got = density(&snapshot, term, given, &target);
            close(&got, &[expected], tolerance(&target));
        }
        assert_eq!(Uniform::on(Rotation).log_mass(), Ok(log_mass));
        assert_eq!(Uniform::in_space(pair.clone()).space(), &pair);
    }
}

#[test]
fn derivatives_are_zero() {
    let target = Target::host();
    let mut writer = Writer::new();
    let x = writer
        .build(|| {
            let x = quantity("x", Rotation);
            Uniform::on(Rotation).score(x);
            x
        })
        .unwrap();
    let program = writer
        .snapshot()
        .compile(&Query::log_density(), &target)
        .unwrap();
    assert_eq!(program.free().len(), 1);
    let rows = 3;
    let value = continuous(&rotations(), 4)
        .load(rows, &target)
        .unwrap()
        .unwrap();
    let at = Values::new().with(x, value);
    let (values, gradient) = program.gradient(&at).unwrap();
    let density = common::host(&values, program.targets()[0]);
    close(&density, &[-(8.0 * PI * PI).ln(); 3], 1e-14);
    let gradient = gradient.get(x).expect("a frame at the free rotation");
    assert_eq!((gradient.rows(), gradient.dimension()), (rows, 3));
    assert!(gradient.to_host().iter().all(|g| *g == 0.0));
    // The Hessian's products with the three directions of each row's tangent space.
    let identity: Vec<f64> = (0..rows * 9)
        .map(|i| if (i % 9) % 4 == 0 { 1.0 } else { 0.0 })
        .collect();
    let velocities = Frames::new().with(x, Frame::from_host(&identity, rows, 3, 3, &target));
    let second = program.hessian_products(&at, &velocities).unwrap();
    for frames in [&second.alpha, &second.beta] {
        let frame = frames.get(x).expect("a frame at the free rotation");
        assert!(frame.to_host().iter().all(|h| *h == 0.0));
    }
}
