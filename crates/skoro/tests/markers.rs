#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)] // a test crate
//! A crate's own marker: `Complex`, a flat embedded type with a product of its own.

mod common;

use common::complex::{Complex, product};
use skoro::scope::quantity;
use skoro::tensor::{HostData, HostValue, Value};
use skoro::testing::check_operation;
use skoro::{Declares, Error, QDyn, Query, Values, Vector, Writer};

fn host(rows: usize, data: Vec<f64>) -> HostValue {
    let storage = data.len() / rows;
    HostValue::new(Some(rows), storage, HostData::Continuous(data)).unwrap()
}

/// Two rows of `Complex<2>`, real parts then imaginary parts.
fn data(seed: f64) -> Vec<f64> {
    (0..8).map(|i| (seed + 0.9 * i as f64).sin()).collect()
}

fn values(seed: f64) -> HostValue {
    host(2, data(seed))
}

skoro::precision_tests! {
    fn feature_marker_defines_its_product(target) {
        let mut writer = Writer::new();
        let (a, b, c) = writer
            .build(|| {
                let a = quantity("a", Complex::<2>);
                let b = quantity("b", Complex::<2>);
                (a, b, a * b)
            })
            .unwrap();
        let program = writer
            .snapshot()
            .compile(&Query::of([QDyn::from(c)]), &target)
            .unwrap();
        let load = |seed: f64| values(seed).load(2, &target).unwrap().unwrap();
        let out = program
            .forward(&Values::new().with(a, load(0.3)).with(b, load(1.1)))
            .unwrap();
        let Value::Continuous(points) = out.get(c).unwrap() else {
            panic!("a continuous value")
        };
        let (got, x, y) = (points.to_host(), data(0.3), data(1.1));
        for row in 0..2 {
            let at = |v: &[f64], i: usize| v[row * 4 + i];
            for k in 0..2 {
                let (p, q) = (at(&x, k), at(&x, k + 2));
                let (r, s) = (at(&y, k), at(&y, k + 2));
                let tolerance = if target.precision() == skoro::Precision::Single { 1e-6 } else { 1e-15 };
                assert!((at(&got, k) - (p * r - q * s)).abs() < tolerance);
                assert!((at(&got, k + 2) - (p * s + q * r)).abs() < tolerance);
            }
        }
        let space = Complex::<2>.space();
        let inputs = [(space.clone(), values(0.3)), (space, values(1.1))];
        if let Err(error) = check_operation(&*product::<2>().operation(), &inputs, &target) {
            panic!("{error}");
        }
    }
}

#[test]
fn feature_marker_is_not_a_vector() {
    let mut writer = Writer::new();
    let failure = writer
        .build(|| {
            let z = quantity("z", Complex::<2>);
            assert!(z.untyped().narrow::<Vector<4>>().is_failed());
        })
        .unwrap_err();
    assert!(
        matches!(failure.error, Error::SpaceMismatch { .. }),
        "{failure:?}"
    );
    let failure = writer
        .build(|| {
            let z = quantity("z", Complex::<2>);
            let v = quantity("v", Vector::<4>);
            assert!((z.untyped() + v.untyped()).is_failed());
        })
        .unwrap_err();
    assert!(
        matches!(failure.error, Error::SpaceMismatch { .. }),
        "{failure:?}"
    );
}
