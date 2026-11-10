#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)] // a test crate
//! A retirement's replacement, linearized: the Kalman filter and its extensions, and improper
//! replacements of what the retired terms do not determine.

mod common;

use skoro_core::scope::{constant, quantity};
use skoro_core::{Addresses, Observations, Q, Real, Writer};
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
