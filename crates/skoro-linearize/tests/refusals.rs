#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)] // a test crate
//! What a linearization refuses, each refusal naming what it names.

use skoro_core::scope::{constant, factor, quantity, score};
use skoro_core::{Error as CoreError, Integer, Positive, Query, Real, Vector, Writer};
use skoro_linearize::{Assembly, Error, Linearizer};
use skoro_normal::{Normal, Scale};
use skoro_tensor::Target;

#[test]
fn a_free_discrete_quantity_is_refused() {
    let mut writer = Writer::new();
    let i = writer
        .build(|| {
            let x = quantity("x", Real);
            Normal::on(Real, constant(Real, 0.0), Scale::sd(1.0)).score(x);
            let i = quantity("i", Integer);
            let value = i.to_real();
            score(i, -(value * value) * 0.5);
            i
        })
        .unwrap();
    let error = Linearizer::new(
        &writer.snapshot(),
        &Query::log_density(),
        &Target::host(),
        Assembly::Residual,
    )
    .unwrap_err();
    assert_eq!(
        error,
        Error::Core(CoreError::FreeDiscrete {
            quantities: vec![i.id()]
        })
    );
}

#[test]
fn a_factor_without_a_residual_is_refused() {
    let mut writer = Writer::new();
    let (x, written) = writer
        .build(|| {
            let y = quantity("y", Real);
            Normal::on(Real, constant(Real, 0.0), Scale::sd(1.0)).score(y);
            let x = quantity("x", Vector::<2>);
            Normal::on(
                Vector::<2>,
                constant(Vector::<2>, [0.0, 0.0]),
                Scale::sd(1.0),
            )
            .score(x);
            (x, factor(-(x * x).sum() * 0.5))
        })
        .unwrap();
    let snapshot = writer.snapshot();
    for assembly in [Assembly::Residual, Assembly::Fisher] {
        let error = Linearizer::new(&snapshot, &Query::log_density(), &Target::host(), assembly)
            .unwrap_err();
        assert_eq!(error, Error::NoResidual { term: written.id() });
    }
    // With `x` observed, the factor is reached from no free quantity: a constant of the query,
    // which needs no residual.
    let given = skoro_core::Observations::new().observe(x, [0.3, -0.2]);
    let query = Query::log_density().given(given);
    let linearizer =
        Linearizer::new(&snapshot, &query, &Target::host(), Assembly::Residual).unwrap();
    assert_eq!(linearizer.terms().len(), 1, "the prior of y alone");
}

#[test]
fn a_free_scale_is_refused_by_the_residual_linearization() {
    let mut writer = Writer::new();
    let (sigma, term) = writer
        .build(|| {
            let sigma = quantity("sigma", Positive::<1>);
            Normal::on(
                Positive::<1>,
                constant(Positive::<1>, [1.0]),
                Scale::sd(0.5),
            )
            .score(sigma);
            let x = quantity("x", Vector::<2>);
            let term = Normal::on(
                Vector::<2>,
                constant(Vector::<2>, [0.0, 0.0]),
                Scale::isotropic(sigma),
            )
            .score(x);
            (sigma, term)
        })
        .unwrap();
    let snapshot = writer.snapshot();
    let error = Linearizer::new(
        &snapshot,
        &Query::log_density(),
        &Target::host(),
        Assembly::Residual,
    )
    .unwrap_err();
    assert_eq!(
        error,
        Error::ScaleNotFixed {
            term: term.id(),
            quantity: sigma.id()
        }
    );
    // The Fisher linearization accepts it: the normal binds its information root.
    assert!(
        Linearizer::new(
            &snapshot,
            &Query::log_density(),
            &Target::host(),
            Assembly::Fisher
        )
        .is_ok()
    );
    // With the scale observed, it is fixed.
    let given = skoro_core::Observations::new().observe(sigma, [0.7]);
    assert!(
        Linearizer::new(
            &snapshot,
            &Query::log_density().given(given),
            &Target::host(),
            Assembly::Residual
        )
        .is_ok()
    );
}
