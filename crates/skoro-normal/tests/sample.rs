#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)] // a test crate
//! A quantity sampled from a normal: what `quantity` and `Normal::score` build, in one call.

mod common;

use common::{close, host, tolerance};
use skoro_core::scope::{constant, quantity, sample};
use skoro_core::{
    AtMostOne, Categorical, Error, NodeId, Observations, Positive, Query, Role, Rows, Snapshot,
    Values, Writer,
};
use skoro_normal::{Form, Normal, Scale, roles};
use skoro_tensor::Target;

/// Which of the normal's roles the term binds, and its sampler's name and inputs.
fn shape(snapshot: &Snapshot, term: NodeId) -> (Vec<bool>, &'static str, Vec<NodeId>) {
    let at_most = |role: &Role<AtMostOne>| snapshot.bound(term, role).unwrap().is_some();
    let mut bound = vec![
        snapshot.bound(term, &roles::RESIDUAL).is_ok(),
        snapshot.bound(term, &roles::MEAN).is_ok(),
        at_most(&roles::LOG_VOLUME),
        at_most(&roles::INFORMATION_ROOT),
    ];
    bound.extend(
        [
            Form::Isotropic,
            Form::Diagonal,
            Form::CovarianceFactor,
            Form::Whitening,
        ]
        .map(|form| at_most(roles::scale(form))),
    );
    let sampler = snapshot.term(term).unwrap().sampler.clone().unwrap();
    let inputs = sampler.inputs.iter().map(|q| q.id()).collect();
    (bound, sampler.sample.name(), inputs)
}

/// The log-density of the term `term`, row by row, given `given`.
fn density(snapshot: &Snapshot, term: NodeId, given: Observations, target: &Target) -> Vec<f64> {
    let query = Query::log_density_of([term]).given(given);
    let program = snapshot.compile(&query, target).unwrap();
    host(&program.forward(&Values::new()).unwrap(), NodeId::DENSITY)
}

skoro_core::precision_tests! {
    fn sample_scores_as_score_does(target) {
        let mut writer = Writer::new();
        let (x, y) = writer
            .build(|| {
                let mean = constant(Positive::<2>, [1.0, 2.0]);
                let normal = Normal::on(Positive::<2>, mean, Scale::sd(0.3));
                let x = sample("x", Positive::<2>, &normal);
                let y = quantity("y", Positive::<2>);
                normal.score(y);
                (x, y)
            })
            .unwrap();
        let snapshot = writer.snapshot();
        let (tx, ty) = (snapshot.scored_by(x).unwrap(), snapshot.scored_by(y).unwrap());
        assert_eq!(shape(&snapshot, tx.log_density), shape(&snapshot, ty.log_density));

        let points = vec![[0.5, 2.5], [1.0, 2.0], [3.0, 0.4]];
        let at = |q| Observations::new().observe(q, Rows(points.clone()));
        let got = density(&snapshot, tx.log_density, at(x), &target);
        let want = density(&snapshot, ty.log_density, at(y), &target);
        close(&got, &want, tolerance(&target));
    }
}

#[test]
fn sample_of_a_normal_refuses_at_its_line() {
    let line = line!() + 5;
    let failure = Writer::new()
        .build(|| {
            let mean = constant(Categorical::<3>, 1usize);
            let normal = Normal::on(Categorical::<3>, mean, Scale::sd(1.0));
            sample("k", Categorical::<3>, normal);
        })
        .unwrap_err();
    assert_eq!(
        failure.error,
        Error::NotContinuous {
            space: skoro_core::Space::Categorical(3)
        }
    );
    assert_eq!(failure.location.line(), line);
}
