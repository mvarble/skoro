#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)] // a test crate
//! The probabilistic graph, on the mathematics' running example.

use std::sync::Arc;

use skoro_core::scope::{constant, factor, propose, quantity, score};
use skoro_core::{
    AnyNumber, AtMostOne, Error, ExactlyOne, Key, NodeId, Param, Positive, Proposal, Q, Real, Role,
    Sample, TermHandle, Vector, Writer,
};
use skoro_tensor::{Target, Value};

type V2 = Q<Vector<2>>;

/// The vehicle of the running example: positions `x0` then `x1`, a landmark `m` measured from
/// each as `z0` and `z1`, the IMU reading `zeta` and the scale `sigma` of the platform's motion.
struct Example {
    writer: Writer,
    x0: V2,
    x1: V2,
    m: V2,
    z0: V2,
    zeta: V2,
    sigma: Q<Positive<1>>,
    /// `t1` to `t7`, at the indices of the mathematics less one.
    t: [TermHandle; 7],
}

fn example() -> Example {
    let mut writer = Writer::new();
    let (x0, x1, m, z0, zeta, sigma, t) = writer
        .build(|| {
            let half = constant(Real, 0.5);
            let x0 = quantity("tick/0/x", Vector::<2>);
            let t1 = score(x0, -(x0 * x0).sum() * half);
            let m = quantity("landmark", Vector::<2>);
            let t2 = score(m, -(m * m).sum() * half);
            let zeta = quantity("tick/1/imu", Vector::<2>);
            let sigma = quantity("sigma", Positive::<1>);
            let x1 = quantity("tick/1/x", Vector::<2>);
            let step = x1 - x0 - zeta;
            let t3 = score(x1, -(step * step).sum());
            let z0 = quantity("tick/0/z", Vector::<2>);
            let r0 = z0 - (m - x0);
            let t4 = score(z0, -(r0 * r0).sum());
            let z1 = quantity("tick/1/z", Vector::<2>);
            let r1 = z1 - (m - x1);
            let t5 = score(z1, -(r1 * r1).sum());
            // L(x0, x1, σ) = -½ σ⁻² Σ (x1 - x0)² - 2 log σ.
            let r = x1 - x0;
            let inverse = (sigma * sigma).recip().as_real().sum();
            let t6 = factor(-(r * r).sum() * half * inverse - 2.0 * sigma.log().sum());
            let t7 = factor(-sigma.log().sum());
            (x0, x1, m, z0, zeta, sigma, [t1, t2, t3, t4, t5, t6, t7])
        })
        .unwrap();
    Example {
        writer,
        x0,
        x1,
        m,
        z0,
        zeta,
        sigma,
        t,
    }
}

fn ids(qs: &[NodeId]) -> Vec<NodeId> {
    let mut ids = qs.to_vec();
    ids.sort_unstable();
    ids
}

#[test]
fn term_neighbours_are_read_off_the_compilation_graph() {
    let e = example();
    let snapshot = e.writer.snapshot();
    let t6 = snapshot.term(e.t[5]).unwrap();
    assert_eq!(t6.scores, None);
    assert_eq!(
        t6.involves.to_vec(),
        ids(&[e.x0.id(), e.x1.id(), e.sigma.id()])
    );
    let t3 = snapshot.term(e.t[2]).unwrap();
    assert_eq!(t3.scores, Some(e.x1.id()));
    assert_eq!(
        t3.conditioned_on().collect::<Vec<_>>(),
        ids(&[e.x0.id(), e.zeta.id()])
    );
    // Constants are among the log-densities' roots, and in no term.
    for term in snapshot.terms() {
        for q in term.involves.iter() {
            assert!(snapshot.node(*q).unwrap().is_quantity());
        }
    }
    assert_eq!(snapshot.terms().len(), 7);
    assert_eq!(snapshot.scored_by(e.x1).unwrap().log_density, e.t[2].id());
    let joined: Vec<NodeId> = snapshot
        .terms_of(e.x0)
        .iter()
        .map(|t| t.log_density)
        .collect();
    assert_eq!(joined, [e.t[0].id(), e.t[2].id(), e.t[3].id(), e.t[5].id()]);
}

#[test]
fn quantity_is_scored_at_most_once() {
    let mut e = example();
    let x0 = e.x0;
    let failure = e.writer.build(|| score(x0, -(x0 * x0).sum())).unwrap_err();
    assert_eq!(failure.error, Error::AlreadyScored { quantity: x0.id() });
}

#[test]
fn term_cannot_be_conditioned_on_a_later_quantity() {
    let mut writer = Writer::new();
    let failure = writer
        .build(|| {
            let early = quantity("early", Vector::<2>);
            let late = quantity("late", Vector::<2>);
            score(early, (early * late).sum())
        })
        .unwrap_err();
    assert!(
        matches!(failure.error, Error::ConditionedOnLater { .. }),
        "{:?}",
        failure.error
    );
    // The other way round is the ancestral order, and is admitted.
    assert!(
        writer
            .build(|| {
                let early = quantity("early", Vector::<2>);
                let late = quantity("late", Vector::<2>);
                score(late, (early * late).sum())
            })
            .is_ok()
    );
}

#[test]
fn unscored_quantity_is_a_parameter() {
    let e = example();
    let snapshot = e.writer.snapshot();
    assert!(
        snapshot.is_parameter(e.zeta),
        "nothing scores the IMU reading"
    );
    assert!(
        snapshot.is_parameter(e.sigma),
        "a likelihood factor is not a score"
    );
    assert!(!snapshot.is_parameter(e.x1));
    assert!(
        !snapshot.is_parameter(e.t[0]),
        "a log-density is not a quantity"
    );
}

#[test]
fn creation_order_is_ancestral() {
    let e = example();
    let snapshot = e.writer.snapshot();
    let order = snapshot.quantities();
    let position = |q: NodeId| order.iter().position(|&o| o == q).unwrap();
    for term in snapshot.terms() {
        if let Some(scored) = term.scores {
            for q in term.conditioned_on() {
                assert!(position(q) < position(scored));
            }
        }
    }
    assert_eq!(snapshot.under(&"tick/1".into()).len(), 3);
}

#[test]
fn blanket_of_the_first_tick() {
    let e = example();
    let blanket = e.writer.snapshot().blanket([e.x0, e.z0]);
    assert_eq!(
        blanket,
        ids(&[e.x1.id(), e.m.id(), e.zeta.id(), e.sigma.id()])
    );
}

const MEAN: Role<ExactlyOne> = Role::new("mean");
const SCALE: Role<AtMostOne> = Role::new("scale");
const PARTS: Role<AnyNumber> = Role::new("parts");

#[test]
fn binding_lookup_checks_multiplicity() {
    let mut writer = Writer::new();
    let (t, mean) = writer
        .build(|| {
            let x = quantity("x", Vector::<2>);
            let mean = x * 2.0;
            let term = factor(-(x * x).sum())
                .bind(&MEAN, mean)
                .bind(&SCALE, x)
                .bind(&SCALE, x)
                .bind(&PARTS, x)
                .bind(&PARTS, mean);
            (term, mean)
        })
        .unwrap();
    let snapshot = writer.snapshot();
    assert_eq!(snapshot.bound(t, &MEAN).unwrap(), mean.untyped());
    assert_eq!(snapshot.bound(t, &PARTS).unwrap().len(), 2);
    assert_eq!(
        snapshot.bound(t, &SCALE).unwrap_err(),
        Error::Binding {
            role: "scale",
            count: 2
        }
    );
    let absent: Role<ExactlyOne> = Role::new("absent");
    assert_eq!(
        snapshot.bound(t, &absent).unwrap_err(),
        Error::Binding {
            role: "absent",
            count: 0
        }
    );
}

#[test]
fn binding_must_stay_within_its_term() {
    let mut e = example();
    let (t1, x1) = (e.t[0], e.x1);
    let failure = e.writer.build(|| t1.bind(&MEAN, x1 * 2.0)).unwrap_err();
    assert!(matches!(
        failure.error,
        Error::BindingOutsideTerm { role: "mean", .. }
    ));
}

/// A sampler that never draws: these tests attach it and look at it.
#[derive(Debug)]
struct Never;

impl Sample for Never {
    fn name(&self) -> &'static str {
        "test.never"
    }

    fn params(&self) -> Vec<Param> {
        Vec::new()
    }

    fn draw(&self, _: &[Value], _: &[Key], _: &Target) -> Value {
        unreachable!("these tests draw nothing")
    }
}

#[test]
fn proposals_are_conditioned_on_earlier_quantities() {
    let mut e = example();
    let (sigma, x1) = (e.sigma, e.x1);
    let proposal = |log_density, inputs| Proposal {
        sample: Arc::new(Never),
        inputs,
        log_density,
    };
    e.writer
        .build(|| propose(sigma, proposal(-sigma.log().sum(), Vec::new())))
        .unwrap();
    assert!(e.writer.snapshot().proposal(sigma).is_some());
    // `x1` was made after `sigma`.
    let failure = e
        .writer
        .build(|| propose(sigma, proposal((x1 * 0.0).sum(), vec![x1.untyped()])))
        .unwrap_err();
    assert_eq!(
        failure.error,
        Error::ConditionedOnLater {
            quantity: sigma.id(),
            later: x1.id()
        }
    );
}
