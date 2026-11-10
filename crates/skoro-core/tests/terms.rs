#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)] // a test crate
//! The probabilistic graph, on the mathematics' running example.

use std::sync::Arc;

use skoro_core::scope::{
    constant, factor, namespace, propose, quantity, quantity_in, sample, sample_in, score,
};
use skoro_core::{
    AnyNumber, AtMostOne, Declares, Distribution, Error, ExactlyOne, Key, NodeId, Param, Positive,
    Proposal, Q, QDyn, Real, Role, Sample, Space, TermHandle, Typed, Vector, Writer,
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

/// A curved type in name, declared with no volume: storage and dimension 2, moved by addition.
fn unmeasured() -> Arc<skoro_core::Embedded> {
    use skoro_core::{Composite, Embedded, Space};
    let pair = [Space::Vector(2), Space::Vector(2)];
    let r = Composite::define("test.unmeasured.retract", Vec::new(), &pair, |q| {
        q[0] + q[1]
    });
    let q = Composite::define("test.unmeasured.local", Vec::new(), &pair, |q| q[1] - q[0]);
    Embedded::declare(
        "test.unmeasured",
        Vec::new(),
        2,
        2,
        r.unwrap(),
        q.unwrap(),
        None,
    )
    .unwrap()
}

#[derive(Clone, Copy, Debug)]
struct Unmeasured;

impl skoro_core::Typed for Unmeasured {
    fn admits(space: &skoro_core::Space) -> bool {
        matches!(space, skoro_core::Space::Embedded(e) if **e == *unmeasured())
    }
}

impl skoro_core::Declares for Unmeasured {
    fn space(&self) -> skoro_core::Space {
        skoro_core::Space::Embedded(unmeasured())
    }
}

fn no_volume(quantity: NodeId) -> Error {
    Error::NoVolume {
        space: skoro_core::Space::Embedded(unmeasured()),
        quantity: Some(quantity),
    }
}

#[test]
fn score_refuses_a_quantity_without_a_measure() {
    let mut writer = Writer::new();
    let x = writer.build(|| quantity("x", Unmeasured)).unwrap();
    let failure = writer
        .build(|| score(x, -(x.include() * x.include()).sum()))
        .unwrap_err();
    assert_eq!(failure.error, no_volume(x.id()));
    // A product with such a factor has no measure either.
    let space = skoro_core::Space::product([skoro_core::Space::Real, Unmeasured.space()]).unwrap();
    let failure = writer
        .build(|| {
            let p = skoro_core::scope::quantity_in("p", space.clone());
            score(
                p.narrow::<skoro_core::ProductDyn>(),
                (p.factor(0) * 0.0).narrow(),
            )
        })
        .unwrap_err();
    assert!(
        matches!(&failure.error, Error::NoVolume { space: s, quantity: Some(_) } if *s == space),
        "{:?}",
        failure.error
    );
}

#[test]
fn factor_over_a_quantity_without_a_measure_is_allowed() {
    use skoro_core::{Observations, Query, Values};
    use skoro_tensor::{HostData, HostValue};
    let mut writer = Writer::new();
    let (x, y) = writer
        .build(|| {
            let x = quantity("x", Unmeasured);
            let s = x.include().sum().narrow::<Real>();
            factor(-(s * s));
            let y = quantity("y", Real);
            let e = y - s;
            score(y, -(e * e));
            (x, y)
        })
        .unwrap();
    let snapshot = writer.snapshot();
    let target = Target::host();
    let point = || HostValue::new(None, 2, HostData::Continuous(vec![0.5, -0.25])).unwrap();
    let load = |value: HostValue| value.load(1, &target).unwrap().unwrap();
    // Observed.
    let given = Observations::new().observe_dyn(x.untyped(), point());
    let observed = snapshot
        .compile(&Query::log_density().given(given), &target)
        .unwrap();
    let at = Values::new().with(
        y,
        load(HostValue::new(None, 1, HostData::Continuous(vec![0.1])).unwrap()),
    );
    observed.forward(&at).unwrap();
    // Free, and differentiated.
    let free = snapshot.compile(&Query::log_density(), &target).unwrap();
    let at = at.with(x, load(point()));
    free.forward(&at).unwrap();
    let frames = skoro_core::Frames::new().with(
        x,
        skoro_tensor::Frame::from_host(&[1.0, 0.0], 1, 2, 1, &target),
    );
    free.tangent(&at, &frames).unwrap();
}

#[test]
fn propose_refuses_a_quantity_without_a_measure() {
    let mut writer = Writer::new();
    let x = writer.build(|| quantity("x", Unmeasured)).unwrap();
    let failure = writer
        .build(|| {
            let log_density = (x.include() * 0.0).sum().narrow::<Real>();
            let proposal = Proposal {
                sample: Arc::new(Never),
                inputs: Vec::new(),
                log_density,
            };
            propose(x, proposal)
        })
        .unwrap_err();
    assert_eq!(failure.error, no_volume(x.id()));
}

#[test]
fn generation_refuses_a_free_quantity_without_a_measure() {
    let mut writer = Writer::new();
    let x = writer
        .build(|| {
            let x = quantity("x", Unmeasured);
            factor(-(x.include() * x.include()).sum().narrow::<Real>());
            x
        })
        .unwrap();
    let generated = writer.snapshot().generate(
        &skoro_core::Observations::new(),
        &Key::from_seed(3),
        2,
        &Target::host(),
    );
    assert!(
        matches!(&generated, Err(Error::CannotGenerate { quantity, .. }) if *quantity == x.id()),
        "{:?}",
        generated.err()
    );
}

/// The marker of a quantity of any space, for a distribution that reads the space from `x`.
#[derive(Clone, Copy, Debug)]
struct Any;

impl Typed for Any {
    fn admits(_: &Space) -> bool {
        true
    }
}

/// `-½|x ⊖ m|²`, a distribution written for the tests, binding its mean.
struct Tether {
    mean: QDyn,
}

impl Distribution for Tether {
    #[track_caller]
    fn score(&self, x: QDyn) -> TermHandle {
        let t = x.local(self.mean);
        score(x.narrow::<Any>(), ((t * t).sum() * -0.5).narrow()).bind(&MEAN, self.mean)
    }
}

/// The term scoring `q`: conditioned on `m` alone, with `m` bound as its mean.
fn assert_tethered(snapshot: &skoro_core::Snapshot, q: NodeId, m: NodeId) {
    let term = snapshot.scored_by(q).unwrap();
    assert_eq!(term.conditioned_on().collect::<Vec<_>>(), [m]);
    assert_eq!(snapshot.bound(term.log_density, &MEAN).unwrap().id(), m);
}

#[test]
fn sample_declares_and_scores() {
    let mut writer = Writer::new();
    let (m, x, y) = writer
        .build(|| {
            let m = quantity("m", Vector::<2>);
            let tether = Tether { mean: m.untyped() };
            namespace("tick", || {
                let x = sample("x", Vector::<2>, &tether);
                let y = sample("y", Vector::<2>, tether);
                (m, x, y)
            })
        })
        .unwrap();
    let snapshot = writer.snapshot();
    for (q, address) in [(x, "tick/x"), (y, "tick/y")] {
        assert_eq!(snapshot.lookup(&address.into()), Some(q.id()));
        assert_eq!(snapshot.space(q), Some(&Vector::<2>.space()));
        assert_tethered(&snapshot, q.id(), m.id());
    }
}

#[test]
fn sample_in_declares_a_quantity_of_its_space() {
    let mut writer = Writer::new();
    let (m, x) = writer
        .build(|| {
            let m = quantity_in("m", Space::Vector(3));
            (m, sample_in("x", Space::Vector(3), Tether { mean: m }))
        })
        .unwrap();
    let snapshot = writer.snapshot();
    assert_eq!(snapshot.lookup(&"x".into()), Some(x.id()));
    assert_eq!(snapshot.space(x), Some(&Space::Vector(3)));
    assert_tethered(&snapshot, x.id(), m.id());
}

#[test]
fn sample_records_a_refusal_at_its_line() {
    let line = line!() + 5;
    let failure = Writer::new()
        .build(|| {
            let m = quantity("m", Unmeasured);
            let tether = Tether { mean: m.untyped() };
            sample("x", Unmeasured, &tether);
        })
        .unwrap_err();
    assert!(
        matches!(failure.error, Error::NoVolume { .. }),
        "{:?}",
        failure.error
    );
    assert_eq!(failure.location.line(), line);
}

#[test]
fn reached_from_is_what_a_term_involves() {
    let mut e = example();
    let snapshot = e.writer.snapshot();
    for term in snapshot.terms() {
        let reached = snapshot.reached_from(term.log_density);
        let mut expected: Vec<NodeId> = term.involves.to_vec();
        if term.scores.is_some_and(|q| !reached.contains(&q)) {
            expected.retain(|&q| Some(q) != term.scores);
        }
        assert_eq!(reached, expected, "{:?}", term.log_density);
    }
    // A mapped node is reached from the quantities it reads, a quantity from itself, and a
    // constant, fixed in the model, from none.
    let (m, x0) = (e.m, e.x0);
    let (between, half) = e
        .writer
        .build(|| ((m - x0).untyped(), constant(Real, 0.5).untyped()))
        .unwrap();
    let snapshot = e.writer.snapshot();
    assert_eq!(snapshot.reached_from(between), ids(&[m.id(), x0.id()]));
    assert_eq!(snapshot.reached_from(e.zeta), [e.zeta.id()]);
    assert!(snapshot.reached_from(half).is_empty());
}
