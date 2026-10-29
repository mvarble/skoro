#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)] // a test crate
//! Retirement, on the scalar chain of the mathematics: `x0 ~ N(a, s²)`, `z0 | x0 ~ N(x0, r²)`,
//! `x1 | x0 ~ N(x0, q²)`, `z1 | x1 ~ N(x1, r²)`, retiring the first tick with `z0` observed.

mod common;

use std::collections::HashSet;

use common::normal::{exact, log_density, log_normal, proposal};
use skoro::scope::{constant, propose, quantity, score};
use skoro::tensor::{Frame, HostData, HostValue, Value};
use skoro::{
    Address, Addresses, Error, Frames, Key, NodeId, Observations, Plan, Q, QDyn, Query, Real,
    Replacement, Snapshot, Target, Values, Writer,
};

const A: f64 = 0.3;
const S: f64 = 1.1;
const R: f64 = 0.5;
const QN: f64 = 0.7;
const Z0: f64 = 0.9;
const Z1: f64 = -0.2;

struct Chain {
    writer: Writer,
    x0: Q<Real>,
    z0: Q<Real>,
    x1: Q<Real>,
    z1: Q<Real>,
    /// The terms of `x0`, `z0`, `x1` and `z1`.
    terms: [NodeId; 4],
    /// `x1 - x0`, a mapped value reached from `x0` that no term reads.
    between: QDyn,
    /// `2 x1`, a mapped value reached from `x1` alone.
    kept: QDyn,
    /// The constant `a`.
    a: QDyn,
}

fn chain() -> Chain {
    let mut writer = Writer::new();
    let (x0, z0, x1, z1, terms, between, kept, a) = writer
        .build(|| {
            let a = constant(Real, A);
            let x0 = quantity("tick/0/x", Real);
            let t0 = score(x0, log_density(x0, a, S)).sampler(exact(a, S));
            let z0 = quantity("tick/0/z", Real);
            let tz0 = score(z0, log_density(z0, x0, R)).sampler(exact(x0, R));
            let x1 = quantity("tick/1/x", Real);
            let t1 = score(x1, log_density(x1, x0, QN)).sampler(exact(x0, QN));
            let z1 = quantity("tick/1/z", Real);
            let tz1 = score(z1, log_density(z1, x1, R)).sampler(exact(x1, R));
            let between = x1.untyped() - x0.untyped();
            let kept = x1.untyped() * 2.0;
            let terms = [t0.id(), tz0.id(), t1.id(), tz1.id()];
            (x0, z0, x1, z1, terms, between, kept, a.untyped())
        })
        .unwrap();
    Chain {
        writer,
        x0,
        z0,
        x1,
        z1,
        terms,
        between,
        kept,
        a,
    }
}

/// [](eq:archive/core/math/kalman): the evidence of `ẑ0`, and the predicted mean and variance of
/// `x1`.
fn kalman() -> (f64, f64, f64) {
    let gain = S * S / (S * S + R * R);
    let mean = A + gain * (Z0 - A);
    let variance = (1.0 - gain) * S * S;
    let evidence = log_normal(Z0, A, (S * S + R * R).sqrt());
    (evidence, mean, variance + QN * QN)
}

/// The exact replacement: `log N(ẑ0; a, s² + r²) + log N(x1; μ, P + q²)`.
fn replacement(x1: Q<Real>) -> Replacement {
    let (evidence, mean, variance) = kalman();
    let predicted = log_density(x1, constant(Real, mean), variance.sqrt());
    Replacement {
        log_density: (predicted.untyped() + evidence).narrow::<Real>(),
        proposals: Vec::new(),
    }
}

fn observed_z0(chain: &Chain) -> Observations {
    Observations::new().observe(chain.z0, Z0)
}

fn retire_exactly(chain: &mut Chain) {
    let x1 = chain.x1;
    let observed = observed_z0(chain);
    chain
        .writer
        .retire(Addresses::new().under("tick/0"), observed)
        .unwrap()
        .replace(|_| replacement(x1))
        .unwrap();
}

fn ids(handles: &[QDyn]) -> Vec<NodeId> {
    handles.iter().map(|q| q.id()).collect()
}

/// Every node the snapshot holds, read back from its terms, samplers, proposals and quantities,
/// and from `also`, names only nodes the snapshot holds, through its inputs and through every
/// sampler and proposal.
fn assert_closed(snapshot: &Snapshot, also: impl IntoIterator<Item = NodeId>) {
    let held = |id: NodeId| snapshot.node(id).is_some();
    let mut stack: Vec<NodeId> = snapshot.quantities();
    stack.extend(also.into_iter().filter(|&id| held(id)));
    for term in snapshot.terms() {
        stack.push(term.log_density);
        stack.extend(term.involves.iter().copied());
        stack.extend(term.scores);
        if let Some(sampler) = &term.sampler {
            stack.extend(sampler.inputs.iter().map(|q| q.id()));
            stack.extend(sampler.log_density.map(|q| q.id()));
        }
    }
    for q in snapshot.quantities() {
        if let Some(proposal) = snapshot.proposal(q) {
            stack.extend(proposal.inputs.iter().map(|q| q.id()));
            stack.push(proposal.log_density.id());
        }
    }
    let mut seen = HashSet::new();
    while let Some(id) = stack.pop() {
        if !seen.insert(id) {
            continue;
        }
        let node = snapshot
            .node(id)
            .unwrap_or_else(|| panic!("{id:?} is named but not held"));
        stack.extend(node.inputs().iter().copied());
    }
    for q in snapshot.quantities() {
        for term in snapshot.terms_of(q) {
            assert!(snapshot.term(term.log_density).is_some());
        }
    }
}

#[test]
fn retirement_reads_the_blanket_and_removes_what_it_names() {
    let mut chain = chain();
    let old = chain.writer.snapshot();
    let observed = observed_z0(&chain);
    let retirement = chain
        .writer
        .retire(Addresses::new().under("tick/0"), observed)
        .unwrap();
    let plan: &Plan = retirement.plan();
    let (x0, z0, x1) = (chain.x0.untyped(), chain.z0.untyped(), chain.x1.untyped());
    assert_eq!(ids(&plan.retired), ids(&[x0, z0]));
    assert_eq!(ids(&plan.integrated), ids(&[x0]));
    let terms: Vec<NodeId> = plan.terms.iter().map(|t| t.log_density).collect();
    assert_eq!(terms, chain.terms[..3]);
    let blanket: Vec<(NodeId, Address)> = plan
        .blanket
        .iter()
        .map(|(q, _, address)| (q.id(), address.clone()))
        .collect();
    assert_eq!(blanket, [(x1.id(), Address::from("tick/1/x"))]);
    assert!(plan.unproposed.is_empty());
    retirement.replace(|_| replacement(chain.x1)).unwrap();

    let new = chain.writer.snapshot();
    let removed = [x0.id(), z0.id(), chain.between.id()]
        .into_iter()
        .chain(chain.terms[..3].iter().copied());
    for id in removed {
        assert!(old.node(id).is_some() && new.node(id).is_none(), "{id:?}");
    }
    for id in [
        x1.id(),
        chain.z1.id(),
        chain.kept.id(),
        chain.a.id(),
        chain.terms[3],
    ] {
        assert!(new.node(id).is_some(), "{id:?} stays");
    }
    assert!(new.term(chain.terms[3]).is_some());
    assert_eq!(new.terms().len(), 2, "the term of z1, and the replacement");
    assert!(new.is_parameter(x1), "x1's term was replaced");
    assert_eq!(new.lookup(&Address::from("tick/0/x")), None);
}

#[test]
fn replacement_outside_the_blanket_is_refused_and_changes_nothing() {
    let mut chain = chain();
    let other = chain.writer.build(|| quantity("other", Real)).unwrap();
    let before = chain.writer.snapshot();
    let x0 = chain.x0;
    let attempts: [(Q<Real>, NodeId); 2] = [(x0, x0.id()), (other, other.id())];
    for (read, expected) in attempts {
        let x1 = chain.x1;
        let observed = observed_z0(&chain);
        let failure = chain
            .writer
            .retire(Addresses::new().under("tick/0"), observed)
            .unwrap()
            .replace(|_| Replacement {
                log_density: log_density(x1, read, 1.0),
                proposals: Vec::new(),
            })
            .unwrap_err();
        assert_eq!(
            failure.error,
            Error::ReplacementOutsideBlanket { quantity: expected }
        );
        let after = chain.writer.snapshot();
        assert_eq!(after.len(), before.len());
        assert_eq!(after.terms().len(), before.terms().len());
        assert!(after.node(x0).is_some());
    }
}

#[test]
fn addresses_name_quantities_exactly_and_namespaces_by_prefix() {
    let mut chain = chain();
    let (x0, z0) = (chain.x0.untyped(), chain.z0.untyped());
    let cases: [(Addresses, Vec<QDyn>); 4] = [
        (Addresses::new().at("tick/0/x"), vec![x0]),
        (Addresses::new().under("tick/0"), vec![x0, z0]),
        (Addresses::new().under("tick/0/x"), vec![]),
        (
            Addresses::new().at("tick/0/x").under("tick/0"),
            vec![x0, z0],
        ),
    ];
    for (addresses, expected) in cases {
        let retirement = chain.writer.retire(addresses, Observations::new()).unwrap();
        assert_eq!(ids(&retirement.plan().retired), ids(&expected));
    }
    let union = chain
        .writer
        .retire(
            Addresses::new().at("tick/0/x").under("tick/0"),
            observed_z0(&chain),
        )
        .unwrap()
        .plan()
        .clone();
    let namespace = chain
        .writer
        .retire(Addresses::new().under("tick/0"), observed_z0(&chain))
        .unwrap()
        .plan()
        .clone();
    assert_eq!(ids(&union.retired), ids(&namespace.retired));
    assert_eq!(ids(&union.integrated), ids(&namespace.integrated));
    let terms = |plan: &Plan| -> Vec<NodeId> { plan.terms.iter().map(|t| t.log_density).collect() };
    assert_eq!(terms(&union), terms(&namespace));
    let blanket =
        |plan: &Plan| -> Vec<NodeId> { plan.blanket.iter().map(|(q, ..)| q.id()).collect() };
    assert_eq!(blanket(&union), blanket(&namespace));
    assert_eq!(ids(&union.unproposed), ids(&namespace.unproposed));
}

#[test]
fn unknown_address_or_observed_outside_is_refused_and_changes_nothing() {
    let mut chain = chain();
    let before = chain.writer.snapshot();
    let unknown = chain
        .writer
        .retire(Addresses::new().at("tick/2/x"), Observations::new())
        .err();
    assert_eq!(
        unknown,
        Some(Error::UnknownAddress {
            address: Address::from("tick/2/x")
        })
    );
    assert_eq!(chain.writer.snapshot(), before);
    let outside = chain
        .writer
        .retire(
            Addresses::new().under("tick/0"),
            Observations::new().observe(chain.z1, Z1),
        )
        .err();
    assert_eq!(
        outside,
        Some(Error::ObservedNotRetired {
            quantity: chain.z1.id()
        })
    );
    assert_eq!(chain.writer.snapshot(), before);
}

#[test]
fn empty_retirement_changes_nothing() {
    let mut chain = chain();
    let before = chain.writer.snapshot();
    for addresses in [Addresses::new(), Addresses::new().under("tick/2")] {
        let retirement = chain.writer.retire(addresses, Observations::new()).unwrap();
        let plan = retirement.plan();
        assert!(plan.retired.is_empty() && plan.integrated.is_empty());
        assert!(plan.terms.is_empty() && plan.blanket.is_empty() && plan.unproposed.is_empty());
        retirement
            .replace(|_| panic!("an empty retirement calls no body"))
            .unwrap();
        assert_eq!(chain.writer.snapshot(), before);
    }
}

/// Gauss–Hermite nodes and weights of `n` points, for `∫ e^{-t²} f(t) dt`.
fn gauss_hermite(n: usize) -> (Vec<f64>, Vec<f64>) {
    let (mut nodes, mut weights) = (vec![0.0; n], vec![0.0; n]);
    let quarter = std::f64::consts::PI.powf(-0.25);
    let mut z: f64 = 0.0;
    for i in 0..n.div_ceil(2) {
        z = match i {
            0 => (2.0 * n as f64 + 1.0).sqrt() - 1.85575 * (2.0 * n as f64 + 1.0).powf(-0.16667),
            1 => z - 1.14 * (n as f64).powf(0.426) / z,
            2 => 1.86 * z - 0.86 * nodes[0],
            3 => 1.91 * z - 0.91 * nodes[1],
            _ => 2.0 * z - nodes[i - 2],
        };
        let mut derivative = 0.0;
        for _ in 0..100 {
            let (mut p1, mut p2) = (quarter, 0.0);
            for j in 0..n {
                let p3 = p2;
                p2 = p1;
                let j = j as f64;
                p1 = z * (2.0 / (j + 1.0)).sqrt() * p2 - (j / (j + 1.0)).sqrt() * p3;
            }
            derivative = (2.0 * n as f64).sqrt() * p2;
            let step = p1 / derivative;
            z -= step;
            if step.abs() < 1e-15 {
                break;
            }
        }
        nodes[i] = z;
        nodes[n - 1 - i] = -z;
        weights[i] = 2.0 / (derivative * derivative);
        weights[n - 1 - i] = weights[i];
    }
    (nodes, weights)
}

fn value(data: &[f64], target: &Target) -> Value {
    HostValue::new(Some(data.len()), 1, HostData::Continuous(data.to_vec()))
        .unwrap()
        .load(data.len(), target)
        .unwrap()
        .unwrap()
}

fn host(values: &Values, q: impl Into<NodeId>) -> Vec<f64> {
    match values.get(q).unwrap() {
        Value::Continuous(points) => points.to_host(),
        Value::Discrete(_) => panic!("a continuous value"),
    }
}

#[test]
fn kalman_replacement_matches_quadrature() {
    let mut chain = chain();
    let target = Target::host();
    let old = chain.writer.snapshot();
    retire_exactly(&mut chain);
    let new = chain.writer.snapshot();
    let at_x1 = 0.4;

    // The new model, free in x1, z1 observed.
    let given = Observations::new().observe(chain.z1, Z1);
    let program = new
        .compile(&Query::log_density().given(given), &target)
        .unwrap();
    let at = Values::new().with(chain.x1, value(&[at_x1], &target));
    let (values, gradient) = program.gradient(&at).unwrap();
    let unit = Frames::new().with(chain.x1, Frame::from_host(&[1.0], 1, 1, 1, &target));
    let hessian = program.hessian_products(&at, &unit).unwrap();
    let new_density = host(&values, NodeId::DENSITY)[0];
    let new_gradient = gradient.get(chain.x1).unwrap().to_host()[0];
    let new_hessian = hessian.beta.get(chain.x1).unwrap().to_host()[0];

    // The old model, z0 and z1 observed, free in x0 and x1, x0 at the quadrature's nodes about the
    // Gaussian it is in x0: N(m, v), with 1/v = 1/s² + 1/r² + 1/q².
    let precision = 1.0 / (S * S) + 1.0 / (R * R) + 1.0 / (QN * QN);
    let v = 1.0 / precision;
    let m = v * (A / (S * S) + Z0 / (R * R) + at_x1 / (QN * QN));
    let (nodes, weights) = gauss_hermite(20);
    let scale = (2.0 * v).sqrt();
    let x0s: Vec<f64> = nodes.iter().map(|t| m + scale * t).collect();
    let rows = x0s.len();
    let given = Observations::new()
        .observe(chain.z0, Z0)
        .observe(chain.z1, Z1);
    let program = old
        .compile(&Query::log_density().given(given), &target)
        .unwrap();
    let at = Values::new()
        .with(chain.x0, value(&x0s, &target))
        .with(chain.x1, value(&vec![at_x1; rows], &target));
    let (values, gradient) = program.gradient(&at).unwrap();
    let unit = Frames::new().with(
        chain.x1,
        Frame::from_host(&vec![1.0; rows], rows, 1, 1, &target),
    );
    let hessian = program.hessian_products(&at, &unit).unwrap();
    let densities = host(&values, NodeId::DENSITY);
    let gradients = gradient.get(chain.x1).unwrap().to_host();
    let hessians = hessian.beta.get(chain.x1).unwrap().to_host();

    // log ∫ e^ℓ dx0, and its derivatives in x1: the mean of ∂ℓ under the weights, and the mean of
    // ∂²ℓ plus the variance of ∂ℓ.
    let logs: Vec<f64> = (0..rows)
        .map(|i| weights[i].ln() + nodes[i] * nodes[i] + densities[i])
        .collect();
    let top = logs.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let total: f64 = logs.iter().map(|l| (l - top).exp()).sum();
    let integrated = top + total.ln() + scale.ln();
    let pi: Vec<f64> = logs.iter().map(|l| (l - top).exp() / total).collect();
    let mean = |f: &dyn Fn(usize) -> f64| (0..rows).map(|i| pi[i] * f(i)).sum::<f64>();
    let g = mean(&|i| gradients[i]);
    let h = mean(&|i| hessians[i]) + mean(&|i| gradients[i] * gradients[i]) - g * g;

    for (name, new, old) in [
        ("log-density", new_density, integrated),
        ("gradient", new_gradient, g),
        ("Hessian", new_hessian, h),
    ] {
        assert!(
            (new - old).abs() < 1e-10 * old.abs().max(1.0),
            "{name}: {new} against {old}"
        );
    }
}

#[test]
fn blanket_quantity_becomes_a_parameter_drawn_from_its_proposal() {
    let (key, rows, target) = (Key::from_seed(21), 8, Target::host());
    // Without a proposal, x1 cannot be drawn.
    let mut chain = chain();
    retire_exactly(&mut chain);
    let refused = chain
        .writer
        .snapshot()
        .generate(&Observations::new(), &key, rows, &target);
    assert!(
        matches!(refused, Err(Error::CannotGenerate { quantity, .. }) if quantity == chain.x1.id()),
        "{refused:?}"
    );

    // With the prediction as its proposal, every row's weight is the evidence.
    let mut chain = self::chain();
    let x1 = chain.x1;
    let observed = observed_z0(&chain);
    chain
        .writer
        .retire(Addresses::new().under("tick/0"), observed)
        .unwrap()
        .replace(|_| {
            let (_, mean, variance) = kalman();
            let proposed = proposal(x1, constant(Real, mean), variance.sqrt());
            Replacement {
                proposals: vec![(x1.untyped(), proposed)],
                ..replacement(x1)
            }
        })
        .unwrap();
    let generated = chain
        .writer
        .snapshot()
        .generate(&Observations::new(), &key, rows, &target)
        .unwrap();
    let (evidence, ..) = kalman();
    for weight in generated.log_weight.to_host() {
        assert!((weight - evidence).abs() < 1e-12, "{weight} and {evidence}");
    }
    assert_closed(&chain.writer.snapshot(), []);
}

#[test]
fn retired_handle_fails_on_the_new_model_and_works_on_the_old() {
    let mut chain = chain();
    let old = chain.writer.snapshot();
    retire_exactly(&mut chain);
    let x0 = chain.x0;
    let failure = chain.writer.build(|| x0.untyped() * 2.0).unwrap_err();
    assert_eq!(failure.error, Error::UnknownNode { id: x0.id() });
    // The old snapshot still holds it, and computes with it.
    let program = old
        .compile(&Query::of([chain.between]), &Target::host())
        .unwrap();
    let at = Values::new()
        .with(x0, value(&[1.0], &Target::host()))
        .with(chain.x1, value(&[3.5], &Target::host()));
    assert_eq!(host(&program.forward(&at).unwrap(), chain.between), [2.5]);
}

#[test]
fn proposal_reading_a_retired_quantity_is_removed_and_reported() {
    let (key, rows, target) = (Key::from_seed(23), 3, Target::host());
    let with_theta = || {
        let mut chain = chain();
        let x0 = chain.x0;
        let theta = chain
            .writer
            .build(|| {
                let theta = quantity("theta", Real);
                propose(theta, proposal(theta, x0, 1.0))
            })
            .unwrap();
        (chain, theta)
    };
    let x1_proposal = |x1: Q<Real>| {
        let (_, mean, variance) = kalman();
        (
            x1.untyped(),
            proposal(x1, constant(Real, mean), variance.sqrt()),
        )
    };

    // Named in the plan, and left without a proposal.
    let (mut chain, theta) = with_theta();
    let (x1, observed) = (chain.x1, observed_z0(&chain));
    let retirement = chain
        .writer
        .retire(Addresses::new().under("tick/0"), observed)
        .unwrap();
    assert_eq!(ids(&retirement.plan().unproposed), [theta.id()]);
    retirement
        .replace(|_| Replacement {
            proposals: vec![x1_proposal(x1)],
            ..replacement(x1)
        })
        .unwrap();
    let snapshot = chain.writer.snapshot();
    assert!(snapshot.proposal(theta).is_none());
    let refused = snapshot.generate(&Observations::new(), &key, rows, &target);
    assert!(
        matches!(refused, Err(Error::CannotGenerate { quantity, .. }) if quantity == theta.id()),
        "{refused:?}"
    );
    assert_closed(&snapshot, [theta.id()]);

    // Given a new proposal with the replacement, it can be drawn again.
    let (mut chain, theta) = with_theta();
    let (x1, observed) = (chain.x1, observed_z0(&chain));
    chain
        .writer
        .retire(Addresses::new().under("tick/0"), observed)
        .unwrap()
        .replace(|_| Replacement {
            proposals: vec![
                x1_proposal(x1),
                (theta.untyped(), proposal(theta, constant(Real, 0.0), 1.0)),
            ],
            ..replacement(x1)
        })
        .unwrap();
    let snapshot = chain.writer.snapshot();
    snapshot
        .generate(&Observations::new(), &key, rows, &target)
        .unwrap();
    assert_closed(&snapshot, []);

    // A replacement proposal that reads a retired quantity is refused.
    let (mut chain, theta) = with_theta();
    let (x0, x1, observed) = (chain.x0, chain.x1, observed_z0(&chain));
    let failure = chain
        .writer
        .retire(Addresses::new().under("tick/0"), observed)
        .unwrap()
        .replace(|_| Replacement {
            proposals: vec![(theta.untyped(), proposal(theta, x0, 2.0))],
            ..replacement(x1)
        })
        .unwrap_err();
    assert_eq!(
        failure.error,
        Error::ProposalReadsRetired {
            quantity: theta.id(),
            retired: x0.id()
        }
    );
    // And so is a proposal for a quantity that keeps its sampler.
    let (z1, observed) = (chain.z1, observed_z0(&chain));
    let failure = chain
        .writer
        .retire(Addresses::new().under("tick/0"), observed)
        .unwrap()
        .replace(|_| Replacement {
            proposals: vec![(z1.untyped(), proposal(z1, constant(Real, 0.0), 1.0))],
            ..replacement(x1)
        })
        .unwrap_err();
    assert_eq!(
        failure.error,
        Error::ProposalNotNeeded { quantity: z1.id() }
    );
}

#[test]
fn nothing_left_refers_to_a_removed_node() {
    let mut chain = chain();
    let old = chain.writer.snapshot();
    let every: Vec<NodeId> = [
        chain.x0.id(),
        chain.z0.id(),
        chain.x1.id(),
        chain.z1.id(),
        chain.between.id(),
        chain.kept.id(),
        chain.a.id(),
    ]
    .into_iter()
    .chain(chain.terms)
    .collect();
    assert_closed(&old, every.clone());
    retire_exactly(&mut chain);
    let new = chain.writer.snapshot();
    assert_closed(&new, every.clone());
    // Retiring the next tick as well, the replacement now among the terms it replaces.
    let observed = Observations::new().observe(chain.z1, Z1);
    let retirement = chain
        .writer
        .retire(Addresses::new().under("tick/1"), observed)
        .unwrap();
    assert_eq!(retirement.plan().terms.len(), 2);
    assert!(retirement.plan().blanket.is_empty());
    retirement
        .replace(|_| Replacement {
            log_density: log_density(constant(Real, 0.0), constant(Real, 0.0), 1.0),
            proposals: Vec::new(),
        })
        .unwrap();
    let last = chain.writer.snapshot();
    assert!(last.quantities().is_empty());
    assert_eq!(last.terms().len(), 1);
    assert_closed(&last, every);
}
