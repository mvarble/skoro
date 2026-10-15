#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)] // a test crate
//! Queries: the density, observed values folded once, selection, and the program cache.

use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use skoro_core::scope::{apply, constant, factor, quantity, score, select};
use skoro_core::testing::Tolerance;
use skoro_core::{
    Categorical, Error, Forward, Frames, Integer, Kernel, Key, Lowered, Lowering, NodeId,
    Observations, Operation, Param, Positive, Program, Q, QDyn, Query, Real, Sample, Sampler,
    Space, TermHandle, Values, Vector, Writer,
};
use skoro_tensor::{Device, Frame, HostData, HostValue, Precision, Target, Value};

type V2 = Q<Vector<2>>;

struct Example {
    writer: Writer,
    x0: V2,
    x1: V2,
    m: V2,
    z0: V2,
    z1: V2,
    zeta: V2,
    sigma: Q<Positive<1>>,
    t: [TermHandle; 7],
}

/// The running example of the mathematics, as in the tests of the probabilistic graph.
fn example() -> Example {
    let mut writer = Writer::new();
    let (x0, x1, m, z0, z1, zeta, sigma, t) = writer
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
            let r = x1 - x0;
            let inverse = (sigma * sigma).recip().as_real().sum();
            let t6 = factor(-(r * r).sum() * half * inverse - 2.0 * sigma.log().sum());
            let t7 = factor(-sigma.log().sum());
            (x0, x1, m, z0, z1, zeta, sigma, [t1, t2, t3, t4, t5, t6, t7])
        })
        .unwrap();
    Example {
        writer,
        x0,
        x1,
        m,
        z0,
        z1,
        zeta,
        sigma,
        t,
    }
}

fn value(data: &[f64], target: &Target) -> Value {
    HostValue::new(Some(1), data.len(), HostData::Continuous(data.to_vec()))
        .unwrap()
        .load(1, target)
        .unwrap()
        .unwrap()
}

fn host(values: &Values, q: impl Into<NodeId>) -> Vec<f64> {
    match values.get(q).unwrap() {
        Value::Continuous(points) => points.to_host(),
        Value::Discrete(integers) => integers.to_host().into_iter().map(|i| i as f64).collect(),
    }
}

impl Example {
    /// Every quantity's value, `x1` at `x1`.
    fn at(&self, x1: [f64; 2], target: &Target) -> Values {
        Values::new()
            .with(self.x0, value(&[0.3, -0.2], target))
            .with(self.m, value(&[1.0, 2.0], target))
            .with(self.zeta, value(&[0.1, 0.1], target))
            .with(self.sigma, value(&[0.8], target))
            .with(self.x1, value(&x1, target))
            .with(self.z0, value(&[0.6, 2.1], target))
            .with(self.z1, value(&[0.5, 1.7], target))
    }

    fn compile(&self, query: &Query) -> Arc<Program> {
        self.writer
            .snapshot()
            .compile(query, &Target::host())
            .unwrap()
    }
}

fn density(program: &Program, at: &Values) -> f64 {
    host(&program.forward(at).unwrap(), NodeId::DENSITY)[0]
}

#[test]
fn query_density_is_the_sum_of_terms() {
    let e = example();
    let at = e.at([0.4, 0.1], &Target::host());
    let whole = density(&e.compile(&Query::log_density()), &at);
    let sum: f64 =
        e.t.iter()
            .map(|t| {
                host(
                    &e.compile(&Query::of([QDyn::from(*t)]))
                        .forward(&at)
                        .unwrap(),
                    *t,
                )[0]
            })
            .sum();
    assert!((whole - sum).abs() < 1e-12, "{whole} against {sum}");
}

#[test]
fn log_density_of_sums_the_named_terms() {
    let e = example();
    let at = e.at([0.4, 0.1], &Target::host());
    let part = density(&e.compile(&Query::log_density_of([e.t[0], e.t[2]])), &at);
    let each = |t: TermHandle| {
        host(
            &e.compile(&Query::of([QDyn::from(t)])).forward(&at).unwrap(),
            t,
        )[0]
    };
    assert!((part - each(e.t[0]) - each(e.t[2])).abs() < 1e-12);
}

#[test]
fn log_density_of_ignores_their_order() {
    let e = example();
    let a = e.compile(&Query::log_density_of([e.t[0], e.t[2], e.t[5]]));
    let b = e.compile(&Query::log_density_of([e.t[5], e.t[0], e.t[2]]));
    assert!(Arc::ptr_eq(&a, &b), "the same program, from the cache");
}

#[test]
fn blanket_density_is_the_conditional_up_to_a_constant() {
    let e = example();
    let snapshot = e.writer.snapshot();
    let terms: Vec<NodeId> = snapshot
        .terms_of(e.x1)
        .iter()
        .map(|t| t.log_density)
        .collect();
    let (whole, part) = (
        e.compile(&Query::log_density()),
        e.compile(&Query::log_density_of(terms)),
    );
    assert!(
        !part.free().iter().any(|f| f.node == e.z0.id()),
        "it reads nothing of z0"
    );
    let target = Target::host();
    let differences: Vec<f64> = [[0.4, 0.1], [-1.0, 2.5], [3.0, 0.0]]
        .iter()
        .map(|x1| {
            let at = e.at(*x1, &target);
            density(&whole, &at) - density(&part, &at)
        })
        .collect();
    for d in &differences {
        assert!((d - differences[0]).abs() < 1e-10, "{differences:?}");
    }
    let at = e.at([0.4, 0.1], &target);
    let velocity = Frames::new().with(
        e.x1,
        Frame::from_host(&[1.0, 0.0, 0.0, 1.0], 1, 2, 2, &target),
    );
    let (a, b) = (
        whole.hessian_products(&at, &velocity).unwrap(),
        part.hessian_products(&at, &velocity).unwrap(),
    );
    assert_eq!(
        a.alpha.get(e.x1).unwrap().to_host(),
        b.alpha.get(e.x1).unwrap().to_host()
    );
    assert_eq!(
        a.beta.get(e.x1).unwrap().to_host(),
        b.beta.get(e.x1).unwrap().to_host()
    );
}

#[test]
fn log_density_of_refuses_a_non_term() {
    let e = example();
    let snapshot = e.writer.snapshot();
    let error = snapshot
        .compile(&Query::log_density_of([e.x0]), &Target::host())
        .unwrap_err();
    assert_eq!(error, Error::NotATerm { id: e.x0.id() });
    let mapped = snapshot.term(e.t[2]).unwrap();
    let inner = snapshot.node(mapped.log_density).unwrap().inputs()[0];
    let error = snapshot
        .compile(&Query::log_density_of([inner]), &Target::host())
        .unwrap_err();
    assert_eq!(error, Error::NotATerm { id: inner });
}

/// A sampler that never draws: these tests only attach it.
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
fn roots_are_values_not_draws() {
    let mut writer = Writer::new();
    let x = writer
        .build(|| {
            let x = quantity("x", Vector::<2>);
            let sampler = Sampler {
                sample: Arc::new(Never),
                inputs: Vec::new(),
                log_density: None,
            };
            score(x, -(x * x).sum()).sampler(sampler);
            factor((x * 3.0).sum());
            x
        })
        .unwrap();
    let program = writer
        .snapshot()
        .compile(&Query::log_density(), &Target::host())
        .unwrap();
    let at = Values::new().with(x, value(&[0.5, -2.0], &Target::host()));
    let (_, gradient) = program.gradient(&at).unwrap();
    // -2x from the score and 3 from the factor: x appears in both, sampler or not.
    assert_eq!(gradient.get(x).unwrap().to_host(), [2.0, 7.0]);
}

#[test]
fn folded_and_unfolded_programs_agree() {
    let e = example();
    let target = Target::host();
    let at = e.at([0.4, 0.1], &target);
    let observed = Observations::new().observe(e.x0, [0.3, -0.2]);
    let folded = e.compile(&Query::log_density().given(observed));
    let unfolded = e.compile(&Query::log_density());
    assert!(folded.free().iter().all(|f| f.node != e.x0.id()));
    assert!((density(&folded, &at) - density(&unfolded, &at)).abs() < 1e-12);
    let (_, a) = folded.gradient(&at).unwrap();
    let (_, b) = unfolded.gradient(&at).unwrap();
    for q in [e.x1, e.m, e.z0, e.z1, e.zeta] {
        let (a, b) = (a.get(q).unwrap().to_host(), b.get(q).unwrap().to_host());
        for (a, b) in a.iter().zip(&b) {
            assert!((a - b).abs() < 1e-12);
        }
    }
}

/// The identity, counting its kernel's calls, of order zero.
#[derive(Debug)]
struct Counted {
    calls: Arc<AtomicUsize>,
}

struct CountedKernel {
    calls: Arc<AtomicUsize>,
}

impl Forward for CountedKernel {
    fn forward(&self, inputs: &[Value]) -> Value {
        self.calls.fetch_add(1, Ordering::Relaxed);
        inputs[0].clone()
    }
}

impl Operation for Counted {
    fn name(&self) -> &'static str {
        "test.counted"
    }

    fn params(&self) -> Vec<Param> {
        Vec::new()
    }

    fn output(&self, inputs: &[Space]) -> Result<Space, Error> {
        Ok(inputs[0].clone())
    }

    fn lower(&self, _: &[Space], _: &mut Lowering<'_>) -> Result<Lowered, Error> {
        let calls = self.calls.clone();
        Ok(Lowered::Kernel(Kernel::Zero(Arc::new(CountedKernel {
            calls,
        }))))
    }
}

fn counted() -> (Arc<dyn Operation>, Arc<AtomicUsize>) {
    let calls = Arc::new(AtomicUsize::new(0));
    (
        Arc::new(Counted {
            calls: calls.clone(),
        }),
        calls,
    )
}

#[test]
fn observed_values_fold_once_per_program() {
    let (operation, calls) = counted();
    let mut writer = Writer::new();
    let (o, f) = writer
        .build(|| {
            let o = quantity("o", Vector::<2>);
            let x = quantity("x", Vector::<2>);
            let c = apply(operation, &[o.untyped()]);
            (o, (c * x.untyped()).sum())
        })
        .unwrap();
    let observed = Observations::new().observe(o, [1.0, 2.0]);
    let program = writer
        .snapshot()
        .compile(&Query::of([f]).given(observed), &Target::host())
        .unwrap();
    assert_eq!(
        calls.load(Ordering::Relaxed),
        1,
        "once, when the program is compiled"
    );
    let x = program.free()[0].node;
    for _ in 0..2 {
        let at = Values::new().with(x, value(&[3.0, 4.0], &Target::host()));
        assert_eq!(host(&program.forward(&at).unwrap(), f), [11.0]);
    }
    assert_eq!(calls.load(Ordering::Relaxed), 1, "and never again");
}

#[test]
fn free_values_are_not_folded() {
    let (operation, calls) = counted();
    let mut writer = Writer::new();
    let (x, f) = writer
        .build(|| {
            let x = quantity("x", Vector::<2>);
            (x, apply(operation, &[x.untyped()]).sum())
        })
        .unwrap();
    let program = writer
        .snapshot()
        .compile(&Query::of([f]), &Target::host())
        .unwrap();
    assert_eq!(calls.load(Ordering::Relaxed), 0);
    for _ in 0..2 {
        let at = Values::new().with(x, value(&[3.0, 4.0], &Target::host()));
        program.forward(&at).unwrap();
    }
    assert_eq!(calls.load(Ordering::Relaxed), 2, "once per evaluation");
}

/// `select(k, [A(x), B(x)])`, with `A` and `B` counted.
fn selection(
    index: HostValue,
) -> (
    Arc<Program>,
    Q<Vector<2>>,
    Arc<AtomicUsize>,
    Arc<AtomicUsize>,
    QDyn,
) {
    let ((a, a_calls), (b, b_calls)) = (counted(), counted());
    let mut writer = Writer::new();
    let (k, x, s) = writer
        .build(|| {
            let k = quantity("k", Categorical::<2>);
            let x = quantity("x", Vector::<2>);
            let (a, b) = (apply(a, &[x.untyped()]), apply(b, &[x.untyped()]));
            let s = select(k, &[a.narrow::<Vector<2>>(), b.narrow::<Vector<2>>()]);
            (k, x, s.untyped())
        })
        .unwrap();
    let observed = Observations::new().observe_dyn(k.untyped(), index);
    let program = writer
        .snapshot()
        .compile(&Query::of([s]).given(observed), &Target::host())
        .unwrap();
    (program, x, a_calls, b_calls, s)
}

#[test]
fn selection_by_an_observed_index_drops_the_other_branches() {
    let shared = HostValue::new(None, 1, HostData::Discrete(vec![0])).unwrap();
    let (program, x, a, b, s) = selection(shared);
    assert_eq!(program.len(), 1, "the chosen branch alone");
    let at = Values::new().with(x, value(&[3.0, 4.0], &Target::host()));
    assert_eq!(host(&program.forward(&at).unwrap(), s), [3.0, 4.0]);
    assert_eq!(
        (a.load(Ordering::Relaxed), b.load(Ordering::Relaxed)),
        (1, 0)
    );
}

#[test]
fn selection_by_a_per_row_index_stays_a_step() {
    let per_row = HostValue::new(Some(2), 1, HostData::Discrete(vec![0, 1])).unwrap();
    let (program, x, a, b, s) = selection(per_row);
    assert_eq!(program.len(), 3, "both branches and the selection");
    let at = Values::new().with(
        x,
        HostValue::new(Some(2), 2, HostData::Continuous(vec![3.0, 4.0, 5.0, 6.0]))
            .unwrap()
            .load(2, &Target::host())
            .unwrap()
            .unwrap(),
    );
    assert_eq!(
        host(&program.forward(&at).unwrap(), s),
        [3.0, 4.0, 5.0, 6.0]
    );
    assert_eq!(
        (a.load(Ordering::Relaxed), b.load(Ordering::Relaxed)),
        (1, 1)
    );
}

#[test]
fn same_observations_reuse_the_program() {
    let e = example();
    let observed = Observations::new().observe(e.x0, [0.3, -0.2]);
    let query = Query::log_density().given(observed.clone());
    let (a, b) = (
        e.compile(&query),
        e.compile(&Query::log_density().given(observed)),
    );
    assert!(Arc::ptr_eq(&a, &b));
    let separately = Observations::new().observe(e.x0, [0.3, -0.2]);
    let c = e.compile(&Query::log_density().given(separately));
    assert!(
        !Arc::ptr_eq(&a, &c),
        "an equal set built separately is another set"
    );
}

#[test]
fn program_cache_is_bounded() {
    let e = example();
    let sets: Vec<Observations> = (0..33)
        .map(|i| Observations::new().observe(e.x0, [f64::from(i), 0.0]))
        .collect();
    let first = e.compile(&Query::log_density().given(sets[0].clone()));
    for set in &sets[1..] {
        e.compile(&Query::log_density().given(set.clone()));
    }
    assert_eq!(e.writer.snapshot().cached_programs(), 32);
    let again = e.compile(&Query::log_density().given(sets[0].clone()));
    assert!(
        !Arc::ptr_eq(&first, &again),
        "the oldest was dropped and is compiled again"
    );
    let at = e.at([0.4, 0.1], &Target::host());
    assert_eq!(
        density(&first, &at),
        density(&again, &at),
        "and the one still held still runs"
    );
}

#[test]
fn new_snapshot_starts_with_an_empty_cache() {
    let mut e = example();
    e.compile(&Query::log_density());
    assert_eq!(e.writer.snapshot().cached_programs(), 1);
    let fork = e.writer.snapshot().fork();
    assert_eq!(
        fork.snapshot().cached_programs(),
        1,
        "a fork shares its snapshot"
    );
    e.writer.build(|| quantity("more", Real)).unwrap();
    assert_eq!(e.writer.snapshot().cached_programs(), 0);
}

#[test]
fn programs_are_cached_per_query_and_target() {
    let e = example();
    let single = Target::new(Device::flex(), Precision::Single).unwrap();
    let snapshot = e.writer.snapshot();
    let double = snapshot
        .compile(&Query::log_density(), &Target::host())
        .unwrap();
    let float = snapshot.compile(&Query::log_density(), &single).unwrap();
    assert!(!Arc::ptr_eq(&double, &float));
    assert!(Arc::ptr_eq(
        &float,
        &snapshot.compile(&Query::log_density(), &single).unwrap()
    ));
    let other = snapshot
        .compile(&Query::of([QDyn::from(e.t[0])]), &single)
        .unwrap();
    assert!(!Arc::ptr_eq(&float, &other));
}

#[test]
fn one_snapshot_serves_two_precisions() {
    let e = example();
    let snapshot = e.writer.snapshot();
    let single = Target::new(Device::flex(), Precision::Single).unwrap();
    let values = std::thread::scope(|threads| {
        let run = |target: Target| {
            let snapshot = snapshot.clone();
            let e = &e;
            threads.spawn(move || {
                let program = snapshot.compile(&Query::log_density(), &target).unwrap();
                density(&program, &e.at([0.4, 0.1], &target))
            })
        };
        let (a, b) = (run(Target::host()), run(single));
        (a.join().unwrap(), b.join().unwrap())
    });
    let relative = Tolerance::of(Precision::Single).relative;
    assert!(
        (values.0 - values.1).abs() <= relative * values.0.abs().max(1.0),
        "{values:?}"
    );
}

#[test]
fn free_discrete_quantities_are_reported() {
    let mut writer = Writer::new();
    let (k, f) = writer
        .build(|| {
            let k = quantity("k", Integer);
            let x = quantity("x", Real);
            (k, x * k.to_real())
        })
        .unwrap();
    let program = writer
        .snapshot()
        .compile(&Query::of([f.untyped()]), &Target::host())
        .unwrap();
    assert_eq!(program.free_discrete(), [k.id()]);
    assert_eq!(
        program.refuse_free_discrete().unwrap_err(),
        Error::FreeDiscrete {
            quantities: vec![k.id()]
        }
    );
}

#[test]
fn gradient_at_a_free_discrete_value_is_conditional() {
    let mut writer = Writer::new();
    let (k, x, f) = writer
        .build(|| {
            let k = quantity("k", Categorical::<2>);
            let x = quantity("x", Real);
            let branches = [x * x, x * 3.0];
            (k, x, select(k, &branches))
        })
        .unwrap();
    let program = writer
        .snapshot()
        .compile(&Query::of([f.untyped()]), &Target::host())
        .unwrap();
    let target = Target::host();
    let rows = |data: HostData, storage| {
        HostValue::new(Some(2), storage, data)
            .unwrap()
            .load(2, &target)
            .unwrap()
            .unwrap()
    };
    let at = Values::new()
        .with(k, rows(HostData::Discrete(vec![0, 1]), 1))
        .with(x, rows(HostData::Continuous(vec![2.0, 2.0]), 1));
    let (_, gradient) = program.gradient(&at).unwrap();
    // Row by row, the gradient of the branch chosen: 2x, then 3.
    assert_eq!(gradient.get(x).unwrap().to_host(), [4.0, 3.0]);
}
