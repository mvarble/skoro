#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)] // a test crate
//! Admissibility: which order a pass asks of each step, refused before anything runs.

use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use skoro_core::primitive::{IsNonnegative, ToReal};
use skoro_core::scope::{apply, quantity};
use skoro_core::testing::lower;
use skoro_core::{
    Error, FirstOrder, Forward, Frames, Integer, Kernel, Lowered, Lowering, Observations,
    Operation, Order, Param, Pass, Program, Pullback, QDyn, Query, Real, SecondOrder,
    SecondPullback, Space, Values, Vector, Writer,
};
use skoro_tensor::{Frame, HostData, HostValue, Points, Target, Value};

/// `x ⊙ x`, lowered to a kernel of the order its variant names: `Zero`, `One` or `Two`. Nothing
/// else states the order.
#[derive(Debug)]
struct Square {
    variant: usize,
    calls: Arc<AtomicUsize>,
}

struct SquareKernel {
    target: Target,
    calls: Arc<AtomicUsize>,
}

struct SquarePullback {
    target: Target,
    x: Points,
    velocity: Option<Frame>,
}

fn frame(tensor: skoro_tensor::burn::Tensor<3>, target: &Target) -> Frame {
    Frame::from_burn(tensor, target).unwrap()
}

fn columns(x: &Points, k: usize) -> skoro_tensor::burn::Tensor<3> {
    let [n, d] = [x.rows(), x.storage()];
    x.clone()
        .into_burn()
        .unsqueeze_dim::<3>(2)
        .expand([n, d, k])
}

impl SquareKernel {
    fn square(&self, inputs: &[Value]) -> (Points, Value) {
        self.calls.fetch_add(1, Ordering::Relaxed);
        let Value::Continuous(x) = &inputs[0] else {
            panic!("a continuous input")
        };
        let y = x.clone().into_burn() * x.clone().into_burn();
        (
            x.clone(),
            Value::Continuous(Points::from_burn(y, &self.target).unwrap()),
        )
    }
}

impl Forward for SquareKernel {
    fn forward(&self, inputs: &[Value]) -> Value {
        self.square(inputs).1
    }
}

impl FirstOrder for SquareKernel {
    fn tangent(&self, inputs: &[Value], tangents: &[Option<Frame>]) -> (Value, Option<Frame>) {
        let (x, y) = self.square(inputs);
        let t = tangents[0].as_ref().map(|t| {
            let k = t.columns();
            frame(
                (columns(&x, k) * t.clone().into_burn()).mul_scalar(2.0),
                &self.target,
            )
        });
        (y, t)
    }

    fn record(&self, inputs: &[Value], _: &[bool]) -> (Value, Box<dyn Pullback>) {
        let (x, y) = self.square(inputs);
        let target = self.target.clone();
        (
            y,
            Box::new(SquarePullback {
                target,
                x,
                velocity: None,
            }),
        )
    }
}

impl SecondOrder for SquareKernel {
    fn record_second(
        &self,
        inputs: &[Value],
        _: &[bool],
        velocities: &[Option<Frame>],
    ) -> (Value, Option<Frame>, Box<dyn SecondPullback>) {
        let (value, velocity) = self.tangent(inputs, velocities);
        let Value::Continuous(x) = &inputs[0] else {
            panic!("a continuous input")
        };
        let pullback = SquarePullback {
            target: self.target.clone(),
            x: x.clone(),
            velocity: velocities[0].clone(),
        };
        (value, velocity, Box::new(pullback))
    }
}

impl Pullback for SquarePullback {
    fn cotangent(&self, alpha: &Frame) -> Vec<Option<Frame>> {
        let k = alpha.columns();
        let pulled = (columns(&self.x, k) * alpha.clone().into_burn()).mul_scalar(2.0);
        vec![Some(frame(pulled, &self.target))]
    }
}

impl SecondPullback for SquarePullback {
    fn second(&self, alpha: &Frame) -> Vec<Option<Frame>> {
        let second = self.velocity.as_ref().map(|v| {
            let [n, d, k] = [v.rows(), v.dimension(), v.columns()];
            let alpha = alpha.clone().into_burn().expand([n, d, k]);
            frame(
                (v.clone().into_burn() * alpha).mul_scalar(2.0),
                &self.target,
            )
        });
        vec![second]
    }
}

impl Operation for Square {
    fn name(&self) -> &'static str {
        ["test.square0", "test.square1", "test.square2"][self.variant]
    }

    fn params(&self) -> Vec<Param> {
        Vec::new()
    }

    fn output(&self, inputs: &[Space]) -> Result<Space, Error> {
        Ok(inputs[0].clone())
    }

    fn lower(&self, _: &[Space], lowering: &mut Lowering<'_>) -> Result<Lowered, Error> {
        let kernel = SquareKernel {
            target: lowering.target().clone(),
            calls: self.calls.clone(),
        };
        Ok(Lowered::Kernel(match self.variant {
            0 => Kernel::Zero(Arc::new(kernel)),
            1 => Kernel::One(Arc::new(kernel)),
            _ => Kernel::Two(Arc::new(kernel)),
        }))
    }
}

fn square(variant: usize) -> (Arc<dyn Operation>, Arc<AtomicUsize>) {
    let calls = Arc::new(AtomicUsize::new(0));
    let operation = Square {
        variant,
        calls: calls.clone(),
    };
    (Arc::new(operation), calls)
}

fn value(data: &[f64], target: &Target) -> Value {
    HostValue::new(Some(1), data.len(), HostData::Continuous(data.to_vec()))
        .unwrap()
        .load(1, target)
        .unwrap()
        .unwrap()
}

/// `f = sum(S(y) + x ⊙ x)` and `g = sum(x)`, with `S` the square of the given variant.
struct Model {
    writer: Writer,
    x: QDyn,
    y: QDyn,
    f: QDyn,
    g: QDyn,
}

fn model(operation: Arc<dyn Operation>) -> Model {
    let mut writer = Writer::new();
    let (x, y, f, g) = writer
        .build(|| {
            let x = quantity("x", Vector::<2>).untyped();
            let y = quantity("y", Vector::<2>).untyped();
            let f = (apply(operation, &[y]) + x * x).sum();
            (x, y, f, x.sum())
        })
        .unwrap();
    Model { writer, x, y, f, g }
}

impl Model {
    fn program(&self, targets: &[QDyn]) -> Arc<Program> {
        let query = Query::of(targets.iter().copied());
        self.writer
            .snapshot()
            .compile(&query, &Target::host())
            .unwrap()
    }

    fn at(&self) -> Values {
        let target = Target::host();
        Values::new()
            .with(self.x, value(&[0.5, -1.5], &target))
            .with(self.y, value(&[2.0, 0.25], &target))
    }

    fn velocity(&self, q: QDyn) -> Frames {
        Frames::new().with(q, Frame::from_host(&[1.0, -2.0], 1, 2, 1, &Target::host()))
    }
}

#[test]
fn first_order_step_off_the_pass_admits_a_hessian() {
    // Fed only by an observed quantity: the step is folded before any pass.
    let (operation, _) = square(1);
    let m = model(operation);
    let observed = Observations::new().observe_dyn(
        m.y,
        HostValue::new(None, 2, HostData::Continuous(vec![2.0, 0.25])).unwrap(),
    );
    let query = Query::of([m.f]).given(observed);
    let program = m
        .writer
        .snapshot()
        .compile(&query, &Target::host())
        .unwrap();
    let at = Values::new().with(m.x, value(&[0.5, -1.5], &Target::host()));
    let velocities =
        Frames::new().with(m.x, Frame::from_host(&[1.0, 0.0], 1, 2, 1, &Target::host()));
    assert!(program.hessian_products(&at, &velocities).is_ok());

    // Reaching no target the pass seeds: the step feeds `f`, and the pass seeds `g`.
    let program = m.program(&[m.f, m.g]);
    let (seeded, targets) = ([m.x.id(), m.y.id()], [m.g.id()]);
    let pass = Pass::Hessian {
        seeded: &seeded,
        targets: &targets,
    };
    assert!(program.admits(&pass).is_ok());
    let seeds = Frames::new().with(m.g, Frame::from_host(&[1.0], 1, 1, 1, &Target::host()));
    let velocities = m.velocity(m.y);
    assert!(program.hessian(&m.at(), &velocities, &seeds).is_ok());
}

#[test]
fn first_order_step_without_velocity_admits_a_hessian() {
    let (one, _) = square(1);
    let (two, _) = square(2);
    let (first, second) = (model(one), model(two));
    let pass = |m: &Model| {
        m.program(&[m.f])
            .hessian_products(&m.at(), &m.velocity(m.x))
            .unwrap()
    };
    let (a, b) = (pass(&first), pass(&second));
    for (qa, qb) in [(first.x, second.x), (first.y, second.y)] {
        assert_eq!(
            a.beta.get(qa).unwrap().to_host(),
            b.beta.get(qb).unwrap().to_host()
        );
        assert_eq!(
            a.alpha.get(qa).unwrap().to_host(),
            b.alpha.get(qb).unwrap().to_host()
        );
    }
}

#[test]
fn first_order_step_with_velocity_refuses_a_hessian() {
    let (operation, _) = square(1);
    let m = model(operation);
    let error = m
        .program(&[m.f])
        .hessian_products(&m.at(), &m.velocity(m.y))
        .unwrap_err();
    assert!(
        matches!(
            error,
            Error::OrderTooLow {
                operation: "test.square1",
                ..
            }
        ),
        "{error:?}"
    );
}

#[test]
fn zero_order_step_on_the_pass_refuses_a_gradient() {
    let (operation, _) = square(0);
    let m = model(operation);
    let program = m.program(&[m.f]);
    assert!(
        program.forward(&m.at()).is_ok(),
        "the forward pass needs order zero alone"
    );
    let error = program.gradient(&m.at()).unwrap_err();
    assert!(matches!(
        error,
        Error::OrderTooLow {
            operation: "test.square0",
            ..
        }
    ));
}

#[test]
fn refusal_comes_before_any_kernel_runs() {
    let (operation, calls) = square(0);
    let m = model(operation);
    let program = m.program(&[m.f]);
    assert!(program.gradient(&m.at()).is_err());
    assert_eq!(calls.load(Ordering::Relaxed), 0);
    let (operation, calls) = square(1);
    let m = model(operation);
    let program = m.program(&[m.f]);
    assert!(program.hessian_products(&m.at(), &m.velocity(m.y)).is_err());
    assert_eq!(calls.load(Ordering::Relaxed), 0);
}

#[test]
fn order_is_read_off_the_kernel() {
    let target = Target::host();
    for (variant, order) in [(0, Order::Zero), (1, Order::One), (2, Order::Two)] {
        let (operation, _) = square(variant);
        let kernel = lower(operation.as_ref(), &[Space::Vector(2)], &target).unwrap();
        assert_eq!(kernel.order(), order);
    }
}

#[test]
fn embedding_and_discretization_have_order_zero() {
    let target = Target::host();
    for (operation, input) in [
        (&ToReal as &dyn Operation, Space::Integer),
        (&IsNonnegative, Space::Real),
    ] {
        let kernel = lower(operation, &[input], &target).unwrap();
        assert!(kernel.first().is_none(), "{}", operation.name());
    }

    // f = x · to_real(k) + to_real(1[x ≥ 0]): the gradient in x is to_real(k), nothing being
    // pulled back through the discretization.
    let mut writer = Writer::new();
    let (x, k, f) = writer
        .build(|| {
            let x = quantity("x", Real);
            let k = quantity("k", Integer);
            (x, k, x * k.to_real() + x.is_nonnegative().to_real())
        })
        .unwrap();
    let program = writer
        .snapshot()
        .compile(&Query::of([f.into()]), &target)
        .unwrap();
    assert_eq!(program.free_discrete(), [k.id()]);
    let k_value = HostValue::new(Some(1), 1, HostData::Discrete(vec![3])).unwrap();
    let at = Values::new()
        .with(x, value(&[0.7], &target))
        .with(k, k_value.load(1, &target).unwrap().unwrap());
    let (_, gradient) = program.gradient(&at).unwrap();
    assert_eq!(gradient.get(x).unwrap().to_host(), [3.0]);
    assert!(
        gradient.get(k).is_none(),
        "a discrete quantity has no cotangent"
    );
}
