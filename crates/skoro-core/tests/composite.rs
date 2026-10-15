#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)] // a test crate
//! Composites, and the registry by which a name finds its code again.

use std::collections::BTreeSet;
use std::sync::Arc;

use skoro_core::primitive::{
    Add, Atan2, Concatenate, Cos, Exp, IntegerAdd, IntegerEq, IntegerLe, IntegerLt, IntegerMul,
    IntegerSub, IsNonnegative, Log, MatMul, Mul, Neg, PositiveAsReal, PositiveExp, PositiveLog,
    PositiveMul, PositiveRecip, Scale, Select, Sin, Slice, Sub, Sum, ToReal, Transpose,
};
use skoro_core::scope::quantity;
use skoro_core::{Composite, Error, Operation, Param, QDyn, Real, Registry, Space, Writer};

/// Every core operation, with inputs it takes.
fn core_operations() -> Vec<(Arc<dyn Operation>, Vec<Space>)> {
    let v = Space::Vector(2);
    let p = Space::Positive(2);
    let product = Space::product([Space::Real, v.clone()]).unwrap();
    let two = |space: &Space| vec![space.clone(), space.clone()];
    vec![
        (Arc::new(Add), two(&v)),
        (Arc::new(Sub), two(&v)),
        (Arc::new(Neg), vec![v.clone()]),
        (Arc::new(Mul), two(&v)),
        (Arc::new(Scale), vec![Space::Real, v.clone()]),
        (Arc::new(MatMul), vec![Space::Matrix(3, 2), v.clone()]),
        (Arc::new(Sum), vec![v.clone()]),
        (Arc::new(Exp), vec![v.clone()]),
        (Arc::new(Log), vec![v.clone()]),
        (Arc::new(Sin), vec![v.clone()]),
        (Arc::new(Cos), vec![v.clone()]),
        (Arc::new(Atan2), two(&v)),
        (Arc::new(Transpose), vec![Space::Matrix(3, 2)]),
        (
            Arc::new(Slice::new(v.clone(), 1..3, 1..3)),
            vec![product.clone()],
        ),
        (
            Arc::new(Concatenate::new(product)),
            vec![Space::Real, v.clone()],
        ),
        (
            Arc::new(Select),
            vec![Space::Categorical(2), v.clone(), v.clone()],
        ),
        (Arc::new(PositiveAsReal), vec![p.clone()]),
        (Arc::new(PositiveExp), vec![v.clone()]),
        (Arc::new(PositiveLog), vec![p.clone()]),
        (Arc::new(PositiveMul), two(&p)),
        (Arc::new(PositiveRecip), vec![p]),
        (Arc::new(IntegerAdd), two(&Space::Integer)),
        (Arc::new(IntegerSub), two(&Space::Integer)),
        (Arc::new(IntegerMul), two(&Space::Integer)),
        (Arc::new(IntegerEq), two(&Space::Integer)),
        (Arc::new(IntegerLt), two(&Space::Integer)),
        (Arc::new(IntegerLe), two(&Space::Integer)),
        (Arc::new(ToReal), vec![Space::Categorical(3)]),
        (Arc::new(IsNonnegative), vec![Space::Real]),
    ]
}

#[test]
fn every_core_operation_round_trips_through_its_name() {
    let registry = Registry::core();
    let operations = core_operations();
    let names: BTreeSet<&str> = operations.iter().map(|(op, _)| op.name()).collect();
    assert_eq!(
        names,
        registry.names().collect(),
        "the registry holds every core operation"
    );
    for (operation, inputs) in operations {
        let made = registry
            .operation(operation.name(), &operation.params())
            .unwrap();
        assert_eq!(made.name(), operation.name());
        assert_eq!(made.params(), operation.params());
        assert_eq!(
            made.output(&inputs),
            operation.output(&inputs),
            "{}",
            operation.name()
        );
        assert!(made.output(&inputs).is_ok(), "{}", operation.name());
    }
}

#[test]
fn unknown_operation_is_refused_by_name() {
    let mut registry = Registry::core();
    let unknown = registry.operation("skoro.frobnicate", &[]).unwrap_err();
    assert_eq!(
        unknown,
        Error::UnknownOperation {
            name: "skoro.frobnicate".to_string()
        }
    );
    let duplicate = registry
        .register_operation("skoro.add", |_| Ok(Arc::new(Add)))
        .unwrap_err();
    assert_eq!(
        duplicate,
        Error::DuplicateName {
            name: "skoro.add".to_string()
        }
    );
    registry
        .register_operation("feature.double", |_| Ok(Arc::new(Add)))
        .unwrap();
    assert!(registry.operation("feature.double", &[]).is_ok());
    let params = registry
        .operation("skoro.add", &[Param::Integer(1)])
        .unwrap_err();
    assert!(matches!(params, Error::InvalidDeclaration { .. }));
}

#[test]
fn a_call_is_one_node_of_the_composite() {
    let double =
        Composite::define("test.double", Vec::new(), &[Space::Real], |x| x[0] + x[0]).unwrap();
    assert_eq!(double.output(), &Space::Real);
    let mut writer = Writer::new();
    let called = writer
        .build(|| double.call(&[quantity("x", Real).untyped()]))
        .unwrap();
    let snapshot = writer.snapshot();
    let node = snapshot.node(called.id()).unwrap();
    assert_eq!(node.operation().unwrap().name(), "test.double");
    assert_eq!(snapshot.len(), 2, "the quantity and the one call");
    let failure = writer
        .build(|| double.call(&[quantity("v", skoro_core::Vector::<2>).untyped()]))
        .unwrap_err();
    assert!(matches!(
        failure.error,
        Error::SpaceMismatch {
            operation: "test.double",
            ..
        }
    ));
}

#[test]
fn a_body_sees_only_its_inputs() {
    let failure = Composite::define("test.quantity", Vec::new(), &[Space::Real], |x| {
        let q: QDyn = quantity("inside", Real).into();
        x[0] + q
    })
    .unwrap_err();
    assert_eq!(failure.error, Error::NotInAModel);

    let mut writer = Writer::new();
    let failure = writer
        .build(|| {
            let outside: QDyn = quantity("outside", Real).into();
            Composite::define("test.capture", Vec::new(), &[Space::Real], |x| {
                x[0] + outside
            })
        })
        .unwrap()
        .unwrap_err();
    assert!(matches!(failure.error, Error::UnknownNode { .. }));
}

mod programs {
    use std::sync::{Arc, OnceLock};

    use skoro_core::scope::{apply, constant, quantity};
    use skoro_core::{
        Composite, Error, Frames, Lowered, Lowering, Operation, Param, QDyn, Query, Real, Space,
        Values, Vector, Writer,
    };
    use skoro_tensor::{Frame, HostData, HostValue, Target, Value};

    fn value(data: &[f64], target: &Target) -> Value {
        HostValue::new(Some(1), data.len(), HostData::Continuous(data.to_vec()))
            .unwrap()
            .load(1, target)
            .unwrap()
            .unwrap()
    }

    /// Every result of a forward pass, a gradient and a Hessian pass, read back.
    fn results(writer: &Writer, x: QDyn, s: QDyn, f: QDyn, target: &Target) -> Vec<Vec<f64>> {
        let program = writer.snapshot().compile(&Query::of([f]), target).unwrap();
        let at = Values::new()
            .with(x, value(&[0.4, -0.9], target))
            .with(s, value(&[1.3], target));
        let velocities = Frames::new()
            .with(x, Frame::from_host(&[1.0, 0.0, 0.5, 1.0], 1, 2, 2, target))
            .with(s, Frame::from_host(&[0.0, 1.0], 1, 1, 2, target));
        let second = program.hessian_products(&at, &velocities).unwrap();
        let Value::Continuous(y) = program.forward(&at).unwrap().get(f).unwrap().clone() else {
            panic!("a continuous value")
        };
        let mut out = vec![y.to_host()];
        for q in [x, s] {
            out.push(second.alpha.get(q).unwrap().to_host());
            out.push(second.beta.get(q).unwrap().to_host());
        }
        out
    }

    /// `sum(exp(x) ⊙ sin(x)) · s`, written in place.
    fn in_place(x: QDyn, s: QDyn) -> QDyn {
        (x.exp() * x.sin()).sum() * s
    }

    #[test]
    fn expanding_a_composite_changes_no_pass() {
        let target = Target::host();
        let part = Composite::define(
            "test.part",
            Vec::new(),
            &[Space::Vector(2), Space::Real],
            |q| in_place(q[0], q[1]),
        )
        .unwrap();
        let mut called = Writer::new();
        let (x, s, f) = called
            .build(|| {
                let x = quantity("x", Vector::<2>).untyped();
                let s = quantity("s", Real).untyped();
                (x, s, part.call(&[x, s]))
            })
            .unwrap();
        let mut written = Writer::new();
        let (x2, s2, f2) = written
            .build(|| {
                let x = quantity("x", Vector::<2>).untyped();
                let s = quantity("s", Real).untyped();
                (x, s, in_place(x, s))
            })
            .unwrap();
        assert_eq!(
            results(&called, x, s, f, &target),
            results(&written, x2, s2, f2, &target),
            "bit for bit"
        );
    }

    #[test]
    fn composites_nest() {
        let target = Target::host();
        let inner = Composite::define("test.inner", Vec::new(), &[Space::Vector(2)], |q| {
            q[0].exp() * q[0].sin()
        })
        .unwrap();
        let outer = Composite::define(
            "test.outer",
            Vec::new(),
            &[Space::Vector(2), Space::Real],
            |q| inner.call(&[q[0]]).sum() * q[1],
        )
        .unwrap();
        let mut called = Writer::new();
        let (x, s, f) = called
            .build(|| {
                let x = quantity("x", Vector::<2>).untyped();
                let s = quantity("s", Real).untyped();
                (x, s, outer.call(&[x, s]))
            })
            .unwrap();
        let mut written = Writer::new();
        let (x2, s2, f2) = written
            .build(|| {
                let x = quantity("x", Vector::<2>).untyped();
                let s = quantity("s", Real).untyped();
                (x, s, in_place(x, s))
            })
            .unwrap();
        assert_eq!(
            results(&called, x, s, f, &target),
            results(&written, x2, s2, f2, &target)
        );
    }

    #[test]
    fn repeated_composite_off_the_chain_is_not_a_cycle() {
        let target = Target::host();
        let scaled = |c: f64| {
            Composite::define(
                "test.scaled",
                vec![Param::Float(c)],
                &[Space::Vector(2)],
                move |q| q[0] * c,
            )
            .unwrap()
        };
        let (once, twice) = (scaled(2.0), scaled(3.0));
        // A body that calls the composite of the same name at another parameter.
        let nested = Composite::define(
            "test.scaled",
            vec![Param::Float(6.0)],
            &[Space::Vector(2)],
            |q| twice.call(&[once.call(&[q[0]])]),
        )
        .unwrap();
        let mut called = Writer::new();
        let (x, s, f) = called
            .build(|| {
                let x = quantity("x", Vector::<2>).untyped();
                let s = quantity("s", Real).untyped();
                // The same composite, called by two nodes in sequence.
                let y = once.call(&[once.call(&[x])]);
                (x, s, (nested.call(&[y]).exp() * x.sin()).sum() * s)
            })
            .unwrap();
        let mut written = Writer::new();
        let (x2, s2, f2) = written
            .build(|| {
                let x = quantity("x", Vector::<2>).untyped();
                let s = quantity("s", Real).untyped();
                let y = x * 2.0 * 2.0;
                (x, s, ((y * 2.0 * 3.0).exp() * x.sin()).sum() * s)
            })
            .unwrap();
        assert_eq!(
            results(&called, x, s, f, &target),
            results(&written, x2, s2, f2, &target)
        );
    }

    #[test]
    fn a_call_of_constants_folds_when_added() {
        let part = Composite::define(
            "test.part",
            Vec::new(),
            &[Space::Vector(2), Space::Real],
            |q| in_place(q[0], q[1]),
        )
        .unwrap();
        let mut writer = Writer::new();
        let f = writer
            .build(|| {
                let x = constant(Vector::<2>, [0.4, -0.9]).untyped();
                let s = constant(Real, 1.3).untyped();
                part.call(&[x, s])
            })
            .unwrap();
        let snapshot = writer.snapshot();
        let folded = snapshot.node(f.id()).unwrap().value().unwrap().clone();
        let expected: f64 = [0.4f64, -0.9]
            .iter()
            .map(|x| x.exp() * x.sin())
            .sum::<f64>()
            * 1.3;
        let HostData::Continuous(data) = folded.data() else {
            panic!("a continuous value")
        };
        assert!(
            (data[0] - expected).abs() < 1e-12,
            "{data:?} against {expected}"
        );
    }

    /// An operation whose fallback applies itself: a cycle through a fused operation.
    #[derive(Debug)]
    struct Cycle {
        afresh: bool,
    }

    static CYCLE: OnceLock<Arc<Composite>> = OnceLock::new();

    fn fallback(afresh: bool) -> Arc<Composite> {
        Composite::define("test.cycle", Vec::new(), &[Space::Real], move |q| {
            apply(Arc::new(Cycle { afresh }), &[q[0]])
        })
        .unwrap()
    }

    impl Operation for Cycle {
        fn name(&self) -> &'static str {
            "test.cycling"
        }

        fn params(&self) -> Vec<Param> {
            Vec::new()
        }

        fn output(&self, inputs: &[Space]) -> Result<Space, Error> {
            Ok(inputs[0].clone())
        }

        fn lower(&self, _: &[Space], _: &mut Lowering<'_>) -> Result<Lowered, Error> {
            let composite = if self.afresh {
                fallback(true)
            } else {
                CYCLE.get_or_init(|| fallback(false)).clone()
            };
            Ok(Lowered::Composite(composite))
        }
    }

    fn cycle_is_refused(afresh: bool) {
        let target = Target::host();
        let cycle = vec!["test.cycle".to_string(), "test.cycle".to_string()];
        let composite = fallback(afresh);
        assert_eq!(
            composite.compile(&target).unwrap_err(),
            Error::CompositeCycle {
                chain: cycle.clone()
            }
        );

        let mut writer = Writer::new();
        let f = writer
            .build(|| composite.call(&[quantity("x", Real).untyped()]))
            .unwrap();
        let error = writer
            .snapshot()
            .compile(&Query::of([f]), &target)
            .unwrap_err();
        assert_eq!(
            error,
            Error::CompositeCycle {
                chain: cycle.clone()
            }
        );

        let mut line = 0;
        let failure = writer
            .build(|| {
                let c = constant(Real, 1.0).untyped();
                line = line!() + 1;
                composite.call(&[c])
            })
            .unwrap_err();
        assert_eq!(failure.error, Error::CompositeCycle { chain: cycle });
        assert_eq!(failure.location.line(), line);
    }

    #[test]
    fn cycle_through_a_fused_operation_is_refused() {
        cycle_is_refused(false);
    }

    #[test]
    fn cycle_through_a_fused_fallback_defined_afresh_is_refused() {
        cycle_is_refused(true);
    }
}

mod fused {
    use std::sync::Arc;

    use skoro_core::scope::apply;
    use skoro_core::testing::{check_agreement, random};
    use skoro_core::{
        Composite, Error, FirstOrder, Forward, Kernel, Lowered, Lowering, Operation, Param,
        Pullback, QDyn, Query, SecondOrder, SecondPullback, Space, Writer,
    };
    use skoro_tensor::burn::Tensor;
    use skoro_tensor::{Device, Frame, Points, Precision, Target, Value};

    /// `e^{-log x}`, fused into `1 / x` on double-precision targets; on any other, its composite.
    #[derive(Debug)]
    struct Recip;

    fn fallback(n: usize) -> Arc<Composite> {
        Composite::define(
            "test.recip.fallback",
            vec![Param::Integer(n as i64)],
            &[Space::Vector(n)],
            |q| (-q[0].log()).exp(),
        )
        .unwrap()
    }

    struct RecipKernel {
        target: Target,
    }

    struct RecipPullback {
        target: Target,
        x: Tensor<2>,
        velocity: Option<Tensor<3>>,
    }

    fn columns(x: &Tensor<2>, k: usize) -> Tensor<3> {
        let [n, d] = x.dims();
        x.clone().unsqueeze_dim::<3>(2).expand([n, d, k])
    }

    fn input(inputs: &[Value]) -> Tensor<2> {
        let Value::Continuous(x) = &inputs[0] else {
            panic!("a continuous input")
        };
        x.clone().into_burn()
    }

    impl RecipKernel {
        fn value(&self, x: &Tensor<2>) -> Value {
            Value::Continuous(Points::from_burn(x.clone().recip(), &self.target).unwrap())
        }

        fn push(&self, x: &Tensor<2>, t: &Frame) -> Frame {
            let k = t.columns();
            let square = x.clone() * x.clone();
            Frame::from_burn(-(t.clone().into_burn() / columns(&square, k)), &self.target).unwrap()
        }
    }

    impl Forward for RecipKernel {
        fn forward(&self, inputs: &[Value]) -> Value {
            self.value(&input(inputs))
        }
    }

    impl FirstOrder for RecipKernel {
        fn tangent(&self, inputs: &[Value], tangents: &[Option<Frame>]) -> (Value, Option<Frame>) {
            let x = input(inputs);
            (
                self.value(&x),
                tangents[0].as_ref().map(|t| self.push(&x, t)),
            )
        }

        fn record(&self, inputs: &[Value], _: &[bool]) -> (Value, Box<dyn Pullback>) {
            let x = input(inputs);
            let target = self.target.clone();
            (
                self.value(&x),
                Box::new(RecipPullback {
                    target,
                    x,
                    velocity: None,
                }),
            )
        }
    }

    impl SecondOrder for RecipKernel {
        fn record_second(
            &self,
            inputs: &[Value],
            _: &[bool],
            velocities: &[Option<Frame>],
        ) -> (Value, Option<Frame>, Box<dyn SecondPullback>) {
            let x = input(inputs);
            let velocity = velocities[0].as_ref().map(|v| self.push(&x, v));
            let pullback = RecipPullback {
                target: self.target.clone(),
                x: x.clone(),
                velocity: velocities[0].clone().map(Frame::into_burn),
            };
            (self.value(&x), velocity, Box::new(pullback))
        }
    }

    impl Pullback for RecipPullback {
        fn cotangent(&self, alpha: &Frame) -> Vec<Option<Frame>> {
            let k = alpha.columns();
            let square = self.x.clone() * self.x.clone();
            let pulled = -(alpha.clone().into_burn() / columns(&square, k));
            vec![Some(Frame::from_burn(pulled, &self.target).unwrap())]
        }
    }

    impl SecondPullback for RecipPullback {
        /// `d/dt (-α / x²) = 2 v α / x³`.
        fn second(&self, alpha: &Frame) -> Vec<Option<Frame>> {
            let second = self.velocity.as_ref().map(|v| {
                let [n, d, k] = v.dims();
                let cube = self.x.clone() * self.x.clone() * self.x.clone();
                let alpha = alpha.clone().into_burn().expand([n, d, k]);
                let hvp = (v.clone() * alpha).mul_scalar(2.0) / columns(&cube, k);
                Frame::from_burn(hvp, &self.target).unwrap()
            });
            vec![second]
        }
    }

    impl Operation for Recip {
        fn name(&self) -> &'static str {
            "test.recip"
        }

        fn params(&self) -> Vec<Param> {
            Vec::new()
        }

        fn output(&self, inputs: &[Space]) -> Result<Space, Error> {
            match inputs {
                [space @ Space::Vector(_)] => Ok(space.clone()),
                _ => Err(Error::SpaceMismatch {
                    operation: "test.recip",
                    inputs: inputs.to_vec(),
                }),
            }
        }

        fn lower(&self, inputs: &[Space], lowering: &mut Lowering<'_>) -> Result<Lowered, Error> {
            let target = lowering.target().clone();
            if target.precision() == Precision::Double {
                return Ok(Lowered::Kernel(Kernel::Two(Arc::new(RecipKernel {
                    target,
                }))));
            }
            Ok(Lowered::Composite(fallback(inputs[0].storage())))
        }
    }

    fn inputs() -> [(Space, skoro_tensor::HostValue); 1] {
        let space = Space::Vector(3);
        [(space.clone(), random(&space, 3, 90, 0.5, 2.0))]
    }

    fn fused(q: &[QDyn]) -> QDyn {
        apply(Arc::new(Recip), q)
    }

    #[test]
    fn fused_operation_agrees_with_its_composite() {
        let target = Target::new(Device::flex(), Precision::Double).unwrap();
        let composite = fallback(3);
        check_agreement(fused, |q| composite.call(q), &inputs(), &target).unwrap();
        // And a composite it does not fuse is told apart.
        let wrong = Composite::define("test.wrong", Vec::new(), &[Space::Vector(3)], |q| {
            q[0].log().exp()
        })
        .unwrap();
        assert!(check_agreement(fused, |q| wrong.call(q), &inputs(), &target).is_err());
    }

    #[test]
    fn fused_operation_falls_back_to_its_composite() {
        let mut writer = Writer::new();
        let f = writer
            .build(|| fused(&[skoro_core::scope::quantity_in("x", Space::Vector(3))]))
            .unwrap();
        let compile = |precision| {
            let target = Target::new(Device::flex(), precision).unwrap();
            writer.snapshot().compile(&Query::of([f]), &target).unwrap()
        };
        assert_eq!(compile(Precision::Double).len(), 1, "its own kernel");
        assert_eq!(
            compile(Precision::Single).len(),
            3,
            "log, neg and exp, from its composite"
        );
        let single = Target::new(Device::flex(), Precision::Single).unwrap();
        let composite = fallback(3);
        check_agreement(fused, |q| composite.call(q), &inputs(), &single).unwrap();
    }
}
