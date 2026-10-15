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
