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
