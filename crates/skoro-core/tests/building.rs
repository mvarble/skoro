#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)] // a test crate
//! Building a model: scopes, failures, publication, namespaces and folding.

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use skoro_core::primitive::{Add, Exp, MatMul, Mul};
use skoro_core::scope::{self, apply, constant, lookup, namespace, quantity};
use skoro_core::testing::lower;
use skoro_core::{
    Address, Error, MatrixDyn, Operation, QDyn, Real, Rows, Space, Vector, VectorDyn, Writer,
};
use skoro_tensor::{HostData, Target, Value};

#[track_caller]
fn add(a: impl Into<QDyn>, b: impl Into<QDyn>) -> QDyn {
    apply(Arc::new(Add), &[a.into(), b.into()])
}

#[test]
fn operation_outside_a_scope_fails() {
    let x = quantity("x", Real);
    assert!(x.is_failed());
    let mut writer = Writer::new();
    let failure = writer.build(|| add(x, x)).unwrap_err();
    assert_eq!(failure.error, Error::OutsideScope);
    assert!(writer.snapshot().is_empty());
}

#[test]
fn failure_poisons_what_is_built_on_it() {
    let mut writer = Writer::new();
    let mut escaped = None;
    let failure = writer
        .build(|| {
            let x = quantity("x", Vector::<2>);
            let y = quantity("y", Vector::<3>);
            let bad = add(x, y);
            let worse = add(bad, bad);
            assert!(bad.is_failed() && worse.is_failed());
            escaped = Some(worse);
        })
        .unwrap_err();
    assert!(matches!(failure.error, Error::SpaceMismatch { .. }));

    // A failed handle that escaped its scope fails the next scope it is used in.
    let escaped = escaped.unwrap();
    let failure = writer.build(|| add(escaped, escaped)).unwrap_err();
    assert_eq!(failure.error, Error::FailedOperand);
}

#[test]
fn scope_reports_the_first_failure_and_where() {
    let mut writer = Writer::new();
    let mut line = 0;
    let failure = writer
        .build(|| {
            let x = quantity("x", Real);
            let v = quantity("v", Vector::<2>);
            line = line!() + 1;
            let first = add(x, v);
            add(first, x);
            quantity("x", Real); // a second failure, not reported
        })
        .unwrap_err();
    assert!(matches!(
        failure.error,
        Error::SpaceMismatch {
            operation: "skoro.add",
            ..
        }
    ));
    assert_eq!(failure.location.line(), line);
    assert_eq!(failure.location.file(), file!());
}

#[test]
fn shape_mismatch_fails_when_the_node_is_added() {
    let mut writer = Writer::new();
    let failure = writer
        .build(|| {
            let a = quantity("a", MatrixDyn(2, 3));
            let b = quantity("b", VectorDyn(2));
            let product = apply(Arc::new(MatMul), &[a.into(), b.into()]);
            // The failure is known at once, before anything is compiled.
            assert!(product.is_failed());
        })
        .unwrap_err();
    let expected = vec![Space::Matrix(2, 3), Space::Vector(2)];
    assert_eq!(
        failure.error,
        Error::ShapeMismatch {
            operation: "skoro.matmul",
            inputs: expected
        }
    );
}

#[test]
fn failed_scope_publishes_nothing() {
    let mut writer = Writer::new();
    writer.build(|| quantity("x", Real)).unwrap();
    let before = writer.snapshot();
    let reader = writer.reader();
    writer
        .build(|| {
            quantity("y", Real);
            quantity("x", Real);
        })
        .unwrap_err();
    assert_eq!(writer.snapshot(), before);
    assert_eq!(reader.latest(), before);
    assert_eq!(before.lookup(&"y".into()), None);
}

#[test]
fn panic_in_a_scope_discards_the_draft() {
    let mut writer = Writer::new();
    writer.build(|| quantity("x", Real)).unwrap();
    let before = writer.snapshot();
    let unwound = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        writer.build(|| {
            namespace("inner", || {
                quantity("y", Real);
                panic!("a panic inside a namespace inside a scope");
            })
        })
    }));
    assert!(unwound.is_err());
    assert_eq!(scope::depth(), 0, "the stack is empty");
    assert_eq!(writer.snapshot(), before);
    let z = writer.build(|| quantity("z", Real)).unwrap();
    assert_eq!(writer.snapshot().address(z.id()), Some(&Address::from("z")));
}

#[test]
fn readers_see_before_or_after() {
    let mut writer = Writer::new();
    writer.build(|| quantity("first", Real)).unwrap();
    let before = writer.snapshot();
    let reader = writer.reader();
    let done = AtomicBool::new(false);
    let seen = std::thread::scope(|threads| {
        let watcher = threads.spawn(|| {
            let mut seen = Vec::new();
            while !done.load(Ordering::Acquire) {
                let latest = reader.latest();
                if !seen.contains(&latest.len()) {
                    seen.push(latest.len());
                }
            }
            seen.push(reader.latest().len());
            seen
        });
        writer
            .build(|| {
                for i in 0..200usize {
                    quantity(i, Real);
                }
            })
            .unwrap();
        done.store(true, Ordering::Release);
        watcher.join().unwrap()
    });
    let after = writer.snapshot();
    assert!(
        seen.iter().all(|&n| n == before.len() || n == after.len()),
        "{seen:?}"
    );
    assert_eq!(seen.last(), Some(&after.len()));
}

#[test]
fn scopes_nest_and_build_into_the_innermost() {
    let (mut outer, mut inner) = (Writer::new(), Writer::new());
    let (x, y, z) = outer
        .build(|| {
            let x = quantity("x", Real);
            let y = inner.build(|| quantity("y", Real)).unwrap();
            let z = quantity("z", Real);
            (x, y, z)
        })
        .unwrap();
    let (outer, inner) = (outer.snapshot(), inner.snapshot());
    assert!(outer.node(x.id()).is_some() && outer.node(z.id()).is_some());
    assert!(outer.node(y.id()).is_none());
    assert!(inner.node(y.id()).is_some() && inner.len() == 1);
}

#[test]
fn namespaces_prefix_addresses() {
    let mut writer = Writer::new();
    let (x, y, z) = writer
        .build(|| {
            let x = namespace("tick/3", || namespace("imu", || quantity("bias", Real)));
            let y = namespace("tick", || quantity(4usize, Real));
            let z = quantity("z", Real);
            (x, y, z)
        })
        .unwrap();
    let snapshot = writer.snapshot();
    assert_eq!(
        snapshot.address(x.id()),
        Some(&Address::from("tick/3/imu/bias"))
    );
    assert_eq!(snapshot.address(y.id()), Some(&Address::from("tick/4")));
    assert_eq!(snapshot.address(z.id()), Some(&Address::from("z")));

    // A namespace opened outside a scope does not apply inside one.
    let mut other = Writer::new();
    let w = namespace("outside", || other.build(|| quantity("w", Real))).unwrap();
    assert_eq!(other.snapshot().address(w.id()), Some(&Address::from("w")));
}

#[test]
fn lookup_is_relative_to_the_open_namespaces() {
    let mut writer = Writer::new();
    let x = writer
        .build(|| namespace("tick/1", || quantity("x", Vector::<2>)))
        .unwrap();
    let failure = writer
        .build(|| namespace("tick/1", || lookup::<Real>("x")))
        .unwrap_err();
    assert!(matches!(
        failure.error,
        Error::SpaceMismatch {
            operation: "lookup",
            ..
        }
    ));

    let (relative, absolute) = writer
        .build(|| {
            let relative = namespace("tick/1", || lookup::<Vector<2>>("x"));
            (relative, lookup::<VectorDyn>("tick/1/x"))
        })
        .unwrap();
    assert_eq!(relative, x);
    assert_eq!(absolute.id(), x.id());
    let missing = writer.build(|| lookup::<Real>("tick/2/x")).unwrap_err();
    assert_eq!(
        missing.error,
        Error::UnknownAddress {
            address: "tick/2/x".into()
        }
    );
}

#[test]
fn address_cannot_be_taken_twice() {
    let mut writer = Writer::new();
    writer.build(|| quantity("a/1", Real)).unwrap();
    let failure = writer.build(|| quantity("a/1", Vector::<2>)).unwrap_err();
    assert_eq!(
        failure.error,
        Error::AddressTaken {
            address: "a/1".into()
        }
    );
    // A name that looks like a number is another address.
    let named = writer
        .build(|| {
            quantity(
                Address::new([
                    skoro_core::Segment::name("a"),
                    skoro_core::Segment::name("1"),
                ]),
                Real,
            )
        })
        .unwrap();
    assert!(!named.is_failed());
}

#[test]
fn constants_fold_when_added() {
    let mut writer = Writer::new();
    let (folded, unfolded) = writer
        .build(|| {
            let a = constant(Vector::<3>, [0.1, -2.0, 3.5]);
            let b = constant(Vector::<3>, [1.0, 0.25, -0.5]);
            let product = apply(Arc::new(Mul), &[a.into(), b.into()]);
            let folded = apply(Arc::new(Exp), &[product]);
            let x = quantity("x", Vector::<3>);
            let unfolded = add(folded, x);
            (folded, unfolded)
        })
        .unwrap();
    let snapshot = writer.snapshot();
    let node = snapshot.node(folded.id()).unwrap();
    assert!(node.is_fixed() && node.operation().unwrap().name() == "skoro.exp");
    let value = node.value().unwrap();
    assert_eq!(value.rows(), None, "a fold of shared constants is shared");
    let HostData::Continuous(data) = value.data() else {
        panic!("a continuous value")
    };

    // The unfolded computation, on the host in double precision.
    let target = Target::host();
    let load = |data: Vec<f64>| {
        skoro_tensor::HostValue::new(None, 3, HostData::Continuous(data))
            .unwrap()
            .load(1, &target)
            .unwrap()
            .unwrap()
    };
    let space = [Space::Vector(3), Space::Vector(3)];
    let mul = lower(&Mul, &space, &target).unwrap();
    let product = mul
        .forward()
        .forward(&[load(vec![0.1, -2.0, 3.5]), load(vec![1.0, 0.25, -0.5])]);
    let exp = lower(&Exp, &space[..1], &target).unwrap();
    let Value::Continuous(expected) = exp.forward().forward(&[product]) else {
        panic!("a continuous value")
    };
    assert_eq!(data, &expected.to_host());
    assert!(!snapshot.node(unfolded.id()).unwrap().is_fixed());
}

#[test]
fn constants_of_rows_fold_row_by_row() {
    let mut writer = Writer::new();
    let (sum, failure) = (
        writer
            .build(|| {
                let a = constant(Real, Rows(vec![1.0, 2.0, 3.0]));
                let b = constant(Real, 10.0);
                add(a, b)
            })
            .unwrap(),
        writer
            .build(|| {
                let a = constant(Real, Rows(vec![1.0, 2.0, 3.0]));
                let b = constant(Real, Rows(vec![1.0, 2.0]));
                add(a, b)
            })
            .unwrap_err(),
    );
    let value = writer
        .snapshot()
        .node(sum.id())
        .unwrap()
        .value()
        .unwrap()
        .clone();
    assert_eq!(value.rows(), Some(3));
    assert_eq!(value.data(), &HostData::Continuous(vec![11.0, 12.0, 13.0]));
    assert_eq!(
        failure.error,
        Error::BatchSize {
            expected: 3,
            found: 2
        }
    );
}

#[test]
fn constants_are_checked_against_their_space() {
    let mut writer = Writer::new();
    let failure = writer
        .build(|| constant(skoro_core::Positive::<2>, [1.0, -1.0]))
        .unwrap_err();
    assert_eq!(
        failure.error,
        Error::ValueShape {
            space: Space::Positive(2)
        }
    );
    let failure = writer
        .build(|| constant(VectorDyn(3), &[1.0, 2.0][..]))
        .unwrap_err();
    assert_eq!(
        failure.error,
        Error::ValueShape {
            space: Space::Vector(3)
        }
    );
    let failure = writer
        .build(|| constant(skoro_core::Categorical::<3>, 3usize))
        .unwrap_err();
    assert_eq!(
        failure.error,
        Error::ValueShape {
            space: Space::Categorical(3)
        }
    );
}

#[test]
fn widening_is_free_and_narrowing_checks_the_space() {
    let mut writer = Writer::new();
    let (x, wide, narrow) = writer
        .build(|| {
            let x = quantity("x", Vector::<3>);
            let wide = x.widen();
            (x, wide, wide.narrow::<Vector<3>>())
        })
        .unwrap();
    assert_eq!(wide.id(), x.id());
    assert_eq!(narrow, x);
    assert_eq!(
        writer.snapshot().len(),
        1,
        "neither widening nor narrowing adds a node"
    );

    let mut line = 0;
    let failure = writer
        .build(|| {
            let x = lookup::<VectorDyn>("x");
            line = line!() + 1;
            let wrong = x.narrow::<Vector<2>>();
            assert!(wrong.is_failed());
            let untyped = x.untyped().narrow::<Real>();
            assert!(untyped.is_failed());
        })
        .unwrap_err();
    assert_eq!(
        failure.error,
        Error::SpaceMismatch {
            operation: "narrow",
            inputs: vec![Space::Vector(3)]
        }
    );
    assert_eq!(failure.location.line(), line);
    let _: &dyn Operation = &Add;
}
