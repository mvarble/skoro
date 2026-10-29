#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)] // a test crate
//! Operators and methods build the nodes of the primitives they name, typed and untyped alike.

use skoro_core::scope::{self, constant, quantity};
use skoro_core::{
    Categorical, Error, Integer, Matrix, MatrixDyn, NodeId, Positive, PositiveDyn, QDyn, Real,
    Snapshot, Space, Vector, VectorDyn, Writer,
};

/// The name of the operation of `q`'s node, and those of its inputs (`constant` and `quantity`
/// for nodes that are not mapped).
fn named(snapshot: &Snapshot, q: impl Into<QDyn>) -> (String, Vec<String>) {
    let node = snapshot
        .node(q.into().id())
        .expect("the node is in the model");
    let name = |id: NodeId| {
        let node = snapshot.node(id).unwrap();
        match (node.operation(), node.is_quantity()) {
            (Some(operation), _) => operation.name().to_string(),
            (None, true) => "quantity".to_string(),
            (None, false) => "constant".to_string(),
        }
    };
    let inputs = node.inputs().iter().map(|&id| name(id)).collect();
    (name(node.id()), inputs)
}

fn op(name: &str, inputs: &[&str]) -> (String, Vec<String>) {
    (
        name.to_string(),
        inputs.iter().map(|s| s.to_string()).collect(),
    )
}

#[test]
fn operators_build_the_nodes_of_their_markers() {
    let mut writer = Writer::new();
    let nodes = writer
        .build(|| {
            let v = quantity("v", Vector::<3>);
            let w = quantity("w", Vector::<3>);
            let m = quantity("m", Matrix::<2, 3>);
            let vd = quantity("vd", VectorDyn(2));
            let md = quantity("md", MatrixDyn(2, 2));
            let s = quantity("s", Real);
            let p = quantity("p", Positive::<2>);
            let q = quantity("q", Positive::<2>);
            let pd = quantity("pd", PositiveDyn(2));
            let i = quantity("i", Integer);
            let j = quantity("j", Integer);
            let k = quantity("k", Categorical::<3>);
            let (vu, wu, su, pu, qu, iu, ju) = (
                v.untyped(),
                w.untyped(),
                s.untyped(),
                p.untyped(),
                q.untyped(),
                i.untyped(),
                j.untyped(),
            );
            vec![
                // Euclidean markers on both sides.
                (
                    QDyn::from(v + w),
                    op("skoro.add", &["quantity", "quantity"]),
                ),
                ((v - w).into(), op("skoro.sub", &["quantity", "quantity"])),
                ((-v).into(), op("skoro.neg", &["quantity"])),
                ((v * w).into(), op("skoro.mul", &["quantity", "quantity"])),
                ((m + m).into(), op("skoro.add", &["quantity", "quantity"])),
                ((vd - vd).into(), op("skoro.sub", &["quantity", "quantity"])),
                ((md * md).into(), op("skoro.mul", &["quantity", "quantity"])),
                ((s + s).into(), op("skoro.add", &["quantity", "quantity"])),
                // A `Real` and a Euclidean marker, either way round: the `Real` comes first.
                ((s * v).into(), op("skoro.scale", &["quantity", "quantity"])),
                ((v * s).into(), op("skoro.scale", &["quantity", "quantity"])),
                ((m * s).into(), op("skoro.scale", &["quantity", "quantity"])),
                // A literal and a Euclidean marker.
                (
                    (2.0 * v).into(),
                    op("skoro.scale", &["constant", "quantity"]),
                ),
                (
                    (v * 2.0).into(),
                    op("skoro.scale", &["constant", "quantity"]),
                ),
                (
                    (v / 2.0).into(),
                    op("skoro.scale", &["constant", "quantity"]),
                ),
                ((v + 1.0).into(), op("skoro.add", &["quantity", "constant"])),
                ((v - 1.0).into(), op("skoro.add", &["quantity", "constant"])),
                (
                    (1.0 - v).into(),
                    op("skoro.add", &["skoro.neg", "constant"]),
                ),
                // Positive markers.
                (
                    (p * q).into(),
                    op("skoro.positive.mul", &["quantity", "quantity"]),
                ),
                (
                    (p / q).into(),
                    op("skoro.positive.mul", &["quantity", "skoro.positive.recip"]),
                ),
                (
                    (3.0 * p).into(),
                    op("skoro.positive.mul", &["constant", "quantity"]),
                ),
                (
                    (pd * pd).into(),
                    op("skoro.positive.mul", &["quantity", "quantity"]),
                ),
                // Integers.
                (
                    (i + j).into(),
                    op("skoro.integer.add", &["quantity", "quantity"]),
                ),
                (
                    (i - j).into(),
                    op("skoro.integer.sub", &["quantity", "quantity"]),
                ),
                (
                    (i * j).into(),
                    op("skoro.integer.mul", &["quantity", "quantity"]),
                ),
                // The same rows through QDyn.
                (vu + wu, op("skoro.add", &["quantity", "quantity"])),
                (vu - wu, op("skoro.sub", &["quantity", "quantity"])),
                (-vu, op("skoro.neg", &["quantity"])),
                (vu * wu, op("skoro.mul", &["quantity", "quantity"])),
                (su * vu, op("skoro.scale", &["quantity", "quantity"])),
                (vu * su, op("skoro.scale", &["quantity", "quantity"])),
                (2.0 * vu, op("skoro.scale", &["constant", "quantity"])),
                (vu + 1.0, op("skoro.add", &["quantity", "constant"])),
                (pu * qu, op("skoro.positive.mul", &["quantity", "quantity"])),
                (
                    pu / qu,
                    op("skoro.positive.mul", &["quantity", "skoro.positive.recip"]),
                ),
                (
                    3.0 * pu,
                    op("skoro.positive.mul", &["constant", "quantity"]),
                ),
                (iu + ju, op("skoro.integer.add", &["quantity", "quantity"])),
                (iu * ju, op("skoro.integer.mul", &["quantity", "quantity"])),
                // The methods and functions of the operations table.
                (
                    m.matmul(v).into(),
                    op("skoro.matmul", &["quantity", "quantity"]),
                ),
                (
                    md.matmul(vd).into(),
                    op("skoro.matmul", &["quantity", "quantity"]),
                ),
                (v.sum().into(), op("skoro.sum", &["quantity"])),
                (v.exp().into(), op("skoro.exp", &["quantity"])),
                (v.log().into(), op("skoro.log", &["quantity"])),
                (m.transpose().into(), op("skoro.transpose", &["quantity"])),
                (v.sin().into(), op("skoro.sin", &["quantity"])),
                (v.cos().into(), op("skoro.cos", &["quantity"])),
                (
                    v.atan2(w).into(),
                    op("skoro.atan2", &["quantity", "quantity"]),
                ),
                (v.slice(1..3).into(), op("skoro.slice", &["quantity"])),
                (
                    m.reshape(VectorDyn(6)).into(),
                    op("skoro.concatenate", &["quantity"]),
                ),
                (
                    scope::concatenate(&[vu, su], Space::Vector(4)),
                    op("skoro.concatenate", &["quantity", "quantity"]),
                ),
                (
                    scope::select(k, &[v, w, v]).into(),
                    op(
                        "skoro.select",
                        &["quantity", "quantity", "quantity", "quantity"],
                    ),
                ),
                (
                    p.as_real().into(),
                    op("skoro.positive.as_real", &["quantity"]),
                ),
                (
                    v.exp_positive().into(),
                    op("skoro.positive.exp", &["quantity"]),
                ),
                (p.log().into(), op("skoro.positive.log", &["quantity"])),
                (pu.log(), op("skoro.positive.log", &["quantity"])),
                (vu.log(), op("skoro.log", &["quantity"])),
                (p.recip().into(), op("skoro.positive.recip", &["quantity"])),
                (
                    i.eq(j).into(),
                    op("skoro.integer.eq", &["quantity", "quantity"]),
                ),
                (
                    i.lt(j).into(),
                    op("skoro.integer.lt", &["quantity", "quantity"]),
                ),
                (
                    i.le(j).into(),
                    op("skoro.integer.le", &["quantity", "quantity"]),
                ),
                (i.to_real().into(), op("skoro.to_real", &["quantity"])),
                (k.to_real().into(), op("skoro.to_real", &["quantity"])),
                (
                    s.is_nonnegative().into(),
                    op("skoro.is_nonnegative", &["quantity"]),
                ),
                (
                    scope::product(&[su, pu]).into(),
                    op("skoro.concatenate", &["quantity", "quantity"]),
                ),
            ]
        })
        .unwrap();
    let snapshot = writer.snapshot();
    for (q, expected) in nodes {
        assert!(!q.is_failed(), "{expected:?}");
        assert_eq!(named(&snapshot, q), expected);
    }
}

#[test]
fn products_slice_into_their_factors() {
    let mut writer = Writer::new();
    let (product, second) = writer
        .build(|| {
            let product = scope::product(&[
                quantity("a", Vector::<2>).untyped(),
                quantity("b", Positive::<1>).untyped(),
            ]);
            (product, product.factor(1))
        })
        .unwrap();
    let snapshot = writer.snapshot();
    let space = Space::product([Space::Vector(2), Space::Positive(1)]).unwrap();
    assert_eq!(snapshot.space(product.id()), Some(&space));
    assert_eq!(snapshot.space(second.id()), Some(&Space::Positive(1)));
    let failure = writer
        .build(|| {
            let i = quantity("i", Integer).untyped();
            scope::product(&[i])
        })
        .unwrap_err();
    assert_eq!(
        failure.error,
        Error::NotContinuous {
            space: Space::Integer
        }
    );
}

#[test]
fn literals_fold_and_positive_literals_are_positive() {
    let mut writer = Writer::new();
    let folded = writer
        .build(|| 2.0 * constant(Vector::<2>, [1.0, -3.0]) + 1.0)
        .unwrap();
    let node = writer.snapshot();
    let value = node.node(folded.id()).unwrap().value().unwrap().clone();
    assert_eq!(
        value.data(),
        &skoro_tensor::HostData::Continuous(vec![3.0, -5.0])
    );
    let failure = writer
        .build(|| -1.0 * quantity("p", Positive::<1>))
        .unwrap_err();
    assert_eq!(
        failure.error,
        Error::ValueShape {
            space: Space::Positive(1)
        }
    );
}

#[test]
fn dyn_operands_of_different_spaces_fail() {
    let mut writer = Writer::new();
    let mut line = 0;
    let failure = writer
        .build(|| {
            let a = quantity("a", VectorDyn(3));
            let b = quantity("b", VectorDyn(4));
            line = line!() + 1;
            let sum = a + b;
            assert!(sum.is_failed());
        })
        .unwrap_err();
    assert!(
        matches!(&failure.error, Error::SpaceMismatch { operation: "skoro.add", inputs }
            if *inputs == [Space::Vector(3), Space::Vector(4)]),
        "{:?}",
        failure.error
    );
    assert_eq!(failure.location.line(), line);

    // Untyped, the operator dispatches on the spaces, and reports the operator it was.
    let failure = writer
        .build(|| {
            let a = quantity("a", VectorDyn(3)).untyped();
            let b = quantity("b", VectorDyn(4)).untyped();
            line = line!() + 1;
            let _ = a + b;
        })
        .unwrap_err();
    assert!(matches!(
        failure.error,
        Error::SpaceMismatch { operation: "+", .. }
    ));
    assert_eq!(failure.location.line(), line);
}
