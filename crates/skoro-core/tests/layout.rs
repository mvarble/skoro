#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)] // a test crate
//! The layout of a query's quantities: the step `retract` and its inverse `local`, on every kind
//! of factor a product has, flat, positive and curved.

use std::sync::{Arc, LazyLock};

use skoro_core::scope::quantity;
use skoro_core::{
    Composite, Declares, Embedded, Error, Layout, NodeId, Param, Positive, Real, Space, Typed,
    Values, Vector, Writer,
};
use skoro_tensor::{HostData, HostValue, Points, Target, Value};

/// The circle of `tests/embedded.rs`: storage 2, dimension 1, its retraction a rotation.
fn circle() -> Arc<Embedded> {
    let rotation = Composite::define(
        "test.layout.rotation",
        Vec::new(),
        &[Space::Vector(2), Space::Vector(1)],
        |q| {
            let (x, t) = (q[0], q[1]);
            let (c, s) = (t.cos().reshape(Space::Real), t.sin().reshape(Space::Real));
            let (x1, x2) = (x.slice(0..1), x.slice(1..2));
            skoro_core::scope::concatenate(&[c * x1 - s * x2, s * x1 + c * x2], Space::Vector(2))
        },
    )
    .unwrap();
    let angle = Composite::define(
        "test.layout.angle",
        vec![Param::Float(1.0)],
        &[Space::Vector(2), Space::Vector(2)],
        |q| {
            let (x, y) = (q[0], q[1]);
            let (x1, x2, y1, y2) = (x.slice(0..1), x.slice(1..2), y.slice(0..1), y.slice(1..2));
            (x1 * y2 - x2 * y1).atan2(x1 * y1 + x2 * y2)
        },
    )
    .unwrap();
    Embedded::declare(
        "test.layout.circle",
        Vec::new(),
        2,
        1,
        rotation,
        angle,
        None,
    )
    .unwrap()
}

static CIRCLE: LazyLock<Arc<Embedded>> = LazyLock::new(circle);

#[derive(Clone, Copy, Debug)]
struct Circle;

impl Typed for Circle {
    fn admits(space: &Space) -> bool {
        *space == Space::Embedded(CIRCLE.clone())
    }
}

impl Declares for Circle {
    fn space(&self) -> Space {
        Space::Embedded(CIRCLE.clone())
    }
}

fn load(data: Vec<f64>, rows: usize, target: &Target) -> Value {
    let storage = data.len() / rows;
    HostValue::new(Some(rows), storage, HostData::Continuous(data))
        .unwrap()
        .load(rows, target)
        .unwrap()
        .unwrap()
}

fn host(values: &Values, id: NodeId) -> Vec<f64> {
    match values.get(id).unwrap() {
        Value::Continuous(points) => points.to_host(),
        Value::Discrete(_) => panic!("a continuous value"),
    }
}

/// A real, a vector, a positive vector and a circle, with their layout.
struct Setup {
    layout: Layout,
    ids: [NodeId; 4],
}

fn setup() -> Setup {
    let mut writer = Writer::new();
    let ids = writer
        .build(|| {
            [
                quantity("a", Real).id(),
                quantity("v", Vector::<2>).id(),
                quantity("p", Positive::<2>).id(),
                quantity("c", Circle).id(),
            ]
        })
        .unwrap();
    let layout = Layout::new(&writer.snapshot(), &ids).unwrap();
    Setup { layout, ids }
}

/// Three rows of a point of the layout, the circle's on the unit circle.
fn points(ids: &[NodeId; 4], target: &Target) -> Values {
    let angle = [0.4_f64, 1.7, -2.5];
    let circle: Vec<f64> = angle.iter().flat_map(|t| [t.cos(), t.sin()]).collect();
    Values::new()
        .with(ids[0], load(vec![0.5, -1.0, 2.0], 3, target))
        .with(
            ids[1],
            load(vec![0.1, 0.2, -0.3, 0.4, 1.0, -2.0], 3, target),
        )
        .with(ids[2], load(vec![1.0, 2.0, 0.5, 3.0, 4.0, 0.25], 3, target))
        .with(ids[3], load(circle, 3, target))
}

#[test]
fn local_inverts_retract() {
    let target = Target::host();
    let Setup { layout, ids } = setup();
    assert_eq!(layout.dimension(), 6);
    let at = points(&ids, &target);
    let tau = vec![
        0.1, 0.2, -0.3, 0.4, -0.5, 0.6, //
        -0.2, 0.1, 0.3, -0.4, 0.5, -0.6, //
        0.3, -0.3, 0.2, 0.1, 0.9, 1.2,
    ];
    let moved = layout
        .retract(&at, &Points::from_host(&tau, 3, 6, &target), &target)
        .unwrap();
    let back = layout.local(&moved, &at, &target).unwrap().to_host();
    assert_eq!(back.len(), tau.len());
    for (got, want) in back.iter().zip(&tau) {
        assert!((got - want).abs() < 1e-10, "{back:?} against {tau:?}");
    }
    // The tangent at a point to itself is zero.
    let zero = layout.local(&at, &at, &target).unwrap().to_host();
    assert!(zero.iter().all(|v| v.abs() < 1e-12), "{zero:?}");
    // The step is nothing but the retraction of the flat, positive and curved factors.
    let a = host(&moved, ids[0]);
    assert!((a[0] - 0.6).abs() < 1e-12);
}

#[test]
fn local_refuses_as_retract_does() {
    let target = Target::host();
    let Setup { layout, ids } = setup();
    let at = points(&ids, &target);
    let mut missing = Values::new();
    for id in &ids[..3] {
        missing.insert(*id, at.get(*id).unwrap().clone());
    }
    assert_eq!(
        layout.local(&missing, &at, &target).unwrap_err(),
        Error::MissingValue { id: ids[3] }
    );
    assert_eq!(
        layout.local(&at, &missing, &target).unwrap_err(),
        Error::MissingValue { id: ids[3] }
    );
    let discrete = HostValue::new(Some(3), 1, HostData::Discrete(vec![0, 1, 2]))
        .unwrap()
        .load(3, &target)
        .unwrap()
        .unwrap();
    let mut wrong = at.clone();
    wrong.insert(ids[0], discrete);
    assert!(matches!(
        layout.local(&wrong, &at, &target).unwrap_err(),
        Error::NotContinuous { .. }
    ));
}
