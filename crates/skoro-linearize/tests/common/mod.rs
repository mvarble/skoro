//! What the linearization's tests share: values on a target, the query's negative log-density at
//! retracted points, and comparisons.
#![allow(dead_code, clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use skoro_core::{Layout, NodeId, Query, Snapshot, Values};
use skoro_tensor::{Points, Precision, Target, Value};

/// `rows × storage` values, row by row, as a continuous value on `target`.
pub fn points(data: &[f64], rows: usize, storage: usize, target: &Target) -> Value {
    assert_eq!(data.len(), rows * storage);
    Value::Continuous(Points::from_host(data, rows, storage, target))
}

/// A continuous value on the host, row by row.
pub fn host(values: &Values, id: impl Into<NodeId>) -> Vec<f64> {
    match values.get(id.into()).unwrap() {
        Value::Continuous(points) => points.to_host(),
        Value::Discrete(_) => panic!("a continuous value"),
    }
}

/// `f(p̂ ⊕ τ) = -log p` of `query` at `point` retracted along `steps`, one step per row.
pub fn f_at(
    snapshot: &Snapshot,
    query: &Query,
    layout: &Layout,
    point: &Values,
    steps: &[Vec<f64>],
    target: &Target,
) -> Vec<f64> {
    let d = layout.dimension();
    let data: Vec<f64> = steps.concat();
    let tangent = Points::from_host(&data, steps.len(), d, target);
    let moved = layout.retract(point, &tangent, target).unwrap();
    let program = snapshot.compile(query, target).unwrap();
    host(&program.forward(&moved).unwrap(), NodeId::DENSITY)
        .into_iter()
        .map(|l| -l)
        .collect()
}

/// The relative error allowed of a value computed at `target`'s precision.
pub fn tolerance(target: &Target) -> f64 {
    match target.precision() {
        Precision::Double => 1e-10,
        Precision::Single => 2e-4,
    }
}

pub fn close(got: &[f64], want: &[f64], relative: f64, what: &str) {
    assert_eq!(got.len(), want.len(), "{what}: {got:?} against {want:?}");
    for (g, w) in got.iter().zip(want) {
        assert!(
            (g - w).abs() <= relative * w.abs().max(1.0),
            "{what}: {got:?} against {want:?}"
        );
    }
}
