//! What the uniform's tests share: a term's value, and comparing values.
#![allow(dead_code, clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use skoro_core::{NodeId, Observations, Query, Snapshot, Values};
use skoro_tensor::{HostData, HostValue, Precision, Target, Value};

pub fn host(values: &Values, id: NodeId) -> Vec<f64> {
    match values.get(id).unwrap() {
        Value::Continuous(points) => points.to_host(),
        Value::Discrete(integers) => integers.to_host().into_iter().map(|i| i as f64).collect(),
    }
}

/// The log-density of the term `term`, row by row, given `given`.
pub fn density(
    snapshot: &Snapshot,
    term: NodeId,
    given: Observations,
    target: &Target,
) -> Vec<f64> {
    let query = Query::log_density_of([term]).given(given);
    let program = snapshot.compile(&query, target).unwrap();
    host(&program.forward(&Values::new()).unwrap(), NodeId::DENSITY)
}

/// `rows` points of storage `storage`, laid end to end.
pub fn continuous(data: &[f64], storage: usize) -> HostValue {
    let rows = data.len() / storage;
    HostValue::new(Some(rows), storage, HostData::Continuous(data.to_vec())).unwrap()
}

/// One category per row.
pub fn discrete(data: &[i64]) -> HostValue {
    HostValue::new(Some(data.len()), 1, HostData::Discrete(data.to_vec())).unwrap()
}

/// The relative error allowed of a value computed at `target`'s precision.
pub fn tolerance(target: &Target) -> f64 {
    match target.precision() {
        Precision::Double => 1e-14,
        Precision::Single => 1e-6,
    }
}

pub fn close(got: &[f64], want: &[f64], relative: f64) {
    assert_eq!(got.len(), want.len(), "{got:?} against {want:?}");
    for (g, w) in got.iter().zip(want) {
        assert!(
            (g - w).abs() <= relative * w.abs().max(1.0),
            "{got:?} against {want:?}"
        );
    }
}

/// `x / |x|`.
pub fn unit<const N: usize>(x: [f64; N]) -> [f64; N] {
    let n = x.iter().map(|v| v * v).sum::<f64>().sqrt();
    x.map(|v| v / n)
}

/// Three unit quaternions, laid end to end.
pub fn rotations() -> Vec<f64> {
    [
        [1.0, 0.0, 0.0, 0.0],
        unit([0.5, 0.1, -0.3, 0.7]),
        unit([-0.2, 0.9, 0.4, 0.1]),
    ]
    .concat()
}
