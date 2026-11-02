//! What the normal's tests share: evaluating nodes, and dense linear algebra on the host.
#![allow(dead_code, clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use skoro_core::{NodeId, Observations, QDyn, Query, Snapshot, Values};
use skoro_tensor::{Precision, Target, Value};

/// The values of `nodes` on `target`, given `given`, each row by row.
pub fn evaluate(
    snapshot: &Snapshot,
    nodes: &[QDyn],
    given: Observations,
    target: &Target,
) -> Vec<Vec<f64>> {
    let query = Query::of(nodes.iter().copied()).given(given);
    let program = snapshot.compile(&query, target).unwrap();
    let out = program.forward(&Values::new()).unwrap();
    nodes.iter().map(|q| host(&out, q.id())).collect()
}

pub fn host(values: &Values, id: NodeId) -> Vec<f64> {
    match values.get(id).unwrap() {
        Value::Continuous(points) => points.to_host(),
        Value::Discrete(_) => panic!("a continuous value"),
    }
}

/// The relative error allowed of a value computed at `target`'s precision.
pub fn tolerance(target: &Target) -> f64 {
    match target.precision() {
        Precision::Double => 1e-11,
        Precision::Single => 5e-5,
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

/// The lower-triangular `L` with `A = LLᵀ`, row by row.
pub fn cholesky(a: &[f64], d: usize) -> Vec<f64> {
    let mut l = vec![0.0f64; d * d];
    for j in 0..d {
        let pivot = a[j * d + j] - (0..j).map(|k| l[j * d + k].powi(2)).sum::<f64>();
        l[j * d + j] = pivot.sqrt();
        for i in j + 1..d {
            let dot: f64 = (0..j).map(|k| l[i * d + k] * l[j * d + k]).sum();
            l[i * d + j] = (a[i * d + j] - dot) / l[j * d + j];
        }
    }
    l
}

/// `log N(t; 0, Σ)` on the host.
pub fn log_gaussian(t: &[f64], sigma: &[f64]) -> f64 {
    let d = t.len();
    let l = cholesky(sigma, d);
    let mut z = vec![0.0; d];
    for i in 0..d {
        let rest: f64 = (0..i).map(|j| l[i * d + j] * z[j]).sum();
        z[i] = (t[i] - rest) / l[i * d + i];
    }
    let log_det: f64 = (0..d).map(|i| l[i * d + i].ln()).sum();
    -0.5 * z.iter().map(|z| z * z).sum::<f64>()
        - log_det
        - 0.5 * d as f64 * (2.0 * std::f64::consts::PI).ln()
}

/// `A⁻¹`, row by row, of a symmetric positive definite `A`.
pub fn inverse(a: &[f64], d: usize) -> Vec<f64> {
    let l = cholesky(a, d);
    let mut inverse = vec![0.0; d * d];
    for c in 0..d {
        let mut y = vec![0.0; d];
        for i in 0..d {
            let rest: f64 = (0..i).map(|j| l[i * d + j] * y[j]).sum();
            y[i] = (if i == c { 1.0 } else { 0.0 } - rest) / l[i * d + i];
        }
        let mut x = vec![0.0; d];
        for i in (0..d).rev() {
            let rest: f64 = (i + 1..d).map(|j| l[j * d + i] * x[j]).sum();
            x[i] = (y[i] - rest) / l[i * d + i];
        }
        for r in 0..d {
            inverse[r * d + c] = x[r];
        }
    }
    inverse
}
