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

/// A problem's rows over all of `D`, stacked, and its right-hand sides.
pub fn dense_rows(problem: &skoro_elimination::Problem) -> (Vec<Vec<f64>>, Vec<f64>) {
    let structure = problem.structure();
    let d = structure.dimension();
    let (mut a, mut b) = (Vec::new(), Vec::new());
    for (blocks, rows) in structure.factors().iter().zip(problem.factors()) {
        for r in 0..rows.b.len() {
            let mut row = vec![0.0; d];
            let mut column = 0;
            for &k in blocks {
                for c in structure.range(k) {
                    row[c] = rows.a.get(r, column);
                    column += 1;
                }
            }
            a.push(row);
            b.push(rows.b[r]);
        }
    }
    (a, b)
}

/// `JᵀJ`, from a problem's rows.
pub fn gram(problem: &skoro_elimination::Problem) -> Vec<Vec<f64>> {
    let d = problem.structure().dimension();
    let (a, _) = dense_rows(problem);
    (0..d)
        .map(|i| {
            (0..d)
                .map(|j| a.iter().map(|row| row[i] * row[j]).sum())
                .collect()
        })
        .collect()
}

/// `g = ∇f`, the query's negative log-density's chart gradient at `point`, per row, stacked in
/// the layout's order, by the core's cotangent pass.
pub fn gradient(
    snapshot: &Snapshot,
    query: &Query,
    layout: &Layout,
    point: &Values,
    target: &Target,
) -> Vec<Vec<f64>> {
    let program = snapshot.compile(query, target).unwrap();
    let (_, frames) = program.gradient(point).unwrap();
    let parts: Vec<(Vec<f64>, usize)> = layout
        .quantities()
        .iter()
        .zip(layout.blocks())
        .map(|(&q, block)| (frames.get(q).unwrap().to_host(), block.tangent.len()))
        .collect();
    let rows = parts.first().map_or(1, |(data, d)| data.len() / d.max(&1));
    (0..rows)
        .map(|r| {
            parts
                .iter()
                .flat_map(|(data, d)| data[r * d..(r + 1) * d].iter().map(|g| -g))
                .collect()
        })
        .collect()
}

/// `x` with `M x = v`, by Gaussian elimination with partial pivoting.
#[allow(clippy::needless_range_loop)]
pub fn solve(m: &[Vec<f64>], v: &[f64]) -> Vec<f64> {
    let n = v.len();
    let mut a: Vec<Vec<f64>> = m
        .iter()
        .zip(v)
        .map(|(r, v)| [r.clone(), vec![*v]].concat())
        .collect();
    for c in 0..n {
        let p = (c..n)
            .max_by(|&i, &j| a[i][c].abs().total_cmp(&a[j][c].abs()))
            .unwrap();
        a.swap(c, p);
        for r in c + 1..n {
            let f = a[r][c] / a[c][c];
            for k in c..=n {
                a[r][k] -= f * a[c][k];
            }
        }
    }
    let mut x = vec![0.0; n];
    for i in (0..n).rev() {
        let rest: f64 = (i + 1..n).map(|j| a[i][j] * x[j]).sum();
        x[i] = (a[i][n] - rest) / a[i][i];
    }
    x
}

/// The rows of a batch value, one `storage` slice each.
pub fn row(value: &Value, r: usize) -> Vec<f64> {
    match value {
        Value::Continuous(points) => {
            let s = points.storage();
            points.to_host()[r * s..(r + 1) * s].to_vec()
        }
        Value::Discrete(_) => panic!("a continuous value"),
    }
}

/// The values of `nodes`, given the observations of `query`, at `point` retracted along `steps`,
/// one step per row; with no steps, at `point` itself.
pub fn nodes_at(
    snapshot: &Snapshot,
    query: &Query,
    nodes: &[skoro_core::QDyn],
    layout: &Layout,
    point: &Values,
    steps: Option<&[Vec<f64>]>,
    target: &Target,
) -> Values {
    let program = snapshot
        .compile(
            &Query::of(nodes.iter().copied()).given(query.observations().clone()),
            target,
        )
        .unwrap();
    let at = match steps {
        Some(steps) => {
            let data: Vec<f64> = steps.concat();
            let tangent = Points::from_host(&data, steps.len(), layout.dimension(), target);
            layout.retract(point, &tangent, target).unwrap()
        }
        None => point.clone(),
    };
    program.forward(&at).unwrap()
}

/// Factor `f`'s rows over all of `D`.
pub fn factor_rows(problem: &skoro_elimination::Problem, f: usize) -> Vec<Vec<f64>> {
    let structure = problem.structure();
    let rows = &problem.factors()[f];
    (0..rows.b.len())
        .map(|r| {
            let mut row = vec![0.0; structure.dimension()];
            let mut column = 0;
            for &k in &structure.factors()[f] {
                for c in structure.range(k) {
                    row[c] = rows.a.get(r, column);
                    column += 1;
                }
            }
            row
        })
        .collect()
}

/// The chart gradient of the single `Real` node `node`, given the observations of `query`, at
/// `point`, per row, over the layout's coordinates: zero at the quantities it is not reached from.
pub fn node_gradient(
    snapshot: &Snapshot,
    query: &Query,
    node: skoro_core::QDyn,
    layout: &Layout,
    point: &Values,
    target: &Target,
) -> Vec<Vec<f64>> {
    let program = snapshot
        .compile(
            &Query::of([node]).given(query.observations().clone()),
            target,
        )
        .unwrap();
    let (values, frames) = program.gradient(point).unwrap();
    let rows = host(&values, node).len();
    (0..rows)
        .map(|r| {
            layout
                .quantities()
                .iter()
                .zip(layout.blocks())
                .flat_map(|(&q, block)| {
                    let d = block.tangent.len();
                    match frames.get(q) {
                        Some(frame) => frame.to_host()[r * d..(r + 1) * d].to_vec(),
                        None => vec![0.0; d],
                    }
                })
                .collect()
        })
        .collect()
}
