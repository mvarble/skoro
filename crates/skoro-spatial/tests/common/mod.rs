//! What the tests share: evaluating handles and composites on a target, the fixed and random rows,
//! and the rotations and motions on the host, written with matrices.
#![allow(dead_code, clippy::unwrap_used, clippy::expect_used, clippy::panic)]

pub mod host;

use std::sync::Arc;

use skoro_core::scope::quantity_in;
use skoro_core::testing::random;
use skoro_core::{Composite, Frames, QDyn, Query, Space, Values, Writer};
use skoro_tensor::{Frame, HostData, HostValue, Precision, Target, Value};

/// Values of one space, `rows` of them laid end to end.
#[derive(Clone, Debug)]
pub struct Rows {
    pub space: Space,
    pub data: Vec<f64>,
}

impl Rows {
    pub fn new(space: Space, data: Vec<f64>) -> Rows {
        Rows { space, data }
    }

    pub fn vectors<const N: usize>(rows: &[[f64; N]]) -> Rows {
        Rows::new(Space::Vector(N), rows.concat())
    }

    pub fn rows(&self) -> usize {
        self.data.len() / self.space.storage()
    }

    pub fn host(&self) -> HostValue {
        HostValue::new(
            Some(self.rows()),
            self.space.storage(),
            HostData::Continuous(self.data.clone()),
        )
        .unwrap()
    }

    pub fn input(&self) -> (Space, HostValue) {
        (self.space.clone(), self.host())
    }

    pub fn with_space(&self, space: Space) -> Rows {
        Rows::new(space, self.data.clone())
    }
}

fn load(rows: &Rows, batch: usize, target: &Target) -> Value {
    rows.host().load(batch, target).unwrap().unwrap()
}

fn continuous(value: &Value) -> Vec<f64> {
    match value {
        Value::Continuous(points) => points.to_host(),
        Value::Discrete(_) => panic!("a continuous value"),
    }
}

/// `f` at the quantities `inputs`, row by row, on `target`: one vector per output.
pub fn evaluate_many(
    f: impl FnOnce(&[QDyn]) -> Vec<QDyn>,
    inputs: &[Rows],
    target: &Target,
) -> Vec<Vec<f64>> {
    let mut writer = Writer::new();
    let (roots, outs) = writer
        .build(|| {
            let roots: Vec<QDyn> = inputs
                .iter()
                .enumerate()
                .map(|(i, rows)| quantity_in(i, rows.space.clone()))
                .collect();
            let outs = f(&roots);
            (roots, outs)
        })
        .unwrap();
    let program = writer
        .snapshot()
        .compile(&Query::of(outs.iter().copied()), target)
        .unwrap();
    let batch = inputs.iter().map(Rows::rows).max().unwrap_or(1);
    let mut at = Values::new();
    for (q, rows) in roots.iter().zip(inputs) {
        at.insert(*q, load(rows, batch, target));
    }
    let out = program.forward(&at).unwrap();
    outs.iter()
        .map(|q| continuous(out.get(*q).unwrap()))
        .collect()
}

/// The tangent pass of `f` at `inputs` along each tangent coordinate of input `wrt`, on
/// `target`: per row, the derivative in charts, `d_out × d_in`, laid out as `[row][i][c]`.
pub fn jacobian(
    f: impl FnOnce(&[QDyn]) -> QDyn,
    inputs: &[Rows],
    wrt: usize,
    target: &Target,
) -> Vec<f64> {
    let mut writer = Writer::new();
    let (roots, out) = writer
        .build(|| {
            let roots: Vec<QDyn> = inputs
                .iter()
                .enumerate()
                .map(|(i, rows)| quantity_in(i, rows.space.clone()))
                .collect();
            let out = f(&roots);
            (roots, out)
        })
        .unwrap();
    let program = writer
        .snapshot()
        .compile(&Query::of([out]), target)
        .unwrap();
    let batch = inputs.iter().map(Rows::rows).max().unwrap_or(1);
    let mut at = Values::new();
    for (q, rows) in roots.iter().zip(inputs) {
        at.insert(*q, load(rows, batch, target));
    }
    let d = inputs[wrt].space.dimension().unwrap();
    let identity: Vec<f64> = (0..batch)
        .flat_map(|_| (0..d).flat_map(move |i| (0..d).map(move |c| f64::from(u8::from(i == c)))))
        .collect();
    let frames = Frames::new().with(roots[wrt], Frame::from_host(&identity, batch, d, d, target));
    let (_, tangents) = program.tangent(&at, &frames).unwrap();
    tangents.get(out).unwrap().to_host()
}

/// The Hessian pass of a `Real`-valued `f` at `inputs`, seeded with one, along each tangent
/// coordinate of input `wrt`: per row, `β` at that input, `d × d`, laid out as `[row][i][c]`.
pub fn hessian(
    f: impl FnOnce(&[QDyn]) -> QDyn,
    inputs: &[Rows],
    wrt: usize,
    target: &Target,
) -> Vec<f64> {
    second_order(f, inputs, wrt, target).1
}

/// The same pass's `α` and `β` at input `wrt`: per row, the gradient, `d`, and the Hessian
/// products, `d × d`.
pub fn second_order(
    f: impl FnOnce(&[QDyn]) -> QDyn,
    inputs: &[Rows],
    wrt: usize,
    target: &Target,
) -> (Vec<f64>, Vec<f64>) {
    let mut writer = Writer::new();
    let (roots, out) = writer
        .build(|| {
            let roots: Vec<QDyn> = inputs
                .iter()
                .enumerate()
                .map(|(i, rows)| quantity_in(i, rows.space.clone()))
                .collect();
            let out = f(&roots);
            (roots, out)
        })
        .unwrap();
    let program = writer
        .snapshot()
        .compile(&Query::of([out]), target)
        .unwrap();
    let batch = inputs.iter().map(Rows::rows).max().unwrap_or(1);
    let mut at = Values::new();
    for (q, rows) in roots.iter().zip(inputs) {
        at.insert(*q, load(rows, batch, target));
    }
    let d = inputs[wrt].space.dimension().unwrap();
    let identity: Vec<f64> = (0..batch)
        .flat_map(|_| (0..d).flat_map(move |i| (0..d).map(move |c| f64::from(u8::from(i == c)))))
        .collect();
    let velocities =
        Frames::new().with(roots[wrt], Frame::from_host(&identity, batch, d, d, target));
    let second = program.hessian_products(&at, &velocities).unwrap();
    (
        second.alpha.get(roots[wrt]).unwrap().to_host(),
        second.beta.get(roots[wrt]).unwrap().to_host(),
    )
}

/// Rotation vectors in random directions at lengths from 0 to 6, at both sides of the cutoff.
pub fn rotation_vectors(seed: u64) -> Vec<[f64; 3]> {
    let lengths = [
        0.0,
        (0.9 * CUTOFF).sqrt(),
        (1.1 * CUTOFF).sqrt(),
        0.5,
        1.0,
        2.0,
        3.0,
        4.0,
        5.0,
        6.0,
    ];
    vectors::<3>(lengths.len(), seed, -1.0, 1.0)
        .into_iter()
        .zip(lengths)
        .map(|(d, length)| unit(d).map(|v| v * length))
        .collect()
}

/// Twists `(φ, t₁, …, t_k)`: the rotation vectors of `rotation_vectors` and translations drawn from
/// `seed`, laid end to end.
pub fn twists(k: usize, seed: u64) -> Vec<f64> {
    let phi = rotation_vectors(seed);
    let t = vectors::<3>(phi.len() * k, seed + 1, -2.0, 2.0);
    phi.iter()
        .enumerate()
        .flat_map(|(i, phi)| {
            phi.iter()
                .copied()
                .chain(t[i * k..(i + 1) * k].iter().flatten().copied())
                .collect::<Vec<_>>()
        })
        .collect()
}

/// `[row][i][c]` of matrices stored column by column, `n × n` each.
pub fn by_rows_layout(stored: &[f64], n: usize) -> Vec<f64> {
    stored
        .chunks(n * n)
        .flat_map(|m| (0..n).flat_map(move |i| (0..n).map(move |c| m[i + n * c])))
        .collect()
}

/// `f` at the quantities `inputs`, row by row, on `target`.
pub fn evaluate(f: impl FnOnce(&[QDyn]) -> QDyn, inputs: &[Rows], target: &Target) -> Vec<f64> {
    evaluate_many(|q| vec![f(q)], inputs, target).remove(0)
}

/// `f` at `inputs`, on the host in double precision.
pub fn on_host(f: impl FnOnce(&[QDyn]) -> QDyn, inputs: &[Rows]) -> Vec<f64> {
    evaluate(f, inputs, &Target::host())
}

/// A composite's program alone, run at `inputs` on `target`.
pub fn run(composite: &Arc<Composite>, inputs: &[Rows], target: &Target) -> Vec<f64> {
    let program = composite.compile(target).unwrap();
    let free = program.free();
    let batch = inputs.iter().map(Rows::rows).max().unwrap_or(1);
    let mut at = Values::new();
    for (free, rows) in free.iter().zip(inputs) {
        at.insert(free.node, load(rows, batch, target));
    }
    let out = program.forward(&at).unwrap();
    continuous(out.get(program.targets()[0]).unwrap())
}

/// The relative error allowed of a value computed at `target`'s precision, against the larger of
/// its magnitude and one.
pub fn tolerance(target: &Target) -> f64 {
    match target.precision() {
        Precision::Double => 1e-10,
        Precision::Single => 1e-4,
    }
}

pub fn close(got: &[f64], want: &[f64], relative: f64) {
    assert_eq!(got.len(), want.len(), "{got:?} against {want:?}");
    for (i, (g, w)) in got.iter().zip(want).enumerate() {
        assert!(
            (g - w).abs() <= relative * w.abs().max(1.0),
            "entry {i}: {g} against {w}\n{got:?}\nagainst\n{want:?}"
        );
    }
}

/// `x / |x|`.
pub fn unit<const N: usize>(x: [f64; N]) -> [f64; N] {
    let n = x.iter().map(|v| v * v).sum::<f64>().sqrt();
    x.map(|v| v / n)
}

/// `rows` unit quaternions drawn from `seed`.
pub fn random_quaternions(rows: usize, seed: u64) -> Vec<[f64; 4]> {
    vectors::<4>(rows, seed, -1.0, 1.0)
        .into_iter()
        .map(unit)
        .collect()
}

/// `rows` vectors drawn uniformly in `(low, high)` from `seed`.
pub fn vectors<const N: usize>(rows: usize, seed: u64, low: f64, high: f64) -> Vec<[f64; N]> {
    let HostData::Continuous(data) = random(&Space::Vector(N), rows, seed, low, high)
        .data()
        .clone()
    else {
        panic!("continuous draws")
    };
    data.as_chunks::<N>().0.to_vec()
}

/// The quaternion of the rotation by `angle` about `axis`.
pub fn about(axis: [f64; 3], angle: f64) -> [f64; 4] {
    let a = unit(axis);
    let (s, c) = (0.5 * angle).sin_cos();
    [c, s * a[0], s * a[1], s * a[2]]
}

/// The squared-angle cutoff of the series in `θ²`, and the cutoff of `L`'s series in `n²/w²`.
pub const CUTOFF: f64 = 1.0;
pub const LOG_CUTOFF: f64 = 0.01;

/// The fixed rows: the identity; rotations whose angles sit at 0.9 and 1.1 times the cutoff of
/// `θ²` and of `L`'s `n²/w²`; one with `w < 0`; and the half-turns about each axis.
pub fn fixed_quaternions() -> Vec<[f64; 4]> {
    let axis = [0.3, -0.8, 0.5];
    let mut rows = vec![[1.0, 0.0, 0.0, 0.0]];
    for factor in [0.9, 1.1] {
        rows.push(about(axis, (factor * CUTOFF).sqrt()));
        // n/w = tan(θ/2), so n²/w² at `factor` times the cutoff.
        rows.push(about(axis, 2.0 * (factor * LOG_CUTOFF).sqrt().atan()));
    }
    rows.push(about(axis, 4.0));
    for e in [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]] {
        rows.push(about(e, std::f64::consts::PI));
    }
    rows
}

/// The fixed rows and `random` more, each a unit quaternion.
pub fn quaternions(random: usize, seed: u64) -> Vec<[f64; 4]> {
    let mut rows = fixed_quaternions();
    rows.extend(random_quaternions(random, seed));
    rows
}

/// Each quaternion followed by `k` translations drawn from `seed`.
pub fn with_translations(quaternions: &[[f64; 4]], k: usize, seed: u64) -> Vec<f64> {
    let translations = vectors::<3>(quaternions.len() * k, seed, -2.0, 2.0);
    quaternions
        .iter()
        .enumerate()
        .flat_map(|(i, q)| {
            q.iter()
                .copied()
                .chain(translations[i * k..(i + 1) * k].iter().flatten().copied())
                .collect::<Vec<_>>()
        })
        .collect()
}
