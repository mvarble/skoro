#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)] // a test crate
//! The sphere of the acceptance test, declared from `skoro`'s public interface alone.

mod common;

use std::sync::Arc;

use common::sphere::{SPHERE, Sphere, basis, basis_times, basis_transpose, points, reciprocal};
use skoro::primitive::Select;
use skoro::scope::{apply, constant, quantity};
use skoro::tensor::{Frame, HostData, HostValue, Value};
use skoro::testing::check_embedded;
use skoro::{
    Composite, Embedded, Frames, QDyn, Query, Real, Space, Target, Values, VectorDyn, Writer,
};

fn value(data: &[f64], rows: usize, target: &Target) -> Value {
    HostValue::new(
        Some(rows),
        data.len() / rows,
        HostData::Continuous(data.to_vec()),
    )
    .unwrap()
    .load(rows, target)
    .unwrap()
    .unwrap()
}

fn host(values: &Values, q: impl Into<skoro::NodeId>) -> Vec<f64> {
    match values.get(q).unwrap() {
        Value::Continuous(points) => points.to_host(),
        Value::Discrete(_) => panic!("a continuous value"),
    }
}

#[test]
fn sphere_basis_is_the_derivative_of_its_retraction() {
    let target = Target::host();
    let program = SPHERE.retraction().compile(&target).unwrap();
    let free = program.free();
    let xs = points(4);
    let flat: Vec<f64> = xs.iter().flatten().copied().collect();
    let at = Values::new()
        .with(free[0].node, value(&flat, 4, &target))
        .with(free[1].node, value(&[0.0; 8], 4, &target));
    let identity: Vec<f64> = (0..4).flat_map(|_| [1.0, 0.0, 0.0, 1.0]).collect();
    let frames = Frames::new().with(free[1].node, Frame::from_host(&identity, 4, 2, 2, &target));
    let (_, tangent) = program.tangent(&at, &frames).unwrap();
    let b = tangent.get(program.targets()[0]).unwrap().to_host();
    for (row, x) in xs.iter().enumerate() {
        let expected = basis(*x);
        let at = |i: usize, c: usize| b[(row * 3 + i) * 2 + c];
        for (i, expected) in expected.iter().enumerate() {
            for (c, expected) in expected.iter().enumerate() {
                assert!((at(i, c) - expected).abs() < 1e-12, "B({x:?})");
            }
        }
        for (a, c) in [(0, 0), (0, 1), (1, 1)] {
            let product: f64 = (0..3).map(|i| at(i, a) * at(i, c)).sum();
            let identity = if a == c { 1.0 } else { 0.0 };
            assert!((product - identity).abs() < 1e-12, "orthonormal columns");
        }
        for c in 0..2 {
            let normal: f64 = (0..3).map(|i| x[i] * at(i, c)).sum();
            assert!(normal.abs() < 1e-12, "columns orthogonal to x");
        }
    }
}

#[test]
fn sphere_retraction_reaches_the_far_side() {
    let target = Target::host();
    let mut writer = Writer::new();
    let (s, t, y, back) = writer
        .build(|| {
            let s = quantity("s", Sphere);
            let t = quantity("t", VectorDyn(2));
            let y = s.retract(t);
            (s, t, y, y.local(s))
        })
        .unwrap();
    let program = writer
        .snapshot()
        .compile(&Query::of([QDyn::from(y), QDyn::from(back)]), &target)
        .unwrap();
    let x = points(1)[0];
    let at = Values::new()
        .with(s, value(&x, 1, &target))
        .with(t, value(&[6.0, 8.0], 1, &target));
    let out = program.forward(&at).unwrap();
    let y = host(&out, y);
    let norm: f64 = y.iter().map(|v| v * v).sum::<f64>().sqrt();
    assert!((norm - 1.0).abs() < 1e-12, "on the sphere: {y:?}");
    let back = host(&out, back);
    assert!(
        (back[0] - 6.0).abs() < 1e-9 && (back[1] - 8.0).abs() < 1e-9,
        "{back:?}"
    );
}

/// How the private sphere computes its exponential map near `t = 0`.
#[derive(Clone, Copy, PartialEq)]
enum Near {
    /// The closed form alone: `0 / 0` at `t = 0`.
    Closed,
    /// Switched to its series by a selection, the closed form still evaluated at `t = 0`.
    Selected,
    /// Switched, and the closed form's input guarded by a second selection.
    Guarded,
}

/// The squared radius below which the series is used.
const CUT: f64 = 1e-8;

fn select(index: QDyn, values: [QDyn; 2]) -> QDyn {
    apply(Arc::new(Select), &[index, values[0], values[1]])
}

/// The exponential map, `R(x, t) = cos|t| x + (sin|t| / |t|) B(x) t`.
fn exponential(near: Near) -> Arc<Composite> {
    let name = match near {
        Near::Closed => "sphere.exp.closed",
        Near::Selected => "sphere.exp.selected",
        Near::Guarded => "sphere.exp.guarded",
    };
    Composite::define(
        name,
        Vec::new(),
        &[Space::Vector(3), Space::Vector(2)],
        move |q| {
            let (x, t) = (q[0], q[1]);
            let ss = (t * t).sum();
            let small = (-ss + CUT).is_nonnegative();
            let one = constant(Real, 1.0).untyped();
            let input = if near == Near::Guarded {
                select(small, [ss, one])
            } else {
                ss
            };
            let n = (input.log() * 0.5).exp();
            let (cos, sinc) = (n.cos(), n.sin() * reciprocal(n));
            let (cos, sinc) = match near {
                Near::Closed => (cos, sinc),
                Near::Selected | Near::Guarded => (
                    select(small, [cos, -ss * 0.5 + 1.0 + ss * ss * (1.0 / 24.0)]),
                    select(
                        small,
                        [sinc, -ss * (1.0 / 6.0) + 1.0 + ss * ss * (1.0 / 120.0)],
                    ),
                ),
            };
            cos * x + sinc * basis_times(x, t)
        },
    )
    .unwrap()
}

/// Its inverse, `Q(x, y) = (θ / r) B(x)ᵀ y` with `r = |B(x)ᵀ y|` and `θ = atan2(r, xᵀ y)`, guarded
/// at `r = 0` as the guarded retraction is.
fn logarithm() -> Arc<Composite> {
    Composite::define(
        "sphere.log",
        Vec::new(),
        &[Space::Vector(3), Space::Vector(3)],
        |q| {
            let (x, y) = (q[0], q[1]);
            let bty = basis_transpose(x, y);
            let rr = (bty * bty).sum();
            let small = (-rr + CUT).is_nonnegative();
            let one = constant(Real, 1.0).untyped();
            let r = (select(small, [rr, one]).log() * 0.5).exp();
            let theta = r.atan2((x * y).sum());
            let ratio = select(small, [theta * reciprocal(r), rr * (1.0 / 6.0) + 1.0]);
            ratio * bty
        },
    )
    .unwrap()
}

fn exponential_sphere(near: Near) -> Arc<Embedded> {
    let name = match near {
        Near::Closed => "sphere.exp.closed",
        Near::Selected => "sphere.exp.selected",
        Near::Guarded => "sphere.exp.guarded",
    };
    Embedded::declare(name, Vec::new(), 3, 2, exponential(near), logarithm()).unwrap()
}

#[test]
fn removable_singularity_is_reported() {
    let target = Target::host();
    let rows = 3;
    let flat: Vec<f64> = points(rows).iter().flatten().copied().collect();
    let points = HostValue::new(Some(rows), 3, HostData::Continuous(flat)).unwrap();

    let closed = check_embedded(&exponential_sphere(Near::Closed), &points, &target).unwrap_err();
    assert!(
        closed.contains("skoro.retract") && closed.contains("pass is not finite"),
        "{closed}"
    );

    let selected =
        check_embedded(&exponential_sphere(Near::Selected), &points, &target).unwrap_err();
    assert!(
        selected.contains("cotangent pass is not finite"),
        "{selected}"
    );

    if let Err(error) = check_embedded(&exponential_sphere(Near::Guarded), &points, &target) {
        panic!("{error}");
    }
}
