#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)] // a test crate
//! The acceptance test of the mathematics: the retraction of a runtime product with a curved
//! factor, and the chart derivatives of a nonlinear function through it, against the factor-wise
//! computation, finite differences, and the Hessian of the pull-back.

mod common;

use common::sphere::{Sphere, basis, points};
use skoro::scope::{constant, product, quantity_in};
use skoro::tensor::{Frame, HostData, HostValue, Value};
use skoro::testing::Tolerance;
use skoro::{
    Declares, Frames, NodeId, PositiveDyn, Precision, QDyn, Query, Space, Target, Values, Vector,
    Writer,
};

const ROWS: usize = 2;

/// Two Euclidean factors, a real line, a positive factor and the sphere.
fn factors() -> Vec<Space> {
    vec![
        Space::Vector(2),
        Space::Matrix(2, 2),
        Space::Real,
        Space::Positive(2),
        Sphere.space(),
    ]
}

fn product_space() -> Space {
    Space::product(factors()).unwrap()
}

/// The point of each factor at each row, in storage.
fn factor_points() -> Vec<Vec<f64>> {
    let sphere = points(ROWS);
    let mut out = vec![Vec::new(); 5];
    for (row, s) in sphere.iter().enumerate() {
        let r = row as f64;
        out[0].extend([0.4 - 0.3 * r, -0.2 + 0.5 * r]);
        out[1].extend([0.3, -0.6 + r, 0.8, 0.1 * r - 0.2]);
        out[2].push(0.7 - 0.9 * r);
        out[3].extend([0.6 + 0.2 * r, 1.4 - 0.3 * r]);
        out[4].extend(s);
    }
    out
}

/// The product's point at each row: its factors' storage, concatenated.
fn product_point() -> Vec<f64> {
    let parts = factor_points();
    let widths: Vec<usize> = factors().iter().map(Space::storage).collect();
    (0..ROWS)
        .flat_map(|row| {
            parts
                .iter()
                .zip(&widths)
                .flat_map(move |(part, w)| part[row * w..(row + 1) * w].to_vec())
        })
        .collect()
}

fn load(data: &[f64], target: &Target) -> Value {
    HostValue::new(
        Some(ROWS),
        data.len() / ROWS,
        HostData::Continuous(data.to_vec()),
    )
    .unwrap()
    .load(ROWS, target)
    .unwrap()
    .unwrap()
}

/// Entries of moderate size, fixed by `seed`.
fn entries(seed: usize, len: usize) -> Vec<f64> {
    (0..len)
        .map(|i| 0.8 * ((seed * 97 + i) as f64 * 0.61).sin())
        .collect()
}

fn frame(data: &[f64], dimension: usize, columns: usize, target: &Target) -> Frame {
    Frame::from_host(data, ROWS, dimension, columns, target)
}

fn host_value(values: &Values, q: impl Into<NodeId>) -> Vec<f64> {
    match values.get(q).unwrap() {
        Value::Continuous(points) => points.to_host(),
        Value::Discrete(_) => panic!("a continuous value"),
    }
}

fn host_frame(frames: &Frames, q: impl Into<NodeId>) -> Vec<f64> {
    frames.get(q).unwrap().to_host()
}

/// The `[ROWS, dimension, columns]` frame `data`, its tangent rows `range` alone.
fn block(
    data: &[f64],
    dimension: usize,
    columns: usize,
    range: &std::ops::Range<usize>,
) -> Vec<f64> {
    (0..ROWS)
        .flat_map(|row| {
            range
                .clone()
                .flat_map(move |i| (0..columns).map(move |c| (row * dimension + i) * columns + c))
        })
        .map(|index| data[index])
        .collect()
}

/// The frames of each block, `[ROWS, dᵢ, columns]`, stacked back into one of the product.
fn stack(blocks: &[Vec<f64>], dims: &[usize], columns: usize) -> Vec<f64> {
    let mut out = Vec::new();
    for row in 0..ROWS {
        for (data, d) in blocks.iter().zip(dims) {
            out.extend_from_slice(&data[row * d * columns..(row + 1) * d * columns]);
        }
    }
    out
}

/// Each entry within `ulps` units in the last place of the target's precision.
fn within_ulps(a: &[f64], b: &[f64], ulps: f64, precision: Precision, what: &str) {
    assert_eq!(a.len(), b.len(), "{what}: lengths");
    let epsilon = match precision {
        Precision::Single => f64::from(f32::EPSILON),
        Precision::Double => f64::EPSILON,
    };
    for (i, (x, y)) in a.iter().zip(b).enumerate() {
        let scale = x.abs().max(y.abs()).max(f64::MIN_POSITIVE);
        assert!(
            (x - y).abs() <= ulps * epsilon * scale,
            "{what}: entry {i}, {x} and {y}"
        );
    }
}

/// Each entry within `relative` of the other, against the larger of its magnitude and one.
fn close(a: &[f64], b: &[f64], relative: f64, what: &str) {
    assert_eq!(a.len(), b.len(), "{what}: lengths");
    for (i, (x, y)) in a.iter().zip(b).enumerate() {
        let scale = x.abs().max(y.abs()).max(1.0);
        assert!(
            (x - y).abs() <= relative * scale,
            "{what}: entry {i}, {x} and {y}"
        );
    }
}

/// A nonlinear `f` on the factors, reading the positive factor and the sphere through their
/// inclusions, with every pair of factors coupled somewhere.
fn f(parts: &[QDyn]) -> QDyn {
    let (v, m, r) = (parts[0], parts[1], parts[2]);
    let p = parts[3].narrow::<PositiveDyn>().as_real().untyped();
    let s = parts[4].include();
    let a = constant(Vector::<3>, [0.3, -0.7, 0.5]).untyped();
    let along = (s * a).sum();
    (v * v).sum() * 0.5
        + (m * m).sum() * 0.25
        + r * v.sum()
        + r * r * r * 0.1
        + (p * p).sum() * 0.3
        + along * p.sum()
        + along * along
        + along * r
        + (m.slice(0..2) * v).sum() * along
}

skoro::precision_tests! {
    fn product_retraction_agrees_with_its_factors(target) {
        let blocks = product_space().blocks();
        let d = product_space().dimension().unwrap();
        let mut writer = Writer::new();
        let (x, t, a, b) = writer
            .build(|| {
                let x = quantity_in("x", product_space());
                let t = quantity_in("t", Space::Vector(d));
                let parts: Vec<QDyn> = blocks
                    .iter()
                    .enumerate()
                    .map(|(i, block)| x.factor(i).retract(t.slice(block.tangent.clone())))
                    .collect();
                (x, t, x.retract(t), product(&parts).untyped())
            })
            .unwrap();
        let snapshot = writer.snapshot();
        let compile = |q: QDyn| snapshot.compile(&Query::of([q]), &target).unwrap();
        let (pa, pb) = (compile(a), compile(b));
        let at = Values::new()
            .with(x, load(&product_point(), &target))
            .with(t, load(&entries(1, ROWS * d), &target));
        let precision = target.precision();
        let agree = |p: &[f64], q: &[f64], what: &str| within_ulps(p, q, 4.0, precision, what);

        agree(
            &host_value(&pa.forward(&at).unwrap(), a),
            &host_value(&pb.forward(&at).unwrap(), b),
            "values",
        );
        let columns = 3;
        let tangents = Frames::new()
            .with(x, frame(&entries(2, ROWS * d * columns), d, columns, &target))
            .with(t, frame(&entries(3, ROWS * d * columns), d, columns, &target));
        agree(
            &host_frame(&pa.tangent(&at, &tangents).unwrap().1, a),
            &host_frame(&pb.tangent(&at, &tangents).unwrap().1, b),
            "tangent frames",
        );
        let seed = frame(&entries(4, ROWS * d * columns), d, columns, &target);
        let ca = pa.cotangent(&at, &Frames::new().with(a, seed.clone())).unwrap().1;
        let cb = pb.cotangent(&at, &Frames::new().with(b, seed)).unwrap().1;
        for q in [x, t] {
            agree(&host_frame(&ca, q), &host_frame(&cb, q), "cotangent frames");
        }
        let seed = frame(&entries(5, ROWS * d), d, 1, &target);
        let sa = pa.hessian(&at, &tangents, &Frames::new().with(a, seed.clone())).unwrap();
        let sb = pb.hessian(&at, &tangents, &Frames::new().with(b, seed)).unwrap();
        for q in [x, t] {
            agree(&host_frame(&sa.beta, q), &host_frame(&sb.beta, q), "Hessian products");
        }
    }

    fn chart_derivatives_agree_with_blocks_and_finite_differences(target) {
        let blocks = product_space().blocks();
        let d = product_space().dimension().unwrap();
        let dims: Vec<usize> = blocks.iter().map(|b| b.tangent.len()).collect();
        let mut writer = Writer::new();
        let (x, t, moved, f_product, ys, f_blocks) = writer
            .build(|| {
                let x = quantity_in("x", product_space());
                let t = quantity_in("t", Space::Vector(d));
                let parts: Vec<QDyn> = (0..blocks.len()).map(|i| x.factor(i)).collect();
                let ys: Vec<QDyn> = factors()
                    .into_iter()
                    .enumerate()
                    .map(|(i, space)| quantity_in(format!("y/{i}").as_str(), space))
                    .collect();
                (x, t, x.retract(t), f(&parts), ys.clone(), f(&ys))
            })
            .unwrap();
        let snapshot = writer.snapshot();
        let compile = |q: QDyn| snapshot.compile(&Query::of([q]), &target).unwrap();
        let (p_product, p_blocks, p_moved) = (compile(f_product), compile(f_blocks), compile(moved));
        let Tolerance { step, relative } = Tolerance::of(target.precision());

        let point = product_point();
        let at = Values::new().with(x, load(&point, &target));
        let at_blocks = factor_points()
            .iter()
            .zip(&ys)
            .fold(Values::new(), |at, (data, y)| at.with(*y, load(data, &target)));

        // The chart gradient, and the factor-wise assembly of its blocks.
        let gradient = |at: &Values| host_frame(&p_product.gradient(at).unwrap().1, x);
        let g = gradient(&at);
        let g_blocks = p_blocks.gradient(&at_blocks).unwrap().1;
        let assembled: Vec<Vec<f64>> = ys.iter().map(|y| host_frame(&g_blocks, *y)).collect();
        close(&g, &stack(&assembled, &dims, 1), relative, "the gradient's blocks");

        // Hessian products, the same way.
        let columns = 2;
        let velocities = entries(6, ROWS * d * columns);
        let second = p_product
            .hessian_products(&at, &Frames::new().with(x, frame(&velocities, d, columns, &target)))
            .unwrap();
        let hv = host_frame(&second.beta, x);
        let block_velocities = blocks.iter().zip(&ys).zip(&dims).fold(
            Frames::new(),
            |frames, ((b, y), dim)| {
                let data = block(&velocities, d, columns, &b.tangent);
                frames.with(*y, frame(&data, *dim, columns, &target))
            },
        );
        let second_blocks = p_blocks.hessian_products(&at_blocks, &block_velocities).unwrap();
        let assembled: Vec<Vec<f64>> =
            ys.iter().map(|y| host_frame(&second_blocks.beta, *y)).collect();
        close(&hv, &stack(&assembled, &dims, columns), relative, "the Hessian's blocks");

        // x ⊕ τ, for τ the same at every row.
        let retract = |tau: &[f64]| -> Values {
            let tangent: Vec<f64> = (0..ROWS).flat_map(|_| tau.to_vec()).collect();
            let moved_at = at.clone().with(t, load(&tangent, &target));
            let y = host_value(&p_moved.forward(&moved_at).unwrap(), moved);
            Values::new().with(x, load(&y, &target))
        };
        let f_at = |at: &Values| host_value(&p_product.forward(at).unwrap(), f_product);

        // The gradient by central differences of f ∘ R_x, coordinate by coordinate.
        for k in 0..d {
            let mut tau = vec![0.0; d];
            tau[k] = step;
            let plus = f_at(&retract(&tau));
            tau[k] = -step;
            let minus = f_at(&retract(&tau));
            let difference: Vec<f64> =
                plus.iter().zip(&minus).map(|(a, b)| (a - b) / (2.0 * step)).collect();
            let column: Vec<f64> = (0..ROWS).map(|row| g[row * d + k]).collect();
            close(&column, &difference, relative, &format!("the gradient's entry {k}"));
        }

        // The chart Hessian by central differences of the gradient along x ⊕ s v.
        for c in 0..columns {
            for row in 0..ROWS {
                let v: Vec<f64> = (0..d).map(|i| velocities[(row * d + i) * columns + c]).collect();
                let scaled = |s: f64| v.iter().map(|vi| s * vi).collect::<Vec<f64>>();
                let plus = gradient(&retract(&scaled(step)));
                let minus = gradient(&retract(&scaled(-step)));
                let difference: Vec<f64> = (0..d)
                    .map(|i| (plus[row * d + i] - minus[row * d + i]) / (2.0 * step))
                    .collect();
                let product: Vec<f64> = (0..d).map(|i| hv[(row * d + i) * columns + c]).collect();
                close(&product, &difference, relative, &format!("H v at row {row}, column {c}"));
            }
        }
    }
}

/// The sphere's `R(x, u)`, on the host.
fn sphere_retract(x: [f64; 3], u: [f64; 2]) -> [f64; 3] {
    let b = basis(x);
    let s = (u[0] * u[0] + u[1] * u[1]) / 4.0;
    std::array::from_fn(|i| ((1.0 - s) * x[i] + b[i][0] * u[0] + b[i][1] * u[1]) / (1.0 + s))
}

/// `Dφ_t(0) = L_x(y) B(y)` at `y = x_t`, for `L_x(y) = ∂_y Q(x, y)`: how a step in the chart at
/// `y` reads in the chart at `x`.
fn chart_change(x: [f64; 3], y: [f64; 3]) -> [[f64; 2]; 2] {
    let (bx, by) = (basis(x), basis(y));
    let denominator = 1.0 + (0..3).map(|i| x[i] * y[i]).sum::<f64>();
    let bty: [f64; 2] = std::array::from_fn(|a| (0..3).map(|i| bx[i][a] * y[i]).sum());
    // L[a][i] = 2 B(x)[i][a] / den - 2 (B(x)ᵀ y)[a] x[i] / den².
    let l = |a: usize, i: usize| {
        2.0 * bx[i][a] / denominator - 2.0 * bty[a] * x[i] / (denominator * denominator)
    };
    std::array::from_fn(|a| std::array::from_fn(|c| (0..3).map(|i| l(a, i) * by[i][c]).sum()))
}

#[test]
fn chart_hessian_differs_from_the_pullback_hessian_on_the_sphere_alone() {
    let target = Target::host();
    let blocks = product_space().blocks();
    let d = product_space().dimension().unwrap();
    let sphere = blocks[4].tangent.clone();
    let mut writer = Writer::new();
    let (x, t, f_chart, f_pullback) = writer
        .build(|| {
            let x = quantity_in("x", product_space());
            let t = quantity_in("t", Space::Vector(d));
            let moved = x.retract(t);
            let at_x: Vec<QDyn> = (0..blocks.len()).map(|i| x.factor(i)).collect();
            let at_moved: Vec<QDyn> = (0..blocks.len()).map(|i| moved.factor(i)).collect();
            (x, t, f(&at_x), f(&at_moved))
        })
        .unwrap();
    let snapshot = writer.snapshot();
    let compile = |q: QDyn| snapshot.compile(&Query::of([q]), &target).unwrap();
    let (p_chart, p_pullback) = (compile(f_chart), compile(f_pullback));

    let columns = 2;
    let velocities = entries(8, ROWS * d * columns);
    let v = frame(&velocities, d, columns, &target);
    let at = Values::new().with(x, load(&product_point(), &target));
    let g = host_frame(&p_chart.gradient(&at).unwrap().1, x);
    let chart = p_chart
        .hessian_products(&at, &Frames::new().with(x, v.clone()))
        .unwrap();
    let chart = host_frame(&chart.beta, x);
    // The pull-back f ∘ R_x, at t = 0, moving t alone.
    let at_zero = at.clone().with(t, load(&vec![0.0; ROWS * d], &target));
    let pullback = p_pullback
        .hessian_products(&at_zero, &Frames::new().with(t, v))
        .unwrap();
    let pullback = host_frame(&pullback.beta, t);

    let sphere_points = points(ROWS);
    let mut largest: f64 = 0.0;
    for row in 0..ROWS {
        for c in 0..columns {
            let entry = |data: &[f64], i: usize| data[(row * d + i) * columns + c];
            for i in (0..d).filter(|i| !sphere.contains(i)) {
                let (h, p) = (entry(&chart, i), entry(&pullback, i));
                assert!(
                    (h - p).abs() < 1e-12,
                    "entry {i}, off the sphere: {h} and {p}"
                );
            }
            // The correction on the sphere's block, (d/dt Dφ_t(0))ᵀ g_s, by central differences
            // of the closed forms.
            let xs = sphere_points[row];
            let vs = [
                entry(&velocities, sphere.start),
                entry(&velocities, sphere.start + 1),
            ];
            let gs = [g[row * d + sphere.start], g[row * d + sphere.start + 1]];
            let step = 1e-5;
            let at_step = |s: f64| chart_change(xs, sphere_retract(xs, [s * vs[0], s * vs[1]]));
            let (plus, minus) = (at_step(step), at_step(-step));
            for a in 0..2 {
                let correction: f64 = (0..2)
                    .map(|b| (plus[b][a] - minus[b][a]) / (2.0 * step) * gs[b])
                    .sum();
                let i = sphere.start + a;
                let difference = entry(&chart, i) - entry(&pullback, i);
                assert!(
                    (difference - correction).abs() < 1e-7,
                    "the sphere's entry {a}: {difference} and {correction}"
                );
                largest = largest.max(difference.abs());
            }
        }
    }
    assert!(
        largest > 1e-2,
        "away from a stationary point, the correction is not small: {largest}"
    );
}
