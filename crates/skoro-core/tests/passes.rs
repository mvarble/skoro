#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)] // a test crate
//! The passes over a compiled program, on a composite with fan-out and several roots.

use std::sync::Arc;

use skoro_core::scope::quantity;
use skoro_core::testing::Tolerance;
use skoro_core::{
    Composite, Frames, Key, Program, QDyn, Query, Real, Space, Values, Vector, Writer,
};
use skoro_tensor::{Frame, HostData, HostValue, Target, Value};

const ROWS: usize = 3;

/// The model: `f = sum(exp(x ⊙ y) + s sin(x) + x ⊙ x)`, through a composite, and the vector
/// `g = x ⊙ y + sin(y)`; `x` fans out to three operations.
struct Model {
    x: QDyn,
    y: QDyn,
    s: QDyn,
    f: QDyn,
    g: QDyn,
}

fn mix() -> Arc<Composite> {
    let v = Space::Vector(2);
    Composite::define(
        "test.mix",
        Vec::new(),
        &[v.clone(), v, Space::Real],
        |inputs| {
            let (x, y, s) = (inputs[0], inputs[1], inputs[2]);
            ((x * y).exp() + s * x.sin() + x * x).sum()
        },
    )
    .unwrap()
}

fn model(writer: &mut Writer) -> Model {
    let mix = mix();
    writer
        .build(|| {
            let x = quantity("x", Vector::<2>).untyped();
            let y = quantity("y", Vector::<2>).untyped();
            let s = quantity("s", Real).untyped();
            let f = mix.call(&[x, y, s]);
            let g = x * y + y.sin();
            Model { x, y, s, f, g }
        })
        .unwrap()
}

fn uniforms(seed: u64, n: usize) -> Vec<f64> {
    Key::from_seed(seed)
        .uniforms(n)
        .into_iter()
        .map(|u| 2.0 * u - 1.0)
        .collect()
}

fn value(data: &[f64], rows: usize, storage: usize, target: &Target) -> Value {
    HostValue::new(Some(rows), storage, HostData::Continuous(data.to_vec()))
        .unwrap()
        .load(rows, target)
        .unwrap()
        .unwrap()
}

/// Host values of the roots, `rows` of each.
struct Point {
    x: Vec<f64>,
    y: Vec<f64>,
    s: Vec<f64>,
}

impl Point {
    fn random(seed: u64, rows: usize) -> Point {
        Point {
            x: uniforms(seed, 2 * rows),
            y: uniforms(seed + 1, 2 * rows),
            s: uniforms(seed + 2, rows),
        }
    }

    fn values(&self, m: &Model, rows: usize, target: &Target) -> Values {
        Values::new()
            .with(m.x, value(&self.x, rows, 2, target))
            .with(m.y, value(&self.y, rows, 2, target))
            .with(m.s, value(&self.s, rows, 1, target))
    }

    /// The point moved by `h` along column `c` of the frames `[rows, d, columns]`.
    fn moved(&self, frames: &[&[f64]; 3], columns: usize, c: usize, h: f64, rows: usize) -> Point {
        let step = |x: &[f64], f: &[f64], d: usize| -> Vec<f64> {
            (0..rows * d)
                .map(|i| x[i] + h * f[i * columns + c])
                .collect()
        };
        Point {
            x: step(&self.x, frames[0], 2),
            y: step(&self.y, frames[1], 2),
            s: step(&self.s, frames[2], 1),
        }
    }
}

fn host(values: &Values, q: QDyn) -> Vec<f64> {
    match values.get(q).unwrap() {
        Value::Continuous(points) => points.to_host(),
        Value::Discrete(_) => panic!("a continuous value"),
    }
}

fn close(found: &[f64], expected: &[f64], relative: f64, what: &str) {
    assert_eq!(found.len(), expected.len(), "{what}");
    let scale = expected.iter().fold(1.0f64, |m, e| m.max(e.abs()));
    for (f, e) in found.iter().zip(expected) {
        assert!(
            (f - e).abs() <= relative * scale,
            "{what}: {found:?} against {expected:?}"
        );
    }
}

fn frames(m: &Model, data: &[&[f64]; 3], columns: usize, rows: usize, target: &Target) -> Frames {
    Frames::new()
        .with(m.x, Frame::from_host(data[0], rows, 2, columns, target))
        .with(m.y, Frame::from_host(data[1], rows, 2, columns, target))
        .with(m.s, Frame::from_host(data[2], rows, 1, columns, target))
}

fn compile(writer: &Writer, targets: &[QDyn], target: &Target) -> Arc<Program> {
    writer
        .snapshot()
        .compile(&Query::of(targets.iter().copied()), target)
        .unwrap()
}

skoro_core::precision_tests! {
    fn tangent_pass_matches_finite_differences(target) {
        let mut writer = Writer::new();
        let m = model(&mut writer);
        let program = compile(&writer, &[m.f, m.g], &target);
        let tolerance = Tolerance::of(target.precision());
        let point = Point::random(1, ROWS);
        let columns = 2;
        let t = [uniforms(10, ROWS * 4), uniforms(11, ROWS * 4), uniforms(12, ROWS * 2)];
        let t = [&t[0][..], &t[1][..], &t[2][..]];
        let (_, tangents) = program
            .tangent(&point.values(&m, ROWS, &target), &frames(&m, &t, columns, ROWS, &target))
            .unwrap();
        for (q, d) in [(m.f, 1), (m.g, 2)] {
            let found = tangents.get(q).unwrap().to_host();
            let mut expected = vec![0.0; ROWS * d * columns];
            for c in 0..columns {
                let h = tolerance.step;
                let up = program.forward(&point.moved(&t, columns, c, h, ROWS).values(&m, ROWS, &target)).unwrap();
                let down = program.forward(&point.moved(&t, columns, c, -h, ROWS).values(&m, ROWS, &target)).unwrap();
                let (up, down) = (host(&up, q), host(&down, q));
                for i in 0..ROWS * d {
                    expected[i * columns + c] = (up[i] - down[i]) / (2.0 * h);
                }
            }
            close(&found, &expected, tolerance.relative, "the tangent pass");
        }
    }

    fn cotangent_pass_is_its_transpose(target) {
        let mut writer = Writer::new();
        let m = model(&mut writer);
        let program = compile(&writer, &[m.f, m.g], &target);
        let point = Point::random(2, ROWS);
        let at = point.values(&m, ROWS, &target);
        let columns = 2;
        let t = [uniforms(20, ROWS * 4), uniforms(21, ROWS * 4), uniforms(22, ROWS * 2)];
        let t = [&t[0][..], &t[1][..], &t[2][..]];
        let (_, pushed) = program.tangent(&at, &frames(&m, &t, columns, ROWS, &target)).unwrap();
        let (af, ag) = (uniforms(23, ROWS * columns), uniforms(24, ROWS * 2 * columns));
        let seeds = Frames::new()
            .with(m.f, Frame::from_host(&af, ROWS, 1, columns, &target))
            .with(m.g, Frame::from_host(&ag, ROWS, 2, columns, &target));
        let (_, pulled) = program.cotangent(&at, &seeds).unwrap();
        // ⟨α, J t⟩ = Σ_q ⟨(Jᵀ α)_q, t_q⟩, per row and column.
        let pair = |a: &[f64], b: &[f64], d: usize| -> Vec<f64> {
            let mut out = vec![0.0; ROWS * columns];
            for r in 0..ROWS {
                for i in 0..d {
                    for c in 0..columns {
                        let at = (r * d + i) * columns + c;
                        out[r * columns + c] += a[at] * b[at];
                    }
                }
            }
            out
        };
        let sum = |parts: Vec<Vec<f64>>| -> Vec<f64> {
            parts.into_iter().reduce(|a, b| a.iter().zip(&b).map(|(a, b)| a + b).collect()).unwrap()
        };
        let left = sum(vec![
            pair(&af, &pushed.get(m.f).unwrap().to_host(), 1),
            pair(&ag, &pushed.get(m.g).unwrap().to_host(), 2),
        ]);
        let right = sum(vec![
            pair(&pulled.get(m.x).unwrap().to_host(), t[0], 2),
            pair(&pulled.get(m.y).unwrap().to_host(), t[1], 2),
            pair(&pulled.get(m.s).unwrap().to_host(), t[2], 1),
        ]);
        close(&right, &left, Tolerance::AGREEMENT, "the cotangent pass against the tangent pass");
    }

    fn hessian_pass_matches_finite_differences_of_the_gradient(target) {
        let mut writer = Writer::new();
        let m = model(&mut writer);
        let program = compile(&writer, &[m.f], &target);
        let tolerance = Tolerance::of(target.precision());
        let point = Point::random(3, ROWS);
        let columns = 3;
        let v = [uniforms(30, ROWS * 2 * columns), uniforms(31, ROWS * 2 * columns), uniforms(32, ROWS * columns)];
        let v = [&v[0][..], &v[1][..], &v[2][..]];
        let second = program
            .hessian_products(&point.values(&m, ROWS, &target), &frames(&m, &v, columns, ROWS, &target))
            .unwrap();
        let (_, gradient) = program.gradient(&point.values(&m, ROWS, &target)).unwrap();
        for (q, d) in [(m.x, 2), (m.y, 2), (m.s, 1)] {
            close(&second.alpha.get(q).unwrap().to_host(), &gradient.get(q).unwrap().to_host(), Tolerance::AGREEMENT, "α is the gradient");
            let found = second.beta.get(q).unwrap().to_host();
            let mut expected = vec![0.0; ROWS * d * columns];
            for c in 0..columns {
                let h = tolerance.step;
                let grad = |h: f64| {
                    let at = point.moved(&v, columns, c, h, ROWS).values(&m, ROWS, &target);
                    program.gradient(&at).unwrap().1.get(q).unwrap().to_host()
                };
                let (up, down) = (grad(h), grad(-h));
                for i in 0..ROWS * d {
                    expected[i * columns + c] = (up[i] - down[i]) / (2.0 * h);
                }
            }
            close(&found, &expected, tolerance.relative, "β against central differences of the gradient");
        }
    }

    fn hessian_is_symmetric_on_flat_spaces(target) {
        let mut writer = Writer::new();
        let m = model(&mut writer);
        let program = compile(&writer, &[m.f], &target);
        let point = Point::random(4, 1);
        // The identity frame over the five coordinates, x then y then s.
        let n = 5;
        let unit = |offset: usize, d: usize| -> Vec<f64> {
            (0..d).flat_map(|i| (0..n).map(move |c| if c == offset + i { 1.0 } else { 0.0 })).collect()
        };
        let v = [unit(0, 2), unit(2, 2), unit(4, 1)];
        let v = [&v[0][..], &v[1][..], &v[2][..]];
        let second = program
            .hessian_products(&point.values(&m, 1, &target), &frames(&m, &v, n, 1, &target))
            .unwrap();
        let rows: Vec<f64> = [m.x, m.y, m.s]
            .iter()
            .flat_map(|q| second.beta.get(*q).unwrap().to_host())
            .collect();
        let h = |i: usize, j: usize| rows[i * n + j];
        for i in 0..n {
            for j in 0..i {
                let scale = h(i, j).abs().max(h(j, i).abs()).max(1.0);
                assert!((h(i, j) - h(j, i)).abs() <= Tolerance::AGREEMENT * scale, "H[{i}][{j}]");
            }
        }
    }

    fn unseeded_root_still_receives_its_cotangents(target) {
        let mut writer = Writer::new();
        let m = model(&mut writer);
        let program = compile(&writer, &[m.f], &target);
        let point = Point::random(5, ROWS);
        let at = point.values(&m, ROWS, &target);
        let only_x = Frames::new().with(m.x, Frame::from_host(&uniforms(50, ROWS * 2), ROWS, 2, 1, &target));
        let second = program.hessian_products(&at, &only_x).unwrap();
        let (_, gradient) = program.gradient(&at).unwrap();
        for q in [m.x, m.y, m.s] {
            close(&second.alpha.get(q).unwrap().to_host(), &gradient.get(q).unwrap().to_host(), Tolerance::AGREEMENT, "α at every root");
            assert!(second.beta.get(q).is_some(), "β at every root");
        }
    }

    fn fan_out_accumulates_cotangents(target) {
        let mut writer = Writer::new();
        let (x, f) = writer
            .build(|| {
                let x = quantity("x", Vector::<2>).untyped();
                (x, (x * x + x.sin() + x).sum())
            })
            .unwrap();
        let program = compile(&writer, &[f], &target);
        let data = [0.3, -1.2];
        let at = Values::new().with(x, value(&data, 1, 2, &target));
        let (_, gradient) = program.gradient(&at).unwrap();
        let expected: Vec<f64> = data.iter().map(|x| 2.0 * x + x.cos() + 1.0).collect();
        close(&gradient.get(x).unwrap().to_host(), &expected, Tolerance::AGREEMENT, "the sum along every consumer");
    }

    fn rows_do_not_interact(target) {
        let mut writer = Writer::new();
        let m = model(&mut writer);
        let program = compile(&writer, &[m.f], &target);
        let point = Point::random(6, ROWS);
        let v = [uniforms(60, ROWS * 2), uniforms(61, ROWS * 2), uniforms(62, ROWS)];
        let v = [&v[0][..], &v[1][..], &v[2][..]];
        let all = program
            .hessian_products(&point.values(&m, ROWS, &target), &frames(&m, &v, 1, ROWS, &target))
            .unwrap();
        for r in 0..ROWS {
            let row = Point {
                x: point.x[2 * r..2 * r + 2].to_vec(),
                y: point.y[2 * r..2 * r + 2].to_vec(),
                s: point.s[r..r + 1].to_vec(),
            };
            let vr = [&v[0][2 * r..2 * r + 2], &v[1][2 * r..2 * r + 2], &v[2][r..r + 1]];
            let one = program
                .hessian_products(&row.values(&m, 1, &target), &frames(&m, &vr, 1, 1, &target))
                .unwrap();
            for (q, d) in [(m.x, 2), (m.y, 2), (m.s, 1)] {
                let a = all.beta.get(q).unwrap().to_host();
                let b = one.beta.get(q).unwrap().to_host();
                close(&a[r * d..(r + 1) * d], &b, Tolerance::AGREEMENT, "a row alone");
            }
        }
    }

    fn values_are_freed_after_their_last_reader(target) {
        let mut writer = Writer::new();
        let (x, y) = writer
            .build(|| {
                let x = quantity("x", Vector::<2>).untyped();
                let mut y = x;
                for _ in 0..20 {
                    y = y.sin();
                }
                (x, y)
            })
            .unwrap();
        let program = compile(&writer, &[y], &target);
        assert_eq!(program.len(), 20);
        // The root, then each value until the step after it has read it: at most three at once.
        assert!(program.peak_values() <= 3, "{}", program.peak_values());
        let at = Values::new().with(x, value(&[0.5, -0.5], 1, 2, &target));
        let expected: Vec<f64> = [0.5f64, -0.5].iter().map(|&x| (0..20).fold(x, |y, _| y.sin())).collect();
        close(&host(&program.forward(&at).unwrap(), y), &expected, Tolerance::AGREEMENT, "the chain");
    }
}
