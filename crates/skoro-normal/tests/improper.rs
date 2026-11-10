#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)] // a test crate
//! The improper Gaussian factor: its value, its roles, its batches and its refusals, and the
//! statements and examples of its mathematics.

mod common;

use common::{cholesky, close, evaluate, inverse, tolerance};
use skoro_core::scope::quantity;
use skoro_core::{
    AtMostOne, Error, Frames, NodeId, Observations, Positive, Query, Real, Role, Rows, Values,
    Vector, Writer,
};
use skoro_normal::{Form, Improper, Scale, Stacked, roles};
use skoro_tensor::{Frame, HostData, HostValue, Target, Value};

/// `A`, given row by row, stored column by column as a `MatrixDyn` is.
fn by_columns(rows: &[&[f64]]) -> Vec<f64> {
    let d = rows.first().map_or(0, |row| row.len());
    (0..d)
        .flat_map(|j| rows.iter().map(move |row| row[j]))
        .collect()
}

/// `A δ - b` on the host, `A` row by row.
fn residual(a: &[&[f64]], b: &[f64], delta: &[f64]) -> Vec<f64> {
    a.iter()
        .zip(b)
        .map(|(row, b)| row.iter().zip(delta).map(|(a, d)| a * d).sum::<f64>() - b)
        .collect()
}

/// `-½|A δ - b|² - ηᵀδ + c` on the host, `A` row by row.
fn value(a: &[&[f64]], b: &[f64], eta: Option<&[f64]>, c: f64, delta: &[f64]) -> f64 {
    let squares: f64 = residual(a, b, delta).iter().map(|r| r * r).sum();
    let linear: f64 = eta.map_or(0.0, |eta| eta.iter().zip(delta).map(|(e, d)| e * d).sum());
    -0.5 * squares - linear + c
}

const A: [&[f64]; 2] = [&[1.0, -0.5, 0.3], &[0.2, 0.7, -1.1]];
const B: [f64; 2] = [0.4, -0.6];
const ETA: [f64; 3] = [0.1, -0.2, 0.5];
const C: f64 = -1.3;
const P: [f64; 2] = [0.3, -0.2];
const Q: f64 = 1.5;

skoro_core::precision_tests! {
    fn value_is_the_definition(target) {
        let relative = tolerance(&target);
        // Two rows over a Vector(2) and a Positive(1), D = 3, with a linear part.
        let mut writer = Writer::new();
        let (x, y, nodes) = writer
            .build(|| {
                let x = quantity("x", Vector::<2>);
                let y = quantity("y", Positive::<1>);
                let improper = Stacked::new()
                    .block(x, Vector::<2>, P)
                    .block(y, Positive::<1>, [Q])
                    .improper(2, by_columns(&A).as_slice(), &B[..], C)
                    .linear(&ETA[..]);
                (x, y, [improper.log_density().untyped(), improper.residual().untyped()])
            })
            .unwrap();
        let xs = [[0.5, 0.1], [-1.2, 0.7], [P[0], P[1]]];
        let ys = [[0.8], [2.4], [Q]];
        let given = Observations::new()
            .observe(x, Rows(xs.to_vec()))
            .observe(y, Rows(ys.to_vec()));
        let got = evaluate(&writer.snapshot(), &nodes, given, &target);
        let deltas: Vec<[f64; 3]> = xs
            .iter()
            .zip(&ys)
            .map(|(x, y)| [x[0] - P[0], x[1] - P[1], (y[0] / Q).ln()])
            .collect();
        let want: Vec<f64> = deltas.iter().map(|d| value(&A, &B, Some(&ETA), C, d)).collect();
        close(&got[0], &want, relative);
        let want: Vec<f64> = deltas.iter().flat_map(|d| residual(&A, &B, d)).collect();
        close(&got[1], &want, relative);

        // A square, non-symmetric A: given in the other order, the value would differ.
        let a: [&[f64]; 2] = [&[1.0, 2.0], &[-0.5, 3.0]];
        let mut writer = Writer::new();
        let (x, log_density) = writer
            .build(|| {
                let x = quantity("x", Vector::<2>);
                let improper = Stacked::new()
                    .block(x, Vector::<2>, P)
                    .improper(2, by_columns(&a).as_slice(), &B[..], C);
                (x, improper.log_density())
            })
            .unwrap();
        let given = Observations::new().observe(x, Rows(xs.to_vec()));
        let got = &evaluate(&writer.snapshot(), &[log_density.untyped()], given, &target)[0];
        let want: Vec<f64> = xs
            .iter()
            .map(|x| value(&a, &B, None, C, &[x[0] - P[0], x[1] - P[1]]))
            .collect();
        close(got, &want, relative);
    }

    fn rows_differ_across_a_batch(target) {
        let relative = tolerance(&target);
        let a: [[&[f64]; 1]; 3] = [[&[1.0, 2.0]], [&[-0.5, 0.25]], [&[0.0, 3.0]]];
        let b = [[0.1], [-0.4], [2.0]];
        let eta = [[0.3, 0.0], [-1.0, 0.5], [0.0, 0.0]];
        let c = [0.0, -2.0, 1.5];
        let xs = [[0.5, 0.1], [-1.2, 0.7], [2.0, -0.3]];
        let columns: Vec<Vec<f64>> = a.iter().map(|rows| by_columns(rows)).collect();
        let mut writer = Writer::new();
        let (x, log_density) = writer
            .build(|| {
                let x = quantity("x", Vector::<2>);
                let improper = Stacked::new()
                    .block(x, Vector::<2>, P)
                    .improper(
                        1,
                        Rows(columns.iter().map(Vec::as_slice).collect()),
                        Rows(b.iter().map(|b| &b[..]).collect()),
                        Rows(c.to_vec()),
                    )
                    .linear(Rows(eta.iter().map(|e| &e[..]).collect()));
                (x, improper.log_density())
            })
            .unwrap();
        let given = Observations::new().observe(x, Rows(xs.to_vec()));
        let got = &evaluate(&writer.snapshot(), &[log_density.untyped()], given, &target)[0];
        for (row, got) in got.iter().enumerate() {
            // The row's factor, built alone with its own numbers shared.
            let mut writer = Writer::new();
            let (x, alone) = writer
                .build(|| {
                    let x = quantity("x", Vector::<2>);
                    let improper = Stacked::new()
                        .block(x, Vector::<2>, P)
                        .improper(1, columns[row].as_slice(), &b[row][..], c[row])
                        .linear(&eta[row][..]);
                    (x, improper.log_density())
                })
                .unwrap();
            let given = Observations::new().observe(x, xs[row]);
            let want = &evaluate(&writer.snapshot(), &[alone.untyped()], given, &target)[0];
            close(&[*got], want, relative);
            let delta = [xs[row][0] - P[0], xs[row][1] - P[1]];
            close(&[*got], &[value(&a[row], &b[row], Some(&eta[row]), c[row], &delta)], relative);
        }

        // Rows of three beside rows of two pin different batches.
        let mut writer = Writer::new();
        let log_density = writer
            .build(|| {
                let x = quantity("x", Vector::<2>);
                Stacked::new()
                    .block(x, Vector::<2>, P)
                    .improper(
                        1,
                        Rows(columns.iter().map(Vec::as_slice).collect()),
                        Rows(b[..2].iter().map(|b| &b[..]).collect()),
                        0.0,
                    )
                    .log_density()
            })
            .unwrap();
        let error = writer
            .snapshot()
            .compile(&Query::of([log_density.untyped()]), &target)
            .unwrap_err();
        assert!(matches!(error, Error::BatchSize { .. }), "{error:?}");
    }
}

/// `AᵀA`, row by row, and `Aᵀv`, of `A` row by row.
fn normal_equations(a: &[&[f64]], v: &[f64]) -> (Vec<f64>, Vec<f64>) {
    let d = a[0].len();
    let gram = (0..d * d)
        .map(|k| a.iter().map(|row| row[k / d] * row[k % d]).sum())
        .collect();
    let projected = (0..d)
        .map(|j| a.iter().zip(v).map(|(row, v)| row[j] * v).sum())
        .collect();
    (gram, projected)
}

/// The log-density of the factor over `x` a `Vector(2)` and `y` a `Positive(1)` at the anchors
/// `p` and `q`, at the points `xs` and `ys`, on `target`.
fn over_vector_and_positive(
    build: impl Fn(Stacked) -> Improper,
    p: [f64; 2],
    q: f64,
    xs: &[[f64; 2]],
    ys: &[[f64; 1]],
    target: &Target,
) -> Vec<f64> {
    let mut writer = Writer::new();
    let (x, y, log_density) = writer
        .build(|| {
            let x = quantity("x", Vector::<2>);
            let y = quantity("y", Positive::<1>);
            let stacked = Stacked::new()
                .block(x, Vector::<2>, p)
                .block(y, Positive::<1>, [q]);
            (x, y, build(stacked).log_density())
        })
        .unwrap();
    let given = Observations::new()
        .observe(x, Rows(xs.to_vec()))
        .observe(y, Rows(ys.to_vec()));
    evaluate(&writer.snapshot(), &[log_density.untyped()], given, target).remove(0)
}

const XS: [[f64; 2]; 3] = [[0.5, 0.1], [-1.2, 0.7], [0.3, -0.2]];
const YS: [[f64; 1]; 3] = [[0.8], [2.4], [1.5]];

fn loaded(data: &[f64], rows: usize, storage: usize, target: &Target) -> Value {
    HostValue::new(Some(rows), storage, HostData::Continuous(data.to_vec()))
        .unwrap()
        .load(rows, target)
        .unwrap()
        .unwrap()
}

skoro_core::precision_tests! {
    fn full_rank_is_the_stacked_normal(target) {
        let relative = tolerance(&target);
        // Four rows of rank three over D = 3.
        let a: [&[f64]; 4] = [&[1.0, -0.5, 0.3], &[0.2, 0.7, -1.1], &[0.0, 0.4, 0.9], &[0.5, 0.5, 0.5]];
        let b = [0.4, -0.6, 0.2, 1.0];
        for eta in [None, Some([0.1, -0.2, 0.5])] {
            // m = (AᵀA)⁻¹ (Aᵀb - η), and c_N = c + ½ mᵀΛm - ½|b|² - ½ log det Λ + (D/2) log 2π.
            let (lambda, atb) = normal_equations(&a, &b);
            let rhs: Vec<f64> = (0..3).map(|j| atb[j] - eta.map_or(0.0, |e| e[j])).collect();
            let inverse = inverse(&lambda, 3);
            let m: Vec<f64> = (0..3).map(|i| (0..3).map(|j| inverse[i * 3 + j] * rhs[j]).sum()).collect();
            let quadratic: f64 = (0..3).map(|i| m[i] * (0..3).map(|j| lambda[i * 3 + j] * m[j]).sum::<f64>()).sum();
            let log_det: f64 = 2.0 * (0..3).map(|i| cholesky(&lambda, 3)[i * 4].ln()).sum::<f64>();
            let c_n = C + 0.5 * quadratic - 0.5 * b.iter().map(|b| b * b).sum::<f64>() - 0.5 * log_det
                + 1.5 * (2.0 * std::f64::consts::PI).ln();

            let improper = over_vector_and_positive(
                |stacked| {
                    let improper = stacked.improper(4, by_columns(&a).as_slice(), &b[..], C);
                    match eta {
                        Some(eta) => improper.linear(&eta[..]),
                        None => improper,
                    }
                },
                P, Q, &XS, &YS, &target,
            );
            let mut writer = Writer::new();
            let (x, y, normal) = writer
                .build(|| {
                    let x = quantity("x", Vector::<2>);
                    let y = quantity("y", Positive::<1>);
                    let normal = Stacked::new()
                        .block(x, Vector::<2>, P)
                        .block(y, Positive::<1>, [Q])
                        .log_density(&m, Scale::precision(&lambda, 3), c_n);
                    (x, y, normal)
                })
                .unwrap();
            let given = Observations::new()
                .observe(x, Rows(XS.to_vec()))
                .observe(y, Rows(YS.to_vec()));
            let normal = evaluate(&writer.snapshot(), &[normal.untyped()], given, &target).remove(0);
            close(&improper, &normal, relative);
        }
    }

    fn forms_give_one_function(target) {
        let relative = tolerance(&target);
        let base = over_vector_and_positive(
            |stacked| stacked.improper(2, by_columns(&A).as_slice(), &B[..], C).linear(&ETA[..]),
            P, Q, &XS, &YS, &target,
        );

        // 1. An orthogonal Q on [A b].
        let (cos, sin) = (0.6f64, 0.8f64);
        let rotated: Vec<Vec<f64>> = (0..2)
            .map(|i| (0..3).map(|j| if i == 0 { cos * A[0][j] + sin * A[1][j] } else { -sin * A[0][j] + cos * A[1][j] }).collect())
            .collect();
        let rotated_b = [cos * B[0] + sin * B[1], -sin * B[0] + cos * B[1]];
        let rows: Vec<&[f64]> = rotated.iter().map(Vec::as_slice).collect();
        let form = over_vector_and_positive(
            |stacked| stacked.improper(2, by_columns(&rows).as_slice(), &rotated_b[..], C).linear(&ETA[..]),
            P, Q, &XS, &YS, &target,
        );
        close(&form, &base, relative);

        // 2. A zero row appended, with its right-hand side e, and c raised by ½ e².
        let e = 0.7;
        let zero: &[f64] = &[0.0, 0.0, 0.0];
        let appended = [A[0], A[1], zero];
        let form = over_vector_and_positive(
            |stacked| {
                stacked
                    .improper(3, by_columns(&appended).as_slice(), &[B[0], B[1], e][..], C + 0.5 * e * e)
                    .linear(&ETA[..])
            },
            P, Q, &XS, &YS, &target,
        );
        close(&form, &base, relative);

        // 3. b - u, η - Aᵀu, and c - uᵀb + ½|u|².
        let u = [0.3, -1.1];
        let (_, atu) = normal_equations(&A, &u);
        let shifted_b = [B[0] - u[0], B[1] - u[1]];
        let shifted_eta: Vec<f64> = (0..3).map(|j| ETA[j] - atu[j]).collect();
        let shifted_c = C - (u[0] * B[0] + u[1] * B[1]) + 0.5 * (u[0] * u[0] + u[1] * u[1]);
        let form = over_vector_and_positive(
            |stacked| {
                stacked
                    .improper(2, by_columns(&A).as_slice(), &shifted_b[..], shifted_c)
                    .linear(shifted_eta.as_slice())
            },
            P, Q, &XS, &YS, &target,
        );
        close(&form, &base, relative);

        // The anchors moved by w on both flat-in-disguise quantities: b - Aw, and c - ηᵀw.
        let w = [0.25, -0.4, 0.6];
        let moved_b: Vec<f64> = residual(&A, &B, &w).iter().map(|r| -r).collect();
        let moved_c = C - ETA.iter().zip(&w).map(|(e, w)| e * w).sum::<f64>();
        let form = over_vector_and_positive(
            |stacked| stacked.improper(2, by_columns(&A).as_slice(), moved_b.as_slice(), moved_c).linear(&ETA[..]),
            [P[0] + w[0], P[1] + w[1]], Q * w[2].exp(), &XS, &YS, &target,
        );
        close(&form, &base, relative);
    }

    fn positive_example_is_exact(target) {
        let relative = tolerance(&target);
        // A positive x scored by the normal with mean e^(θ + φ) and scale σ, θ ~ N(0, 1), θ retired.
        let sigma: f64 = 0.6;
        let s = (1.0 + sigma * sigma).sqrt();
        let (phi_hat, x_hat): (f64, f64) = (0.4, 1.7);
        let a = [-1.0 / s, 1.0 / s];
        let b = -(x_hat.ln() - phi_hat) / s;
        let c = -x_hat.ln() - 0.5 * (2.0 * std::f64::consts::PI * s * s).ln();
        let mut writer = Writer::new();
        let (phi, x, log_density) = writer
            .build(|| {
                let phi = quantity("phi", Real);
                let x = quantity("x", Positive::<1>);
                let improper = Stacked::new()
                    .block(phi, Real, phi_hat)
                    .block(x, Positive::<1>, [x_hat])
                    .improper(1, &a[..], &[b][..], c)
                    .linear(&[0.0, 1.0][..]);
                (phi, x, improper.log_density())
            })
            .unwrap();
        let points = [(-0.3, 0.5), (0.4, 1.7), (1.2, 3.1), (2.0, 0.9)];
        let given = Observations::new()
            .observe(phi, Rows(points.iter().map(|p| p.0).collect()))
            .observe(x, Rows(points.iter().map(|p| [p.1]).collect()));
        let got = &evaluate(&writer.snapshot(), &[log_density.untyped()], given, &target)[0];
        let want: Vec<f64> = points
            .iter()
            .map(|&(phi, x): &(f64, f64)| {
                -x.ln() - (x.ln() - phi).powi(2) / (2.0 * s * s) - 0.5 * (2.0 * std::f64::consts::PI * s * s).ln()
            })
            .collect();
        close(got, &want, relative);
    }

    fn gauss_newton_is_the_hessian_on_flat_types(target) {
        let relative = tolerance(&target);
        let mut writer = Writer::new();
        let (x, y, log_density) = writer
            .build(|| {
                let x = quantity("x", Vector::<2>);
                let y = quantity("y", Positive::<1>);
                let improper = Stacked::new()
                    .block(x, Vector::<2>, P)
                    .block(y, Positive::<1>, [Q])
                    .improper(2, by_columns(&A).as_slice(), &B[..], C)
                    .linear(&ETA[..]);
                (x, y, improper.log_density())
            })
            .unwrap();
        let program = writer
            .snapshot()
            .compile(&Query::of([log_density.untyped()]), &target)
            .unwrap();
        let rows = XS.len();
        let xs: Vec<f64> = XS.iter().flatten().copied().collect();
        let ys: Vec<f64> = YS.iter().flatten().copied().collect();
        let at = Values::new()
            .with(x, loaded(&xs, rows, 2, &target))
            .with(y, loaded(&ys, rows, 1, &target));
        // In the logarithmic chart of y, D δ is the identity at every point.
        let v = [0.4, -0.7, 1.1, 0.2, 0.5, -0.3, -1.0, 0.8, 0.6];
        let vx: Vec<f64> = (0..rows).flat_map(|r| [v[3 * r], v[3 * r + 1]]).collect();
        let vy: Vec<f64> = (0..rows).map(|r| v[3 * r + 2]).collect();
        let second = program
            .hessian_products(
                &at,
                &Frames::new()
                    .with(x, Frame::from_host(&vx, rows, 2, 1, &target))
                    .with(y, Frame::from_host(&vy, rows, 1, 1, &target)),
            )
            .unwrap();
        let (gram, _) = normal_equations(&A, &B);
        let (mut gradient, mut hessian) = (Vec::new(), Vec::new());
        for r in 0..rows {
            let delta = [XS[r][0] - P[0], XS[r][1] - P[1], (YS[r][0] / Q).ln()];
            let (_, at_r) = normal_equations(&A, &residual(&A, &B, &delta));
            gradient.push((0..3).map(|j| -at_r[j] - ETA[j]).collect::<Vec<_>>());
            hessian.push((0..3).map(|i| -(0..3).map(|j| gram[i * 3 + j] * v[3 * r + j]).sum::<f64>()).collect::<Vec<_>>());
        }
        let split = |blocks: &[Vec<f64>]| -> (Vec<f64>, Vec<f64>) {
            (blocks.iter().flat_map(|b| [b[0], b[1]]).collect(), blocks.iter().map(|b| b[2]).collect())
        };
        let (gx, gy) = split(&gradient);
        let (hx, hy) = split(&hessian);
        close(&second.alpha.get(x).unwrap().to_host(), &gx, relative);
        close(&second.alpha.get(y).unwrap().to_host(), &gy, relative);
        close(&second.beta.get(x).unwrap().to_host(), &hx, relative);
        close(&second.beta.get(y).unwrap().to_host(), &hy, relative);
    }
}

skoro_core::precision_tests! {
    fn no_rows_is_its_linear_part(target) {
        let relative = tolerance(&target);
        let rows = XS.len();
        let xs: Vec<f64> = XS.iter().flatten().copied().collect();
        let ys: Vec<f64> = YS.iter().flatten().copied().collect();
        let v = [0.4, -0.7, 1.1, 0.2, 0.5, -0.3, -1.0, 0.8, 0.6];
        let vx: Vec<f64> = (0..rows).flat_map(|r| [v[3 * r], v[3 * r + 1]]).collect();
        let vy: Vec<f64> = (0..rows).map(|r| v[3 * r + 2]).collect();
        for eta in [None, Some(ETA)] {
            let mut writer = Writer::new();
            let (x, y, log_density, residual) = writer
                .build(|| {
                    let x = quantity("x", Vector::<2>);
                    let y = quantity("y", Positive::<1>);
                    let improper = Stacked::new()
                        .block(x, Vector::<2>, P)
                        .block(y, Positive::<1>, [Q])
                        .improper(0, &[] as &[f64], &[] as &[f64], C);
                    let improper = match eta {
                        Some(eta) => improper.linear(&eta[..]),
                        None => improper,
                    };
                    (x, y, improper.log_density(), improper.residual())
                })
                .unwrap();
            let at = Values::new()
                .with(x, loaded(&xs, rows, 2, &target))
                .with(y, loaded(&ys, rows, 1, &target));
            let frames = Frames::new()
                .with(x, Frame::from_host(&vx, rows, 2, 1, &target))
                .with(y, Frame::from_host(&vy, rows, 1, 1, &target));
            let eta = eta.unwrap_or([0.0; 3]);
            let deltas: Vec<[f64; 3]> = (0..rows)
                .map(|r| [XS[r][0] - P[0], XS[r][1] - P[1], (YS[r][0] / Q).ln()])
                .collect();
            let dot = |a: &[f64], b: &[f64]| a.iter().zip(b).map(|(a, b)| a * b).sum::<f64>();

            // Forward: -ηᵀδ + c, and a residual of no components.
            let program = writer
                .snapshot()
                .compile(&Query::of([log_density.untyped(), residual.untyped()]), &target)
                .unwrap();
            let out = program.forward(&at).unwrap();
            let want: Vec<f64> = deltas.iter().map(|d| C - dot(&eta, d)).collect();
            close(&common::host(&out, log_density.untyped().id()), &want, relative);
            assert!(common::host(&out, residual.untyped().id()).is_empty());

            // Tangent: -ηᵀv.
            let (_, tangents) = program.tangent(&at, &frames).unwrap();
            let want: Vec<f64> = (0..rows).map(|r| -dot(&eta, &v[3 * r..3 * r + 3])).collect();
            close(&tangents.get(log_density).unwrap().to_host(), &want, relative);
            assert!(tangents.get(residual).unwrap().to_host().is_empty());

            // Cotangent and Hessian: the gradient -η, and no curvature.
            let program = writer
                .snapshot()
                .compile(&Query::of([log_density.untyped()]), &target)
                .unwrap();
            let (_, gradient) = program.gradient(&at).unwrap();
            let gx: Vec<f64> = (0..rows).flat_map(|_| [-eta[0], -eta[1]]).collect();
            let gy: Vec<f64> = (0..rows).map(|_| -eta[2]).collect();
            close(&gradient.get(x).unwrap().to_host(), &gx, relative);
            close(&gradient.get(y).unwrap().to_host(), &gy, relative);
            let second = program.hessian_products(&at, &frames).unwrap();
            close(&second.alpha.get(x).unwrap().to_host(), &gx, relative);
            close(&second.beta.get(x).unwrap().to_host(), &vec![0.0; 2 * rows], relative);
            close(&second.beta.get(y).unwrap().to_host(), &vec![0.0; rows], relative);
        }
    }
}

#[test]
fn numbers_that_do_not_fit_are_refused() {
    let line = line!() + 6;
    let failure = Writer::new()
        .build(|| {
            let x = quantity("x", Vector::<2>);
            Stacked::new()
                .block(x, Vector::<2>, P)
                .improper(2, &[1.0, 2.0, 3.0][..], &B[..], C);
        })
        .unwrap_err();
    assert!(
        matches!(failure.error, Error::ValueShape { .. }),
        "{:?}",
        failure.error
    );
    assert_eq!(failure.location.line(), line);
}

#[test]
fn roles_are_bound() {
    let mut writer = Writer::new();
    let (plain, linear, improper) = writer
        .build(|| {
            let x = quantity("x", Vector::<2>);
            let y = quantity("y", Positive::<1>);
            let stacked = Stacked::new()
                .block(x, Vector::<2>, P)
                .block(y, Positive::<1>, [Q]);
            let improper = stacked.improper(2, by_columns(&A).as_slice(), &B[..], C);
            let plain = improper.factor();
            let with_linear = improper.linear(&ETA[..]);
            let linear = with_linear.factor();
            (plain.id(), linear.id(), (improper, with_linear))
        })
        .unwrap();
    let snapshot = writer.snapshot();
    let at_most = |term: NodeId, role: &Role<AtMostOne>| snapshot.bound(term, role).unwrap();
    let residual = |term: NodeId| snapshot.bound(term, &roles::RESIDUAL).unwrap();
    let (without, with): (Improper, Improper) = improper;

    assert_eq!(residual(plain).id(), without.residual().untyped().id());
    assert!(at_most(plain, &roles::LOG_VOLUME).is_none());
    assert_eq!(residual(linear).id(), with.residual().untyped().id());
    let bound = at_most(linear, &roles::LOG_VOLUME).expect("the linear part is bound");
    assert_eq!(bound.id(), with.linear_part().unwrap().untyped().id());
    for term in [plain, linear] {
        assert!(snapshot.bound(term, &roles::MEAN).is_err());
        assert!(at_most(term, &roles::INFORMATION_ROOT).is_none());
        for form in [
            Form::Isotropic,
            Form::Diagonal,
            Form::CovarianceFactor,
            Form::Whitening,
        ] {
            assert!(at_most(term, roles::scale(form)).is_none());
        }
    }
}

#[test]
fn no_blocks_is_refused() {
    let line = line!() + 3;
    let failure = Writer::new()
        .build(|| {
            Stacked::new().improper(0, &[] as &[f64], &[] as &[f64], C);
        })
        .unwrap_err();
    assert!(
        matches!(
            failure.error,
            Error::SpaceMismatch {
                operation: "skoro.concatenate",
                ..
            }
        ),
        "{:?}",
        failure.error
    );
    assert_eq!(failure.location.line(), line);
    let _ = Target::host(); // shared with the tests at both precisions
}

skoro_core::precision_tests! {
    fn block_in_is_block(target) {
        // The same factor over blocks stacked by marker and by space, shared and per-row anchors.
        let (a, b, eta, c): (&[&[f64]], [f64; 2], [f64; 3], f64) =
            (&[&[1.0, 0.5, -0.3], &[0.0, 2.0, 0.7]], [0.2, -0.4], [0.3, 0.0, -0.6], 0.25);
        let (p, q) = ([0.4, -0.1], 1.3);
        let build = |stacked: Stacked| {
            stacked
                .improper(2, by_columns(a).as_slice(), b.as_slice(), c)
                .linear(eta.as_slice())
        };
        let typed = over_vector_and_positive(build, p, q, &XS, &YS, &target);
        let shared = |data: &[f64]| HostValue::new(None, data.len(), HostData::Continuous(data.to_vec())).unwrap();
        let per_row = |data: &[f64], storage: usize| {
            let rows = data.len() / storage;
            HostValue::new(Some(rows), storage, HostData::Continuous(data.to_vec())).unwrap()
        };
        for anchors in [
            (shared(&p), shared(&[q])),
            (per_row(&[p, p, p].concat(), 2), per_row(&[q, q, q], 1)),
        ] {
            let mut writer = Writer::new();
            let (x, y, log_density) = writer
                .build(|| {
                    let x = quantity("x", Vector::<2>);
                    let y = quantity("y", Positive::<1>);
                    let stacked = Stacked::new()
                        .block_in(x.untyped(), &skoro_core::Space::Vector(2), anchors.0.clone())
                        .block_in(y.untyped(), &skoro_core::Space::Positive(1), anchors.1.clone());
                    (x, y, build(stacked).log_density())
                })
                .unwrap();
            let given = Observations::new()
                .observe(x, Rows(XS.to_vec()))
                .observe(y, Rows(YS.to_vec()));
            let dynamic =
                evaluate(&writer.snapshot(), &[log_density.untyped()], given, &target).remove(0);
            close(&dynamic, &typed, tolerance(&target));
        }
        // An anchor that does not fit its space.
        let mut writer = Writer::new();
        let failure = writer
            .build(|| {
                let x = quantity("x", Vector::<2>);
                Stacked::new()
                    .block_in(x.untyped(), &skoro_core::Space::Vector(2), shared(&[1.0, 2.0, 3.0]))
                    .tangent()
            })
            .unwrap_err();
        assert!(matches!(failure.error, Error::ValueShape { .. }));
    }
}
