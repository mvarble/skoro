#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)] // a test crate
//! The improper Gaussian factor: its value, its roles, its batches and its refusals, and the
//! statements and examples of its mathematics.

mod common;

use common::{close, evaluate, tolerance};
use skoro_core::scope::quantity;
use skoro_core::{
    AtMostOne, Error, NodeId, Observations, Positive, Query, Role, Rows, Vector, Writer,
};
use skoro_normal::{Form, Improper, Stacked, roles};
use skoro_tensor::Target;

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
