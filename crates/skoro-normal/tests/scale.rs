#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)] // a test crate
//! The scale's constant helpers: the factors they hold, and what they refuse.

use skoro_core::{Error, NodeId, Snapshot, Writer};
use skoro_normal::{Form, Scale, triangular};
use skoro_tensor::HostData;

/// The host data of the folded node `id`.
fn data(snapshot: &Snapshot, id: NodeId) -> Vec<f64> {
    let value = snapshot.node(id).and_then(|node| node.value()).unwrap();
    match value.data() {
        HostData::Continuous(data) => data.clone(),
        HostData::Discrete(_) => panic!("a continuous value"),
    }
}

/// The dense `d × d` matrix, row by row, of a factor in `triangular(d)`'s storage.
fn dense(storage: &[f64], d: usize) -> Vec<f64> {
    let mut matrix = vec![0.0; d * d];
    for a in 0..d {
        matrix[a * d + a] = storage[a];
    }
    let mut k = d;
    for b in 0..d {
        for a in b + 1..d {
            matrix[a * d + b] = storage[k];
            k += 1;
        }
    }
    matrix
}

/// `AᵀA` or `AAᵀ`, row by row.
fn gram(a: &[f64], d: usize, transposed_first: bool) -> Vec<f64> {
    let at = |i: usize, j: usize| a[i * d + j];
    (0..d * d)
        .map(|k| {
            let (i, j) = (k / d, k % d);
            (0..d)
                .map(|m| {
                    if transposed_first {
                        at(m, i) * at(m, j)
                    } else {
                        at(i, m) * at(j, m)
                    }
                })
                .sum()
        })
        .collect()
}

fn close(a: &[f64], b: &[f64]) {
    for (x, y) in a.iter().zip(b) {
        assert!(
            (x - y).abs() < 1e-12 * y.abs().max(1.0),
            "{a:?} against {b:?}"
        );
    }
}

const SPD: [f64; 9] = [4.0, 1.2, -0.6, 1.2, 3.0, 0.5, -0.6, 0.5, 2.0];

#[test]
fn constant_scale_helpers() {
    let d = 3;
    let mut writer = Writer::new();
    let (covariance, precision, sd, sds) = writer
        .build(|| {
            (
                Scale::covariance(&SPD, d),
                Scale::precision(&SPD, d),
                Scale::sd(0.3),
                Scale::sds(&[0.1, 0.2, 0.3]),
            )
        })
        .unwrap();
    let snapshot = writer.snapshot();
    assert_eq!(covariance.form(), Form::CovarianceFactor);
    assert_eq!(precision.form(), Form::Whitening);
    assert_eq!(snapshot.space(covariance.value()), Some(&triangular(d)));
    assert_eq!(snapshot.space(precision.value()), Some(&triangular(d)));

    // L Lᵀ = Σ, and Wᵀ W = Λ, each lower triangular with a positive diagonal.
    let l = dense(&data(&snapshot, covariance.value().id()), d);
    close(&gram(&l, d, false), &SPD);
    let w = dense(&data(&snapshot, precision.value().id()), d);
    close(&gram(&w, d, true), &SPD);
    for factor in [&l, &w] {
        assert!((0..d).all(|a| factor[a * d + a] > 0.0));
    }
    assert_eq!(data(&snapshot, sd.value().id()), [0.3]);
    assert_eq!(data(&snapshot, sds.value().id()), [0.1, 0.2, 0.3]);

    // A side of one is a positive scalar.
    assert_eq!(
        triangular(1),
        skoro_core::Space::product([skoro_core::Space::Positive(1)]).unwrap()
    );

    // A matrix that is not symmetric positive definite, or not square, is refused.
    let indefinite = [1.0, 2.0, 2.0, 1.0];
    let asymmetric = [2.0, 0.5, 0.4, 2.0];
    let refusals: [(&[f64], usize, bool); 4] = [
        (&indefinite, 2, true),
        (&indefinite, 2, false),
        (&asymmetric, 2, true),
        (&SPD, 2, true),
    ];
    for (matrix, d, as_covariance) in refusals {
        let failure = Writer::new()
            .build(|| {
                if as_covariance {
                    Scale::covariance(matrix, d)
                } else {
                    Scale::precision(matrix, d)
                }
            })
            .unwrap_err();
        assert_eq!(
            failure.error,
            Error::ValueShape {
                space: triangular(d)
            }
        );
    }
}
