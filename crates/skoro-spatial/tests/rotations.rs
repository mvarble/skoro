#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)] // a test crate
//! The maps of the rotations, against their references on the host.

mod common;

use common::host::{M3, cross, mm, mv, qmul, rot, stored, transpose};
use common::{Rows, close, quaternions, random_quaternions, tolerance, vectors};
use skoro_core::{Matrix, Q, QDyn, Space, Vector, VectorDyn};
use skoro_spatial::{ROTATION, Rotation, RotationMaps, rotation, so3};

fn rotations(rows: &[[f64; 4]]) -> Rows {
    Rows::new(Space::Embedded(ROTATION.clone()), rows.concat())
}

fn r(q: QDyn) -> Q<Rotation> {
    q.narrow()
}

/// The quaternions of a flat output, one per row.
fn split<const N: usize>(data: &[f64]) -> Vec<[f64; N]> {
    data.as_chunks::<N>().0.to_vec()
}

fn flat<const N: usize>(rows: impl IntoIterator<Item = [f64; N]>) -> Vec<f64> {
    rows.into_iter().flatten().collect()
}

/// Rotation matrices stored, one per row.
fn matrices(qs: &[[f64; 4]]) -> Vec<f64> {
    qs.iter().flat_map(|q| stored(rot(*q))).collect()
}

/// A stored `Matrix(3, 3)` by rows.
fn by_rows(data: &[f64]) -> M3 {
    std::array::from_fn(|i| std::array::from_fn(|j| data[i + 3 * j]))
}

skoro_core::precision_tests! {
    fn group_laws(target) {
        let a = quaternions(5, 21);
        let rows = a.len();
        let (b, c) = (random_quaternions(rows, 22), random_quaternions(rows, 23));
        let minus: Vec<[f64; 4]> = a.iter().map(|q| q.map(|v| -v)).collect();
        let out = common::evaluate_many(
            |q| {
                let (a, b, c, minus) = (r(q[0]), r(q[1]), r(q[2]), r(q[3]));
                let one = rotation::identity();
                vec![
                    (a * b).untyped(),
                    ((a * b) * c).untyped(),
                    (a * (b * c)).untyped(),
                    (a * one).untyped(),
                    (one * a).untyped(),
                    (a * a.inverse()).untyped(),
                    a.between(b).untyped(),
                    (a.inverse() * b).untyped(),
                    (a * b).matrix().untyped(),
                    a.matrix().untyped(),
                    b.matrix().untyped(),
                    minus.matrix().untyped(),
                ]
            },
            &[rotations(&a), rotations(&b), rotations(&c), rotations(&minus)],
            &target,
        );
        let tol = tolerance(&target);
        let products = flat(a.iter().zip(&b).map(|(a, b)| qmul(*a, *b)));
        close(&out[0], &products, tol);
        close(&out[1], &out[2], tol);
        close(&out[3], &a.concat(), tol);
        close(&out[4], &a.concat(), tol);
        close(&out[5], &flat((0..rows).map(|_| [1.0, 0.0, 0.0, 0.0])), tol);
        close(&out[6], &out[7], tol);
        let between = flat(
            a.iter()
                .zip(&b)
                .map(|(a, b)| qmul([a[0], -a[1], -a[2], -a[3]], *b)),
        );
        close(&out[6], &between, tol);
        // π is a homomorphism, and the same at q and -q.
        close(&out[9], &matrices(&a), tol);
        close(&out[11], &out[9], tol);
        let product: Vec<f64> = split::<9>(&out[9])
            .iter()
            .zip(split::<9>(&out[10]))
            .flat_map(|(ma, mb)| stored(mm(by_rows(ma), by_rows(&mb))))
            .collect();
        close(&out[8], &product, tol);
    }

    fn actions_are_the_matrix(target) {
        let a = quaternions(5, 31);
        let rows = a.len();
        let b = random_quaternions(rows, 32);
        let x = vectors::<3>(rows, 33, -2.0, 2.0);
        let columns = vectors::<12>(rows, 34, -2.0, 2.0);
        let out = common::evaluate_many(
            |q| {
                let (a, b) = (r(q[0]), r(q[1]));
                let x: Q<Vector<3>> = q[2].narrow();
                let m: Q<Matrix<3, 4>> = q[3].narrow();
                vec![
                    (a * x).untyped(),
                    a.unrotate(x).untyped(),
                    (a * x.widen()).untyped(),
                    (a * m).untyped(),
                    a.matrix().untyped(),
                    a.adjoint().untyped(),
                    (a * b).matrix().untyped(),
                    a.matrix().matmul(b.matrix()).untyped(),
                    a.unrotate(a * x).untyped(),
                ]
            },
            &[
                rotations(&a),
                rotations(&b),
                Rows::vectors(&x),
                Rows::new(Space::Matrix(3, 4), columns.concat()),
            ],
            &target,
        );
        let tol = tolerance(&target);
        let rotated = flat(a.iter().zip(&x).map(|(a, x)| mv(rot(*a), *x)));
        close(&out[0], &rotated, tol);
        let back = flat(a.iter().zip(&x).map(|(a, x)| mv(transpose(rot(*a)), *x)));
        close(&out[1], &back, tol);
        close(&out[2], &rotated, tol);
        // Each column of the 3 × 4 matrix rotated: column j is entries 3j..3j + 3.
        let rotated_columns = flat(a.iter().zip(&columns).map(|(a, m)| {
            let mut out = [0.0; 12];
            for j in 0..4 {
                let column = mv(rot(*a), [m[3 * j], m[3 * j + 1], m[3 * j + 2]]);
                out[3 * j..3 * j + 3].copy_from_slice(&column);
            }
            out
        }));
        close(&out[3], &rotated_columns, tol);
        close(&out[4], &matrices(&a), tol);
        close(&out[5], &out[4], tol);
        close(&out[6], &out[7], tol);
        close(&out[8], &x.concat(), tol);
    }

    fn conjugation_is_exact(target) {
        // q ⊕ t = (1 ⊕ π(q) t) ⊗ q, exactly.
        let a = quaternions(5, 41);
        let rows = a.len();
        let directions = vectors::<3>(rows, 42, -1.0, 1.0);
        for length in [0.1, 1.0, 10.0] {
            let t: Vec<[f64; 3]> = directions
                .iter()
                .map(|d| common::unit(*d).map(|v| v * length))
                .collect();
            let out = common::evaluate_many(
                |q| {
                    let (a, t) = (r(q[0]), q[1].narrow::<Vector<3>>());
                    let left = rotation::identity().retract(a.adjoint().matmul(t)) * a;
                    vec![a.retract(t).untyped(), left.untyped()]
                },
                &[rotations(&a), Rows::vectors(&t)],
                &target,
            );
            close(&out[0], &out[1], tolerance(&target));
        }
    }

    fn hat_and_vee(target) {
        let omega = vectors::<3>(6, 51, -2.0, 2.0);
        let x = vectors::<3>(6, 52, -2.0, 2.0);
        let m = vectors::<9>(6, 53, -2.0, 2.0);
        let out = common::evaluate_many(
            |q| {
                let (omega, x) = (q[0].narrow::<Vector<3>>(), q[1].narrow::<Vector<3>>());
                let m: Q<Matrix<3, 3>> = q[2].narrow();
                vec![
                    so3::hat(omega).matmul(x).untyped(),
                    so3::vee(so3::hat(omega)).untyped(),
                    so3::vee(m).untyped(),
                ]
            },
            &[Rows::vectors(&omega), Rows::vectors(&x), Rows::new(Space::Matrix(3, 3), m.concat())],
            &target,
        );
        let tol = tolerance(&target);
        close(&out[0], &flat(omega.iter().zip(&x).map(|(w, x)| cross(*w, *x))), tol);
        close(&out[1], &omega.concat(), tol);
        let skew = flat(m.iter().map(|m| {
            let m = by_rows(m);
            [
                0.5 * (m[2][1] - m[1][2]),
                0.5 * (m[0][2] - m[2][0]),
                0.5 * (m[1][0] - m[0][1]),
            ]
        }));
        close(&out[2], &skew, tol);
    }
}

#[test]
fn a_vector_of_another_length_is_refused() {
    let mut writer = skoro_core::Writer::new();
    let built = writer.build(|| {
        let q = skoro_core::scope::quantity("q", Rotation);
        let x = skoro_core::scope::quantity("x", VectorDyn(4));
        q * x
    });
    let failure = built.expect_err("a vector of length 4 is refused");
    assert!(
        failure.to_string().contains("skoro.spatial.rotate"),
        "{failure}"
    );
}
