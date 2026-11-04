#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)] // a test crate
//! The maps of the rotations, against their references on the host.

mod common;

use common::host::{M3, cross, mm, mv, qmul, rot, stored, transpose};
use common::{CUTOFF, Rows, close, quaternions, random_quaternions, tolerance, vectors};
use skoro_core::Real;
use skoro_core::{Matrix, Q, QDyn, Space, Vector, VectorDyn};
use skoro_spatial::{Axis, ROTATION, Rotation, RotationMaps, host, rotation, so3};

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

/// Rotation vectors in random directions at lengths from 0 to 6, at both sides of the cutoff.
fn rotation_vectors(seed: u64) -> Vec<[f64; 3]> {
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
        .map(|(d, length)| common::unit(d).map(|v| v * length))
        .collect()
}

/// Unit quaternions whose largest coordinate in magnitude is the `m`-th, with either sign.
fn led_by(m: usize) -> Vec<[f64; 4]> {
    let mut rows = Vec::new();
    for (k, q) in random_quaternions(6, 100 + m as u64)
        .into_iter()
        .enumerate()
    {
        let mut q = q.map(|v| 0.4 * v);
        q[m] = if k % 2 == 0 { 1.0 } else { -1.0 };
        rows.push(common::unit(q));
    }
    rows
}

skoro_core::precision_tests! {
    fn from_matrix_branches(target) {
        let tol = tolerance(&target);
        for m in 0..4 {
            let qs = led_by(m);
            let out = common::evaluate(
                |q| rotation::from_matrix(r(q[0]).matrix()).untyped(),
                &[rotations(&qs)],
                &target,
            );
            // ±q, with the m-th coordinate positive.
            let want: Vec<[f64; 4]> = qs
                .iter()
                .map(|q| if q[m] > 0.0 { *q } else { q.map(|v| -v) })
                .collect();
            close(&out, &want.concat(), tol);
            let on_host: Vec<f64> = qs.iter().flat_map(|q| host::from_matrix(host::matrix(*q))).collect();
            close(&on_host, &want.concat(), 1e-14);
        }
        // The storage order: a turn about z alone, given by rows.
        let theta = 0.7f64;
        let (s, c) = theta.sin_cos();
        let rz = [[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]];
        let out = common::evaluate(
            |q| rotation::from_matrix(q[0].narrow::<Matrix<3, 3>>()).untyped(),
            &[Rows::new(Space::Matrix(3, 3), stored(rz))],
            &target,
        );
        let want = [(theta / 2.0).cos(), 0.0, 0.0, (theta / 2.0).sin()];
        close(&out, &want, tol);
        close(&host::from_matrix(rz), &want, 1e-15);
    }

    fn constructions(target) {
        let tol = tolerance(&target);
        let rows = 6;
        let angles: Vec<f64> = vectors::<1>(rows, 71, -6.0, 6.0).concat();
        let axes = vectors::<3>(rows, 72, -2.0, 2.0);
        let out = common::evaluate_many(
            |q| {
                let (axis, angle) = (q[0].narrow::<Vector<3>>(), q[1].narrow::<Real>());
                let about = rotation::about(axis, angle);
                let (a, theta) = about.axis_angle();
                vec![
                    rotation::elementary(Axis::X, angle).untyped(),
                    rotation::elementary(Axis::Y, angle).untyped(),
                    rotation::elementary(Axis::Z, angle).untyped(),
                    about.untyped(),
                    rotation::about(a, theta).untyped(),
                    theta.untyped(),
                ]
            },
            &[Rows::vectors(&axes), Rows::new(Space::Real, angles.clone())],
            &target,
        );
        for (k, e) in [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]].iter().enumerate() {
            let want = flat(angles.iter().map(|t| common::about(*e, *t)));
            close(&out[k], &want, tol);
        }
        let want = flat(angles.iter().zip(&axes).map(|(t, a)| common::about(*a, *t)));
        close(&out[3], &want, tol);
        // axis_angle inverts about, its angle in [0, 2π).
        close(&out[4], &out[3], tol);
        assert!(out[5].iter().all(|t| (0.0..2.0 * std::f64::consts::PI).contains(t)));

        // At the identity: the fixed axis, the angle zero, and finite.
        let out = common::evaluate_many(
            |q| {
                let (a, theta) = r(q[0]).axis_angle();
                vec![a.untyped(), theta.untyped()]
            },
            &[rotations(&[[1.0, 0.0, 0.0, 0.0]])],
            &target,
        );
        close(&out[0], &[1.0, 0.0, 0.0], tol);
        close(&out[1], &[0.0], tol);

        // From two directions: π(q) a/|a| = b/|b|; non-finite for opposite ones.
        let (a, b) = (vectors::<3>(rows, 73, -2.0, 2.0), vectors::<3>(rows, 74, -2.0, 2.0));
        let out = common::evaluate(
            |q| {
                let (a, b) = (q[0].narrow::<Vector<3>>(), q[1].narrow::<Vector<3>>());
                rotation::from_two_vectors(a, b).untyped()
            },
            &[Rows::vectors(&a), Rows::vectors(&b)],
            &target,
        );
        for (row, q) in split::<4>(&out).iter().enumerate() {
            let carried = mv(rot(*q), common::unit(a[row]));
            close(&carried, &common::unit(b[row]), tol);
        }
        // Opposite directions whose norms are exact, so that the argument vanishes exactly
        // rather than by rounding.
        let a = [[1.0, 0.0, 0.0], [0.0, 3.0, 4.0], [0.0, 0.0, -2.0]];
        let opposite: Vec<[f64; 3]> = a.iter().map(|a| a.map(|v| -2.0 * v)).collect();
        let out = common::evaluate(
            |q| {
                let (a, b) = (q[0].narrow::<Vector<3>>(), q[1].narrow::<Vector<3>>());
                rotation::from_two_vectors(a, b).untyped()
            },
            &[Rows::vectors(&a), Rows::vectors(&opposite)],
            &target,
        );
        assert!(out.iter().all(|v| !v.is_finite()), "{out:?}");
    }

    fn alignment(target) {
        let tol = tolerance(&target);
        let z = quaternions(5, 81);
        let reference = random_quaternions(z.len(), 82);
        let out = common::evaluate_many(
            |q| {
                let (z, reference) = (r(q[0]), r(q[1]));
                let aligned = z.aligned_with(reference);
                vec![aligned.untyped(), aligned.matrix().untyped()]
            },
            &[rotations(&z), rotations(&reference)],
            &target,
        );
        close(&out[1], &matrices(&z), tol);
        for (row, aligned) in split::<4>(&out[0]).iter().enumerate() {
            let (z, reference) = (z[row], reference[row]);
            let inner = common::host::dot(&z, &reference);
            let sign = if inner >= 0.0 { 1.0 } else { -1.0 };
            close(aligned, &z.map(|v| sign * v), tol);
            assert!(common::host::dot(aligned, &reference) >= 0.0);
            close(&host::align(z, reference), &z.map(|v| sign * v), 1e-15);
            close(&host::align(z.map(|v| 3.0 * v), reference), &z.map(|v| sign * v), 1e-15);
        }
    }

    fn exp_and_log(target) {
        let tol = tolerance(&target);
        let omega = rotation_vectors(61);
        let out = common::evaluate_many(
            |q| {
                let omega = q[0].narrow::<Vector<3>>();
                let e = rotation::exp(omega);
                vec![e.untyped(), e.log().untyped(), e.matrix().untyped()]
            },
            &[Rows::vectors(&omega)],
            &target,
        );
        let exp = flat(omega.iter().map(|w| {
            let theta = common::host::norm(w);
            let b = if theta == 0.0 { 0.5 } else { (theta / 2.0).sin() / theta };
            [(theta / 2.0).cos(), b * w[0], b * w[1], b * w[2]]
        }));
        close(&out[0], &exp, tol);
        let on_host: Vec<f64> = omega.iter().flat_map(|w| host::exp(*w)).collect();
        close(&on_host, &exp, 1e-14);
        let back: Vec<f64> = split::<4>(&on_host).iter().flat_map(|q| host::log(*q)).collect();
        close(&back, &omega.concat(), 1e-12);
        assert!(host::log([-1.0, 0.0, 0.0, 0.0]).iter().all(|v| !v.is_finite()));
        // Log inverts Exp on the ball of radius 2π.
        close(&out[1], &omega.concat(), tol);
        // Exp at a constant rate is the matrix exponential of [ω]×.
        let integrated: Vec<f64> = omega
            .iter()
            .flat_map(|w| {
                let rows: Vec<Vec<f64>> = common::host::hat(*w).iter().map(|r| r.to_vec()).collect();
                let e = common::host::expm(&rows);
                stored::<3, 3>(std::array::from_fn(|i| std::array::from_fn(|j| e[i][j])))
            })
            .collect();
        close(&out[2], &integrated, tol);

        // The shorter turn: the same at ±q, Log where w > 0, of length at most π.
        let a = quaternions(5, 62);
        let minus: Vec<[f64; 4]> = a.iter().map(|q| q.map(|v| -v)).collect();
        let out = common::evaluate_many(
            |q| {
                let (a, minus) = (r(q[0]), r(q[1]));
                vec![
                    a.log_shorter().untyped(),
                    minus.log_shorter().untyped(),
                    a.log().untyped(),
                ]
            },
            &[rotations(&a), rotations(&minus)],
            &target,
        );
        close(&out[0], &out[1], tol);
        for (row, q) in a.iter().enumerate() {
            let shorter = &out[0][3 * row..3 * row + 3];
            assert!(common::host::norm(shorter) <= std::f64::consts::PI * (1.0 + tol));
            if q[0] > 0.0 {
                close(shorter, &out[2][3 * row..3 * row + 3], tol);
            }
        }
    }

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
