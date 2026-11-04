#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)] // a test crate
//! The maps of the rigid motions and the extended poses, against their homogeneous matrices.

mod common;

use common::host::{dense_mul, dense_stored, homogeneous, motion_inverse, mv, rot, transpose};
use common::{Rows, close, quaternions, tolerance, twists, vectors, with_translations};
use skoro_core::{Declares, Q, QDyn, Space, Vector, VectorDyn};
use skoro_spatial::{
    ExtendedPose, ExtendedPoseMaps, Pose, PoseMaps, Rotation, extended_pose, pose,
};

fn motions(data: &[f64], k: usize) -> Rows {
    let space = if k == 1 {
        Pose.space()
    } else {
        ExtendedPose.space()
    };
    Rows::new(space, data.to_vec())
}

fn p(q: QDyn) -> Q<Pose> {
    q.narrow()
}

fn e(q: QDyn) -> Q<ExtendedPose> {
    q.narrow()
}

/// The rows of a flat output, `n` values each.
fn rows_of(data: &[f64], n: usize) -> Vec<&[f64]> {
    data.chunks(n).collect()
}

/// Matrices stored, one per row, from storage by `f`.
fn stored_rows(data: &[f64], n: usize, f: impl Fn(&[f64]) -> Vec<Vec<f64>>) -> Vec<f64> {
    rows_of(data, n)
        .into_iter()
        .flat_map(|x| dense_stored(&f(x)))
        .collect()
}

/// The twist matrix of `(φ, t₁, …, t_k)`, by rows: `[φ]×` and the `tᵢ` over zeros.
fn twist_matrix(xi: &[f64], k: usize) -> Vec<Vec<f64>> {
    let n = 3 + k;
    let hat = common::host::hat([xi[0], xi[1], xi[2]]);
    let mut m = vec![vec![0.0; n]; n];
    for i in 0..3 {
        m[i][..3].copy_from_slice(&hat[i]);
        for t in 0..k {
            m[i][3 + t] = xi[3 + 3 * t + i];
        }
    }
    m
}

skoro_core::precision_tests! {
    fn motion_exp_and_log(target) {
        let tol = tolerance(&target);
        for k in [1, 2] {
            let (n, d) = (4 + 3 * k, 3 + 3 * k);
            let xi = twists(k, 221);
            let out = common::evaluate_many(
                |q| {
                    if k == 1 {
                        let t = pose::exp(q[0].narrow());
                        vec![t.matrix().untyped(), t.log().untyped()]
                    } else {
                        let t = extended_pose::exp(q[0].narrow());
                        vec![t.matrix().untyped(), t.log().untyped()]
                    }
                },
                &[Rows::new(Space::Vector(d), xi.clone())],
                &target,
            );
            // The motion at a constant twist is the matrix exponential of the twist.
            let integrated: Vec<f64> = xi
                .chunks(d)
                .flat_map(|xi| dense_stored(&common::host::expm(&twist_matrix(xi, k))))
                .collect();
            close(&out[0], &integrated, tol);
            // Its logarithm inverts it on the ball of radius 2π.
            close(&out[1], &xi, tol);

            // The shorter turn is the same at ±q.
            let x = with_translations(&quaternions(5, 222), k, 223);
            let minus: Vec<f64> = x
                .chunks(n)
                .flat_map(|x| x.iter().enumerate().map(|(i, v)| if i < 4 { -v } else { *v }).collect::<Vec<_>>())
                .collect();
            let out = common::evaluate_many(
                |q| {
                    if k == 1 {
                        vec![p(q[0]).log_shorter().untyped(), p(q[1]).log_shorter().untyped()]
                    } else {
                        vec![e(q[0]).log_shorter().untyped(), e(q[1]).log_shorter().untyped()]
                    }
                },
                &[motions(&x, k), motions(&minus, k)],
                &target,
            );
            close(&out[0], &out[1], tol);
        }
    }

    fn adjoints_conjugate(target) {
        let tol = tolerance(&target);
        for k in [1, 2] {
            let (n, d) = (4 + 3 * k, 3 + 3 * k);
            let t = with_translations(&quaternions(5, 231), k, 232);
            let rows = t.len() / n;
            let xi: Vec<f64> = vectors::<9>(rows, 233, -1.0, 1.0)
                .iter()
                .flat_map(|v| v[..d].to_vec())
                .collect();
            let h = 1e-3;
            let small: Vec<f64> = xi.iter().map(|v| h * v).collect();
            let out = common::evaluate_many(
                |q| {
                    let (xi, small) = (q[1].narrow::<VectorDyn>(), q[2].narrow::<VectorDyn>());
                    if k == 1 {
                        let t = p(q[0]);
                        let ad: Q<Vector<6>> = t.adjoint().matmul(xi.narrow::<Vector<6>>());
                        let moved: Q<Vector<6>> = t.adjoint().matmul(small.narrow::<Vector<6>>());
                        vec![
                            (t * pose::exp(xi.narrow()) * t.inverse()).untyped(),
                            pose::exp(ad).untyped(),
                            t.retract(small).untyped(),
                            (pose::identity().retract(moved) * t).untyped(),
                        ]
                    } else {
                        let t = e(q[0]);
                        let ad: Q<Vector<9>> = t.adjoint().matmul(xi.narrow::<Vector<9>>());
                        let moved: Q<Vector<9>> = t.adjoint().matmul(small.narrow::<Vector<9>>());
                        vec![
                            (t * extended_pose::exp(xi.narrow()) * t.inverse()).untyped(),
                            extended_pose::exp(ad).untyped(),
                            t.retract(small).untyped(),
                            (extended_pose::identity().retract(moved) * t).untyped(),
                        ]
                    }
                },
                &[
                    motions(&t, k),
                    Rows::new(Space::Vector(d), xi.clone()),
                    Rows::new(Space::Vector(d), small),
                ],
                &target,
            );
            // T Exp(ξ) T⁻¹ = Exp(Ad(T) ξ), exactly.
            close(&out[0], &out[1], tol);
            // T ⊕ hξ = (1 ⊕ h Ad(T) ξ) T + O(h²).
            close(&out[2], &out[3], tol.max(20.0 * h * h));
        }
    }

    fn motion_group_laws(target) {
        let tol = tolerance(&target);
        for k in [1, 2] {
            let n = 4 + 3 * k;
            let a = with_translations(&quaternions(5, 201), k, 202);
            let rows = a.len() / n;
            let b = with_translations(&common::random_quaternions(rows, 203), k, 204);
            let x = vectors::<3>(rows, 205, -2.0, 2.0);
            let out = common::evaluate_many(
                |q| {
                    if k == 1 {
                        let (a, b, x) = (p(q[0]), p(q[1]), q[2].narrow::<Vector<3>>());
                        vec![
                            (a * b).matrix().untyped(),
                            a.inverse().matrix().untyped(),
                            a.between(b).matrix().untyped(),
                            a.matrix().untyped(),
                            (a * pose::identity()).untyped(),
                            (a * a.inverse()).untyped(),
                            (a * x).untyped(),
                            a.rotate(x).untyped(),
                            a.untransform(x).untyped(),
                        ]
                    } else {
                        let (a, b) = (e(q[0]), e(q[1]));
                        vec![
                            (a * b).matrix().untyped(),
                            a.inverse().matrix().untyped(),
                            a.between(b).matrix().untyped(),
                            a.matrix().untyped(),
                            (a * extended_pose::identity()).untyped(),
                            (a * a.inverse()).untyped(),
                        ]
                    }
                },
                &[motions(&a, k), motions(&b, k), Rows::vectors(&x)],
                &target,
            );
            let h = |x: &[f64]| homogeneous(x, k);
            let (ha, hb) = (rows_of(&a, n), rows_of(&b, n));
            let products: Vec<f64> = ha
                .iter()
                .zip(&hb)
                .flat_map(|(a, b)| dense_stored(&dense_mul(&h(a), &h(b))))
                .collect();
            close(&out[0], &products, tol);
            let inverses = stored_rows(&a, n, |x| h(&motion_inverse(x, k)));
            close(&out[1], &inverses, tol);
            let between: Vec<f64> = ha
                .iter()
                .zip(&hb)
                .flat_map(|(a, b)| dense_stored(&dense_mul(&h(&motion_inverse(a, k)), &h(b))))
                .collect();
            close(&out[2], &between, tol);
            close(&out[3], &stored_rows(&a, n, h), tol);
            close(&out[4], &a, tol);
            let identity: Vec<f64> = (0..rows)
                .flat_map(|_| {
                    let mut x = vec![0.0; n];
                    x[0] = 1.0;
                    x
                })
                .collect();
            close(&out[5], &identity, tol);
            if k == 1 {
                let (mut transformed, mut rotated, mut back) = (vec![], vec![], vec![]);
                for (a, x) in ha.iter().zip(&x) {
                    let r = rot([a[0], a[1], a[2], a[3]]);
                    let rx = mv(r, *x);
                    rotated.extend(rx);
                    transformed.extend([rx[0] + a[4], rx[1] + a[5], rx[2] + a[6]]);
                    back.extend(mv(transpose(r), [x[0] - a[4], x[1] - a[5], x[2] - a[6]]));
                }
                close(&out[6], &transformed, tol);
                close(&out[7], &rotated, tol);
                close(&out[8], &back, tol);
            }
        }
    }

    fn parts_and_products(target) {
        let tol = tolerance(&target);
        let qs = quaternions(5, 211);
        let rows = qs.len();
        let poses = with_translations(&qs, 1, 212);
        let extended = with_translations(&qs, 2, 213);
        let out = common::evaluate_many(
            |q| {
                let (t, x) = (p(q[0]), e(q[1]));
                vec![
                    pose::from_parts(t.rotation(), t.position()).untyped(),
                    t.rotation().untyped(),
                    t.position().untyped(),
                    pose::from_product(t.as_product()).untyped(),
                    t.as_product().untyped(),
                    extended_pose::from_parts(x.rotation(), x.velocity(), x.position()).untyped(),
                    x.velocity().untyped(),
                    x.position().untyped(),
                    x.pose().untyped(),
                ]
            },
            &[motions(&poses, 1), motions(&extended, 2)],
            &target,
        );
        close(&out[0], &poses, tol);
        close(&out[1], &qs.concat(), tol);
        let positions: Vec<f64> = rows_of(&poses, 7).iter().flat_map(|x| x[4..7].to_vec()).collect();
        close(&out[2], &positions, tol);
        close(&out[3], &poses, tol);
        close(&out[4], &poses, tol);
        close(&out[5], &extended, tol);
        let ext = rows_of(&extended, 10);
        close(&out[6], &ext.iter().flat_map(|x| x[4..7].to_vec()).collect::<Vec<_>>(), tol);
        close(&out[7], &ext.iter().flat_map(|x| x[7..10].to_vec()).collect::<Vec<_>>(), tol);
        let pose_of: Vec<f64> = ext
            .iter()
            .flat_map(|x| x[0..4].iter().chain(&x[7..10]).copied().collect::<Vec<_>>())
            .collect();
        close(&out[8], &pose_of, tol);

        // The derivative of from_product in charts is diag(I, Rᵀ): the product moves the position
        // in a fixed frame, the pose in its own.
        let product = Space::product([Rotation.space(), Space::Vector(3)]).unwrap();
        let jacobian = common::jacobian(
            |q| pose::from_product(q[0].narrow()).untyped(),
            &[Rows::new(product, poses.clone())],
            0,
            &target,
        );
        let want: Vec<f64> = rows_of(&poses, 7)
            .iter()
            .flat_map(|x| {
                let rt = transpose(rot([x[0], x[1], x[2], x[3]]));
                let mut j = vec![0.0; 36];
                for i in 0..3 {
                    j[i * 6 + i] = 1.0;
                    for c in 0..3 {
                        j[(3 + i) * 6 + 3 + c] = rt[i][c];
                    }
                }
                j
            })
            .collect();
        assert_eq!(jacobian.len(), rows * 36);
        close(&jacobian, &want, tol);
    }
}
