#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)] // a test crate
//! The derivatives the passes compute, read in the charts of these types, against the Jacobians
//! of the textbooks; and the second derivatives at the centres of the charts against those in
//! exponential coordinates.

mod common;

use common::host::{hat, mv, rot, transpose};
use common::{
    Rows, by_rows_layout, close, quaternions, rotation_vectors, tolerance, twists,
    with_translations,
};
use skoro_core::scope::constant;
use skoro_core::{Declares, Space, Vector};
use skoro_spatial::{
    Pose, ROTATION, Rotation, RotationMaps, extended_pose, host, pose, rotation, so3,
};

/// `n × n` identities, one per row, laid out `[row][i][c]`.
fn identities(rows: usize, n: usize) -> Vec<f64> {
    (0..rows)
        .flat_map(|_| (0..n * n).map(move |ic| f64::from(u8::from(ic / n == ic % n))))
        .collect()
}

skoro_core::precision_tests! {
    fn jacobians_are_the_passes(target) {
        let tol = tolerance(&target);

        // Rotations: the pass of exp is J_r, that of log is J_r⁻¹ of the logarithm.
        let omega = rotation_vectors(301);
        let rows = omega.len();
        let pass = common::jacobian(
            |q| rotation::exp(q[0].narrow()).untyped(),
            &[Rows::vectors(&omega)],
            0,
            &target,
        );
        let out = common::evaluate_many(
            |q| {
                let w = q[0].narrow::<Vector<3>>();
                vec![
                    so3::jr(w).untyped(),
                    so3::jl_inv(w).matmul(so3::jl(w)).untyped(),
                    so3::jr_inv(w).matmul(so3::jr(w)).untyped(),
                ]
            },
            &[Rows::vectors(&omega)],
            &target,
        );
        close(&pass, &by_rows_layout(&out[0], 3), tol);
        close(&out[1], &identities(rows, 3), tol);
        close(&out[2], &identities(rows, 3), tol);

        let qs = quaternions(5, 302);
        let rotations = Rows::new(Space::Embedded(ROTATION.clone()), qs.concat());
        let pass = common::jacobian(
            |q| q[0].narrow::<Rotation>().log().untyped(),
            std::slice::from_ref(&rotations),
            0,
            &target,
        );
        let inverse = common::evaluate(
            |q| so3::jr_inv(q[0].narrow::<Rotation>().log()).untyped(),
            &[rotations],
            &target,
        );
        close(&pass, &by_rows_layout(&inverse, 3), tol);

        // The motions: the pass of exp is 𝒥_r, and the inverses invert.
        for k in [1, 2] {
            let d = 3 + 3 * k;
            let xi = Rows::new(Space::Vector(d), twists(k, 303));
            let rows = xi.rows();
            let pass = common::jacobian(
                |q| {
                    if k == 1 {
                        pose::exp(q[0].narrow()).untyped()
                    } else {
                        extended_pose::exp(q[0].narrow()).untyped()
                    }
                },
                std::slice::from_ref(&xi),
                0,
                &target,
            );
            let out = common::evaluate_many(
                |q| {
                    if k == 1 {
                        let x = q[0].narrow::<Vector<6>>();
                        vec![
                            pose::jr(x).untyped(),
                            pose::jl_inv(x).matmul(pose::jl(x)).untyped(),
                            pose::jr_inv(x).matmul(pose::jr(x)).untyped(),
                        ]
                    } else {
                        let x = q[0].narrow::<Vector<9>>();
                        vec![
                            extended_pose::jr(x).untyped(),
                            extended_pose::jl_inv(x).matmul(extended_pose::jl(x)).untyped(),
                            extended_pose::jr_inv(x).matmul(extended_pose::jr(x)).untyped(),
                        ]
                    }
                },
                &[xi],
                &target,
            );
            close(&pass, &by_rows_layout(&out[0], d), tol);
            close(&out[1], &identities(rows, d), tol);
            close(&out[2], &identities(rows, d), tol);
        }
    }

    fn second_order_agrees_on_rotations(target) {
        // At f32 the passes are held to the host's differences in double precision.
        let tol = match target.precision() {
            skoro_tensor::Precision::Double => 1e-6,
            skoro_tensor::Precision::Single => 1e-3,
        };
        let (a, b) = ([0.3, -0.7, 0.5], [0.8, 0.1, -0.4]);

        // On the rotations, f(x) = ⟨π(x) a, b⟩: the Hessian of t ↦ f(x ⊕ t) at 0, by the pass, is
        // that of t ↦ f(x Exp t), by central differences.
        let qs = quaternions(3, 311);
        let rows = qs.len();
        let pass = common::hessian(
            |q| {
                let x = q[0].narrow::<Rotation>().retract(q[1]);
                let rotated = x * constant(Vector::<3>, a);
                (rotated.untyped() * constant(Vector::<3>, b).untyped()).sum()
            },
            &[
                Rows::new(Space::Embedded(ROTATION.clone()), qs.concat()),
                Rows::vectors(&vec![[0.0; 3]; rows]),
            ],
            1,
            &target,
        );
        let fd: Vec<f64> = qs
            .iter()
            .flat_map(|q| {
                differences(3, |t| {
                    let y = host::compose(*q, host::exp([t[0], t[1], t[2]]));
                    common::host::dot(&mv(rot(y), a), &b)
                })
            })
            .collect();
        close(&pass, &fd, tol);

        // On the poses, f(T) = ⟨T a, b⟩: the two differ by the term ½ φ × ρ of the position, whose
        // second derivatives are ½ c·(eᵢ × eⱼ) in (φᵢ, ρⱼ) and (ρⱼ, φᵢ), with c = Rᵀ b.
        let x = with_translations(&qs, 1, 312);
        let pass = common::hessian(
            |q| {
                let t = q[0].narrow::<Pose>().retract(q[1]);
                let moved = t * constant(Vector::<3>, a);
                (moved.untyped() * constant(Vector::<3>, b).untyped()).sum()
            },
            &[
                Rows::new(Pose.space(), x.clone()),
                Rows::new(Space::Vector(6), vec![0.0; 6 * rows]),
            ],
            1,
            &target,
        );
        let mut corrected = Vec::new();
        for x in x.chunks(7) {
            let (q, p) = ([x[0], x[1], x[2], x[3]], [x[4], x[5], x[6]]);
            let mut h = differences(6, |t| {
                let (phi, rho) = ([t[0], t[1], t[2]], [t[3], t[4], t[5]]);
                let r = rot(host::compose(q, host::exp(phi)));
                let moved = mv(rot(q), jl(phi, rho));
                let point: Vec<f64> = (0..3).map(|i| mv(r, a)[i] + p[i] + moved[i]).collect();
                common::host::dot(&point, &b)
            });
            let c = mv(transpose(rot(q)), b);
            for i in 0..3 {
                for j in 0..3 {
                    let mut e = ([0.0; 3], [0.0; 3]);
                    e.0[i] = 1.0;
                    e.1[j] = 1.0;
                    let term = 0.5 * common::host::dot(&c, &common::host::cross(e.0, e.1));
                    h[i * 6 + 3 + j] -= term;
                    h[(3 + j) * 6 + i] -= term;
                }
            }
            corrected.extend(h);
        }
        close(&pass, &corrected, tol);
    }
}

/// `J_l(φ) ρ` on the host, with the series of `A` and `B` near zero.
fn jl(phi: [f64; 3], rho: [f64; 3]) -> [f64; 3] {
    let t2 = common::host::dot(&phi, &phi);
    let t = t2.sqrt();
    let (a, b) = if t2 < 1e-4 {
        (0.5 - t2 / 24.0, 1.0 / 6.0 - t2 / 120.0)
    } else {
        ((1.0 - t.cos()) / t2, (t - t.sin()) / (t2 * t))
    };
    let h = hat(phi);
    let hr = mv(h, rho);
    let hhr = mv(h, hr);
    std::array::from_fn(|i| rho[i] + a * hr[i] + b * hhr[i])
}

/// The Hessian of `g` at zero by central differences, `d × d` by rows.
fn differences(d: usize, g: impl Fn(&[f64]) -> f64) -> Vec<f64> {
    let h = 1e-3;
    let at = |i: usize, si: f64, j: usize, sj: f64| {
        let mut t = vec![0.0; d];
        t[i] += si * h;
        t[j] += sj * h;
        g(&t)
    };
    let mut out = vec![0.0; d * d];
    for i in 0..d {
        for j in 0..d {
            out[i * d + j] = (at(i, 1.0, j, 1.0) - at(i, 1.0, j, -1.0) - at(i, -1.0, j, 1.0)
                + at(i, -1.0, j, -1.0))
                / (4.0 * h * h);
        }
    }
    out
}
