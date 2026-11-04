#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)] // a test crate
//! Euler angles: round trips for every convention, the fixed axes of the extrinsic ones, the
//! locks, and the aerospace convention against its textbook matrix.

mod common;

use std::f64::consts::PI;

use common::host::{M3, mm, rot, stored};
use common::{Rows, close, tolerance, vectors};
use skoro_core::{QDyn, Real, Space, Vector};
use skoro_spatial::{
    Axis, Euler, Frame, ROTATION, Rotation, RotationMaps, Sequence, angle_residual, host, rotation,
};

fn rotations(rows: &[[f64; 4]]) -> Rows {
    Rows::new(Space::Embedded(ROTATION.clone()), rows.concat())
}

/// The matrix of the turn by `angle` about a coordinate axis, by rows.
fn turn(axis: Axis, angle: f64) -> M3 {
    let (s, c) = angle.sin_cos();
    match axis {
        Axis::X => [[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]],
        Axis::Y => [[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]],
        Axis::Z => [[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]],
    }
}

/// The matrix of the convention's angles: the product of the turns in the order of the axes when
/// intrinsic, in the reverse order when extrinsic.
fn product_of_turns(angles: [f64; 3], convention: Euler) -> M3 {
    let [i, j, k] = convention.axes();
    let (a, b, c) = (turn(i, angles[0]), turn(j, angles[1]), turn(k, angles[2]));
    match convention.frame {
        Frame::Intrinsic => mm(mm(a, b), c),
        Frame::Extrinsic => mm(mm(c, b), a),
    }
}

fn index(axis: Axis) -> usize {
    match axis {
        Axis::X => 0,
        Axis::Y => 1,
        Axis::Z => 2,
    }
}

/// Unit quaternions exactly at the locks of the intrinsic `sequence`, built in storage so that the
/// vanishing norm is zero exactly.
fn locks(sequence: Sequence) -> Vec<[f64; 4]> {
    let [i, j, _] = sequence.axes();
    let (i, j) = (index(i), index(j));
    let l = 3 - i - j;
    let epsilon = if (j + 3 - i) % 3 == 1 { 1.0 } else { -1.0 };
    let quaternion = |w: f64, ui: f64, uj: f64, ul: f64| {
        let mut q = [w, 0.0, 0.0, 0.0];
        q[1 + i] = ui;
        q[1 + j] = uj;
        q[1 + l] = ul;
        q
    };
    let h = std::f64::consts::FRAC_1_SQRT_2;
    if sequence.is_proper() {
        // β = 0 where (u_j, u_l) = 0, and β = π where (w, u_i) = 0.
        vec![
            [1.0, 0.0, 0.0, 0.0],
            quaternion(0.6, 0.8, 0.0, 0.0),
            quaternion(0.0, 0.0, 0.6, 0.8),
        ]
    } else {
        // β = π/2 where w = u_j and u_i = ε u_k, and β = -π/2 where w = -u_j and u_i = -ε u_k.
        vec![
            quaternion(h, 0.0, h, 0.0),
            quaternion(0.5, 0.5, 0.5, epsilon * 0.5),
            quaternion(0.5, 0.5, -0.5, -epsilon * 0.5),
            quaternion(h, 0.0, -h, 0.0),
        ]
    }
}

fn conventions() -> impl Iterator<Item = Euler> {
    Sequence::ALL.into_iter().flat_map(|sequence| {
        [Frame::Intrinsic, Frame::Extrinsic]
            .into_iter()
            .map(move |frame| Euler { sequence, frame })
    })
}

fn matrices(qs: &[[f64; 4]]) -> Vec<f64> {
    qs.iter().flat_map(|q| stored(rot(*q))).collect()
}

skoro_core::precision_tests! {
    fn euler_round_trips(target) {
        let tol = tolerance(&target);
        for convention in conventions() {
            let rows = 8;
            let outer = vectors::<2>(rows, 91, -3.0, 3.0);
            let (low, high) = if convention.sequence.is_proper() {
                (0.3, PI - 0.3)
            } else {
                (-PI / 2.0 + 0.3, PI / 2.0 - 0.3)
            };
            let middle = vectors::<1>(rows, 92, low, high);
            let angles: Vec<[f64; 3]> =
                (0..rows).map(|r| [outer[r][0], middle[r][0], outer[r][1]]).collect();
            let qs: Vec<[f64; 4]> = angles.iter().map(|a| host::from_euler(*a, convention)).collect();
            let minus: Vec<[f64; 4]> = qs.iter().map(|q| q.map(|v| -v)).collect();
            let out = common::evaluate_many(
                |q| {
                    let angles = q[0].narrow::<Vector<3>>();
                    let (plus, minus) = (q[1].narrow::<Rotation>(), q[2].narrow::<Rotation>());
                    let made = rotation::from_euler(angles, convention);
                    vec![
                        made.untyped(),
                        made.matrix().untyped(),
                        plus.euler(convention).untyped(),
                        minus.euler(convention).untyped(),
                    ]
                },
                &[Rows::vectors(&angles), rotations(&qs), rotations(&minus)],
                &target,
            );
            let name = format!("{convention:?}");
            close(&out[0], &qs.concat(), tol);
            let turns: Vec<f64> = angles
                .iter()
                .flat_map(|a| stored(product_of_turns(*a, convention)))
                .collect();
            close(&out[1], &turns, tol);
            close(&out[2], &angles.concat(), tol);
            close(&out[3], &angles.concat(), tol);
            let on_host: Vec<f64> = qs.iter().flat_map(|q| host::euler(*q, convention)).collect();
            close(&on_host, &angles.concat(), 1e-12);
            assert!(out.iter().flatten().all(|v| v.is_finite()), "{name}");

            // At an exact lock: finite angles, γ = 0, and the same rotation.
            let (intrinsic, gamma) = match convention.frame {
                Frame::Intrinsic => (convention.sequence, 2),
                Frame::Extrinsic => (convention.sequence.reversed(), 0),
            };
            let locked = locks(intrinsic);
            let out = common::evaluate_many(
                |q| {
                    let angles = q[0].narrow::<Rotation>().euler(convention);
                    vec![
                        angles.untyped(),
                        rotation::from_euler(angles, convention).matrix().untyped(),
                    ]
                },
                &[rotations(&locked)],
                &target,
            );
            assert!(out[0].iter().all(|v| v.is_finite()), "{name}: {:?}", out[0]);
            for row in 0..locked.len() {
                assert_eq!(out[0][3 * row + gamma], 0.0, "{name}: {:?}", out[0]);
                let on_host = host::euler(locked[row], convention);
                close(&on_host, &out[0][3 * row..3 * row + 3], tol);
                assert_eq!(on_host[gamma], 0.0);
            }
            close(&out[1], &matrices(&locked), tol);

            // Near the lock of a proper sequence, whose rearrangement is the quaternion itself:
            // within the threshold, the convention γ = 0; just outside it, the general formulas;
            // the rotation reproduced either way.
            if intrinsic.is_proper() {
                let near = near_locks(intrinsic);
                let out = common::evaluate_many(
                    |q| {
                        let angles = q[0].narrow::<Rotation>().euler(convention);
                        vec![
                            angles.untyped(),
                            rotation::from_euler(angles, convention).matrix().untyped(),
                        ]
                    },
                    &[rotations(&near)],
                    &target,
                );
                assert!(out.iter().flatten().all(|v| v.is_finite()), "{name}: {out:?}");
                for row in 0..near.len() {
                    let gamma_out = out[0][3 * row + gamma];
                    if row < 2 {
                        assert_eq!(gamma_out, 0.0, "{name}, row {row}: {:?}", out[0]);
                    } else {
                        assert!(gamma_out.abs() > 0.1, "{name}, row {row}: {:?}", out[0]);
                    }
                    let on_host = host::euler(near[row], convention);
                    close(&on_host, &out[0][3 * row..3 * row + 3], tol);
                }
                close(&out[1], &matrices(&near), tol);
            }

            // At the locks of a Tait–Bryan sequence as `from_euler` computes them, where rounding
            // leaves the vanishing norm small rather than zero: β and the combination of α and γ
            // that the lock determines are accurate, and the rotation is reproduced.
            if !intrinsic.is_proper() && convention.frame == Frame::Intrinsic {
                let epsilon = epsilon_of(intrinsic);
                let free = vectors::<2>(6, 93, -3.0, 3.0);
                let angles: Vec<[f64; 3]> = free
                    .iter()
                    .enumerate()
                    .map(|(r, a)| [a[0], if r % 2 == 0 { PI / 2.0 } else { -PI / 2.0 }, a[1]])
                    .collect();
                let qs: Vec<[f64; 4]> =
                    angles.iter().map(|a| host::from_euler(*a, convention)).collect();
                let out = common::evaluate_many(
                    |q| {
                        let angles = q[0].narrow::<Rotation>().euler(convention);
                        vec![
                            angles.untyped(),
                            rotation::from_euler(angles, convention).matrix().untyped(),
                        ]
                    },
                    &[rotations(&qs)],
                    &target,
                );
                assert!(out.iter().flatten().all(|v| v.is_finite()), "{name}: {out:?}");
                for (row, want) in angles.iter().enumerate() {
                    let got = &out[0][3 * row..3 * row + 3];
                    // α - εγ at β = -π/2, α + εγ at β = π/2.
                    let sign = want[1].signum() * epsilon;
                    let combination = |a: &[f64]| a[0] + sign * a[2];
                    let d = combination(got) - combination(want);
                    close(&[got[1], d.sin().atan2(d.cos())], &[want[1], 0.0], tol);
                }
                close(&out[1], &matrices(&qs), tol);
            }
        }
    }
}

/// `ε = ±1` with `e_i × e_j = ε e_l` for the first two axes of `sequence`.
fn epsilon_of(sequence: Sequence) -> f64 {
    let [i, j, _] = sequence.axes();
    if (index(j) + 3 - index(i)) % 3 == 1 {
        1.0
    } else {
        -1.0
    }
}

/// Unit quaternions near the locks of the proper intrinsic `sequence`: two within the threshold,
/// at `β ≈ 2·10⁻¹⁸` and `π - 2·10⁻¹⁸`, and two just outside it, at `2·10⁻¹⁶` and `π - 2·10⁻¹⁶`.
fn near_locks(sequence: Sequence) -> Vec<[f64; 4]> {
    let [i, j, _] = sequence.axes();
    let (i, j) = (index(i), index(j));
    let l = 3 - i - j;
    let quaternion = |w: f64, ui: f64, uj: f64, ul: f64| {
        let mut q = [w, 0.0, 0.0, 0.0];
        q[1 + i] = ui;
        q[1 + j] = uj;
        q[1 + l] = ul;
        q
    };
    vec![
        quaternion(0.6, 0.8, 1e-18, 0.0),
        quaternion(1e-18, 0.0, 0.6, 0.8),
        quaternion(0.6, 0.8, 1e-16, 0.0),
        quaternion(1e-16, 0.0, 0.6, 0.8),
    ]
}

/// `a - b` brought into `(-π, π]`, on the host.
fn on_the_circle(a: f64, b: f64) -> f64 {
    let d = a - b;
    d.sin().atan2(d.cos())
}

skoro_core::precision_tests! {
    fn angle_residual_is_on_the_circle(target) {
        let tol = tolerance(&target);
        let residual_of_reals =
            |q: &[QDyn]| angle_residual(q[0].narrow::<Real>(), q[1].narrow::<Real>()).untyped();

        // At random pairs, away from the residual's own cut, against the host, on `Real` and on
        // `Vector<3>`; and `b + (a ⊖ b)` is `a` within a half-turn of `b`.
        let a = vectors::<3>(16, 94, -10.0, 10.0);
        let b = vectors::<3>(16, 95, -10.0, 10.0);
        let rows: Vec<usize> = (0..a.len())
            .filter(|&r| (0..3).all(|k| on_the_circle(a[r][k], b[r][k]).abs() < PI - 0.1))
            .collect();
        let (a, b): (Vec<[f64; 3]>, Vec<[f64; 3]>) = rows.iter().map(|&r| (a[r], b[r])).unzip();
        assert!(a.len() > 8);
        let want: Vec<f64> = a
            .iter()
            .zip(&b)
            .flat_map(|(a, b)| [0, 1, 2].map(|k| on_the_circle(a[k], b[k])))
            .collect();
        let out = common::evaluate_many(
            |q| {
                let (va, vb) = (q[0].narrow::<Vector<3>>(), q[1].narrow::<Vector<3>>());
                let r = angle_residual(va, vb);
                vec![r.untyped(), (vb + r).untyped()]
            },
            &[Rows::vectors(&a), Rows::vectors(&b)],
            &target,
        );
        close(&out[0], &want, tol);
        for (i, brought) in out[1].iter().enumerate() {
            let (a, b) = (a[i / 3][i % 3], b[i / 3][i % 3]);
            assert!(*brought > b - PI && *brought <= b + PI, "{brought} against {b}");
            let turns = (brought - a) / (2.0 * PI);
            assert!((turns - turns.round()).abs() < tol.max(1e-6), "{brought} against {a}");
        }
        let first: Vec<[f64; 1]> = a.iter().map(|a| [a[0]]).collect();
        let second: Vec<[f64; 1]> = b.iter().map(|b| [b[0]]).collect();
        let reals = |v: &[[f64; 1]]| Rows::new(Space::Real, v.concat());
        let out = common::evaluate(residual_of_reals, &[reals(&first), reals(&second)], &target);
        let want: Vec<f64> = a.iter().zip(&b).map(|(a, b)| on_the_circle(a[0], b[0])).collect();
        close(&out, &want, tol);

        // Across the wrap of either angle, the residual is continuous, and its passes, those of
        // `a - b`, the same on both sides.
        let side = 1e-5;
        for (a, b) in [
            ([PI - side, -PI + side], [3.0, 3.0]),
            ([3.0, 3.0], [PI - side, -PI + side]),
        ] {
            let inputs = [Rows::new(Space::Real, a.to_vec()), Rows::new(Space::Real, b.to_vec())];
            let out = common::evaluate(residual_of_reals, &inputs, &target);
            assert!((out[0] - out[1]).abs() < 2.0 * side + tol.max(1e-6), "{out:?}");
            close(&out, &[on_the_circle(a[0], b[0]), on_the_circle(a[1], b[1])], tol);
            for (wrt, slope) in [(0, 1.0), (1, -1.0)] {
                let tangent = common::jacobian(residual_of_reals, &inputs, wrt, &target);
                close(&tangent, &[slope, slope], tol);
                let (cotangent, second) =
                    common::second_order(residual_of_reals, &inputs, wrt, &target);
                close(&cotangent, &[slope, slope], tol);
                close(&second, &[0.0, 0.0], tol);
            }
        }

        // Along rotations about z through the half-turn, the yaw of `euler` jumps and its residual
        // against a reference yaw of 3 does not.
        let qs = [PI - side, PI + side].map(|angle| common::about([0.0, 0.0, 1.0], angle));
        let out = common::evaluate_many(
            |q| {
                let angles = q[0].narrow::<Rotation>().euler(Euler::YAW_PITCH_ROLL);
                let reference = skoro_core::scope::constant(Vector::<3>, [3.0, 0.0, 0.0]);
                vec![angles.untyped(), angle_residual(angles, reference).untyped()]
            },
            &[rotations(&qs)],
            &target,
        );
        let (yaw, residual) = ([out[0][0], out[0][3]], [out[1][0], out[1][3]]);
        assert!((yaw[0] - yaw[1] - (2.0 * PI - 2.0 * side)).abs() < tol.max(1e-5), "{yaw:?}");
        close(&residual, &[PI - side - 3.0, PI + side - 3.0], tol.max(1e-6));
    }
}

#[test]
fn yaw_pitch_roll_is_the_aerospace_convention() {
    // The body-to-navigation matrix of yaw ψ, pitch ϑ and roll φ.
    let (psi, theta, phi) = (0.5f64, 0.3f64, -0.2f64);
    let (sp, cp) = psi.sin_cos();
    let (st, ct) = theta.sin_cos();
    let (sf, cf) = phi.sin_cos();
    let textbook = [
        [ct * cp, sf * st * cp - cf * sp, cf * st * cp + sf * sp],
        [ct * sp, sf * st * sp + cf * cp, cf * st * sp - sf * cp],
        [-st, sf * ct, cf * ct],
    ];
    let q = host::from_euler([psi, theta, phi], Euler::YAW_PITCH_ROLL);
    let m = rot(q);
    for i in 0..3 {
        close(&m[i], &textbook[i], 1e-14);
    }
    close(
        &host::euler(q, Euler::YAW_PITCH_ROLL),
        &[psi, theta, phi],
        1e-14,
    );
    assert_eq!(Euler::YAW_PITCH_ROLL.axes(), [Axis::Z, Axis::Y, Axis::X]);
}
