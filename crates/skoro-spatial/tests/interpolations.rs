#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)] // a test crate
//! The interpolations of the rotations and the motions, and the angle between rotations.

mod common;

use common::host::{conj, qmul, rot, stored};
use common::{Rows, close, quaternions, random_quaternions, tolerance, with_translations};
use skoro_core::{Declares, Real, Space};
use skoro_spatial::{
    ExtendedPose, ExtendedPoseMaps, Pose, PoseMaps, ROTATION, Rotation, RotationMaps,
};

fn rotations(rows: &[[f64; 4]]) -> Rows {
    Rows::new(Space::Embedded(ROTATION.clone()), rows.concat())
}

fn reals(rows: usize, t: f64) -> Rows {
    Rows::new(Space::Real, vec![t; rows])
}

fn matrices(data: &[f64]) -> Vec<f64> {
    data.chunks(4)
        .flat_map(|q| stored(rot([q[0], q[1], q[2], q[3]])))
        .collect()
}

skoro_core::precision_tests! {
    fn interpolations(target) {
        let tol = tolerance(&target);
        let a = quaternions(4, 401);
        let rows = a.len();
        let b = random_quaternions(rows, 402);
        let at = |t: f64| {
            common::evaluate_many(
                |q| {
                    let (a, b, t) = (q[0].narrow::<Rotation>(), q[1].narrow::<Rotation>(), q[2].narrow::<Real>());
                    let (long, short) = (a.slerp(b, t), a.slerp_shorter(b, t));
                    vec![
                        long.untyped(),
                        short.untyped(),
                        a.angle_to(long).untyped(),
                        a.angle_to(short).untyped(),
                        a.angle_to(b).untyped(),
                    ]
                },
                &[rotations(&a), rotations(&b), reals(rows, t)],
                &target,
            )
        };

        // At the endpoints: the first rotation, and the second, as a quaternion along the
        // geodesic and as a rotation along the shorter turn.
        let start = at(0.0);
        close(&start[0], &a.concat(), tol);
        close(&start[1], &a.concat(), tol);
        let end = at(1.0);
        close(&end[0], &b.concat(), tol);
        close(&matrices(&end[1]), &matrices(&b.concat()), tol);

        // The angle against the host's arccos((tr π(q̄₁ q₂) - 1) / 2).
        let angles: Vec<f64> = a
            .iter()
            .zip(&b)
            .map(|(a, b)| {
                let m = rot(qmul(conj(*a), *b));
                ((m[0][0] + m[1][1] + m[2][2] - 1.0) / 2.0).clamp(-1.0, 1.0).acos()
            })
            .collect();
        close(&end[4], &angles, tol.max(1e-6));

        // The angle from the start grows linearly along the shorter turn; along the geodesic of the
        // quaternions it is the longer turn's where ⟨q₁, q₂⟩ < 0.
        for t in [0.25, 0.5, 0.75] {
            let out = at(t);
            for row in 0..rows {
                let angle = angles[row];
                close(&[out[3][row]], &[t * angle], tol.max(1e-6));
                let along = if common::host::dot(&a[row], &b[row]) >= 0.0 {
                    t * angle
                } else {
                    // A turn by t (2π - angle), whose own angle is at most π.
                    let turned = t * (2.0 * std::f64::consts::PI - angle);
                    turned.min(2.0 * std::f64::consts::PI - turned)
                };
                close(&[out[2][row]], &[along], tol.max(1e-6));
            }
        }

        // The motions: endpoints, the screw's twist constant in t, the parts' position linear.
        for k in [1, 2] {
            let n = 4 + 3 * k;
            let space = if k == 1 { Pose.space() } else { ExtendedPose.space() };
            let x = with_translations(&a, k, 403);
            let y = with_translations(&b, k, 404);
            let at = |t: f64| {
                common::evaluate_many(
                    |q| {
                        let t = q[2].narrow::<Real>();
                        if k == 1 {
                            let (x, y) = (q[0].narrow::<Pose>(), q[1].narrow::<Pose>());
                            let (screw, short) = (x.interpolate(y, t), x.interpolate_shorter(y, t));
                            vec![
                                screw.untyped(),
                                short.untyped(),
                                x.between(screw).log().untyped(),
                                x.between(y).log().untyped(),
                                x.between(short).log_shorter().untyped(),
                                x.between(y).log_shorter().untyped(),
                                x.interpolate_parts(y, t).untyped(),
                                x.interpolate_parts_shorter(y, t).untyped(),
                            ]
                        } else {
                            let (x, y) = (q[0].narrow::<ExtendedPose>(), q[1].narrow::<ExtendedPose>());
                            let (screw, short) = (x.interpolate(y, t), x.interpolate_shorter(y, t));
                            vec![
                                screw.untyped(),
                                short.untyped(),
                                x.between(screw).log().untyped(),
                                x.between(y).log().untyped(),
                                x.between(short).log_shorter().untyped(),
                                x.between(y).log_shorter().untyped(),
                            ]
                        }
                    },
                    &[Rows::new(space.clone(), x.clone()), Rows::new(space.clone(), y.clone()), reals(rows, t)],
                    &target,
                )
            };
            let start = at(0.0);
            close(&start[0], &x, tol);
            close(&start[1], &x, tol);
            let end = at(1.0);
            close(&end[0], &y, tol);
            let same_motion = |got: &[f64], want: &[f64]| {
                for (g, w) in got.chunks(n).zip(want.chunks(n)) {
                    close(&stored(rot([g[0], g[1], g[2], g[3]])), &stored(rot([w[0], w[1], w[2], w[3]])), tol);
                    close(&g[4..], &w[4..], tol);
                }
            };
            same_motion(&end[1], &y);
            for t in [0.3, 0.6] {
                let out = at(t);
                let scaled: Vec<f64> = out[3].iter().map(|v| t * v).collect();
                close(&out[2], &scaled, tol.max(1e-6));
                let scaled: Vec<f64> = out[5].iter().map(|v| t * v).collect();
                close(&out[4], &scaled, tol.max(1e-6));
                if k == 1 {
                    for (row, (g, h)) in out[6].chunks(7).zip(out[7].chunks(7)).enumerate() {
                        let (p, r) = (&x[row * 7 + 4..row * 7 + 7], &y[row * 7 + 4..row * 7 + 7]);
                        let line: Vec<f64> = (0..3).map(|i| (1.0 - t) * p[i] + t * r[i]).collect();
                        close(&g[4..], &line, tol);
                        close(&h[4..], &line, tol);
                    }
                }
            }
            if k == 1 {
                let end = at(1.0);
                close(&end[6], &y, tol);
                same_motion(&end[7], &y);
            }
        }
    }
}
