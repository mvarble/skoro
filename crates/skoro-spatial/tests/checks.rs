#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)] // a test crate
//! Every composite of the crate against finite differences, and every discretization on both
//! sides of its jumps.

mod common;

use std::sync::Arc;

use common::host::{conj, qmul, rot, stored};
use common::{Rows, quaternions, random_quaternions, vectors, with_translations};
use skoro_core::scope::{concatenate, constant, project};
use skoro_core::testing::{check_agreement, check_operation};
use skoro_core::{Composite, Declares, Param, QDyn, Space, Vector, VectorDyn};
use skoro_spatial::{
    Euler, ExtendedPose, ExtendedPoseMaps, Frame, Pose, PoseMaps, Rotation, RotationMaps, Sequence,
    extended_pose, host, pose, rotation, so3,
};
use skoro_tensor::{HostValue, Precision, Target};

/// A map of the crate, by name, with the spaces of its inputs.
type Map = Box<dyn Fn(&[QDyn]) -> QDyn>;

/// The map `f` as a composite, with a second, `Real` input that adds zero to its output:
/// `check_operation` asks a composite's second-order map with its only input's velocity missing
/// for a frame as wide as the velocities, which a composite given no velocity cannot know.
fn composite(index: usize, inputs: &[Space], f: &Map) -> Arc<Composite> {
    let params = vec![Param::Integer(index as i64)];
    let inner = Composite::define("test.spatial.map", params.clone(), inputs, |q| f(q)).unwrap();
    let mut spaces = inputs.to_vec();
    spaces.push(Space::Real);
    let output = inner.output().clone();
    Composite::define("test.spatial.with_real", params, &spaces, |q| {
        let (out, r) = (inner.call(&q[..q.len() - 1]), q[q.len() - 1]);
        match &output {
            Space::Embedded(declaration) => {
                let zeros = vec![0.0; declaration.dimension()];
                let zeros = constant(VectorDyn(zeros.len()), zeros.as_slice());
                out.retract(r * zeros.untyped())
            }
            space => {
                let zeros = vec![0.0; space.storage()];
                let zeros = constant(VectorDyn(zeros.len()), zeros.as_slice());
                out + r * zeros.untyped().reshape(space.clone())
            }
        }
    })
    .unwrap()
}

fn rotation_space() -> Space {
    Rotation.space()
}

fn rows_of(space: Space, data: Vec<f64>) -> Rows {
    Rows::new(space, data)
}

/// `rows` reals in `(low, high)`.
fn reals(rows: usize, seed: u64, low: f64, high: f64) -> Rows {
    Rows::new(Space::Real, vectors::<1>(rows, seed, low, high).concat())
}

/// Rows of `rows` vectors of length `n` at moderate lengths, the identity and the cutoffs among
/// them for the rotation vector in front.
fn twists(n: usize, rows: usize, seed: u64) -> Rows {
    let lengths = [
        0.0,
        (0.9 * common::CUTOFF).sqrt(),
        (1.1 * common::CUTOFF).sqrt(),
        0.6,
        1.5,
        3.0,
    ];
    let directions = vectors::<3>(rows, seed, -1.0, 1.0);
    let rest = vectors::<6>(rows, seed + 1, -1.5, 1.5);
    let data = (0..rows)
        .flat_map(|r| {
            let phi = common::unit(directions[r]).map(|v| v * lengths[r % lengths.len()]);
            phi.into_iter()
                .chain(rest[r][..n - 3].iter().copied())
                .collect::<Vec<_>>()
        })
        .collect();
    Rows::new(Space::Vector(n), data)
}

/// `min(n_a, n_b)` of the statement euler for the intrinsic `sequence`, as a fraction of `|q|²`:
/// the distance from its lock.
fn lock_distance(sequence: Sequence, q: [f64; 4]) -> f64 {
    let (w, ui, uj, ul) = euler_parts(sequence, q);
    (w * w + ui * ui).min(uj * uj + ul * ul)
}

fn index(axis: skoro_spatial::Axis) -> usize {
    match axis {
        skoro_spatial::Axis::X => 0,
        skoro_spatial::Axis::Y => 1,
        skoro_spatial::Axis::Z => 2,
    }
}

/// `i`, `j`, `l` and `ε` of an intrinsic sequence.
fn frame(sequence: Sequence) -> (usize, usize, usize, f64) {
    let [i, j, _] = sequence.axes();
    let (i, j) = (index(i), index(j));
    let epsilon = if (j + 3 - i) % 3 == 1 { 1.0 } else { -1.0 };
    (i, j, 3 - i - j, epsilon)
}

/// `(w, u_i, u_j, u_l)` read off `q`, or off its rearrangement for a Tait–Bryan sequence.
fn euler_parts(sequence: Sequence, q: [f64; 4]) -> (f64, f64, f64, f64) {
    let (i, j, l, epsilon) = frame(sequence);
    let at = |a: usize| q[1 + a];
    if sequence.is_proper() {
        (q[0], at(i), at(j), at(l))
    } else {
        (
            q[0] - at(j),
            at(i) - epsilon * at(l),
            at(j) + q[0],
            at(l) + epsilon * at(i),
        )
    }
}

/// The largest coordinate of `q` in magnitude beats the next by more than `margin`.
fn decided(q: &[f64; 4], margin: f64) -> bool {
    let mut m: Vec<f64> = q.iter().map(|v| v.abs()).collect();
    m.sort_by(|a, b| b.total_cmp(a));
    m[0] - m[1] > margin
}

fn check(name: &str, index: usize, inputs: &[Rows], f: Map, target: &Target) {
    let spaces: Vec<Space> = inputs.iter().map(|r| r.space.clone()).collect();
    let composite = composite(index, &spaces, &f);
    let rows = inputs[0].rows();
    assert!(rows >= 2, "{name}: {rows} rows");
    let mut values: Vec<(Space, HostValue)> = inputs.iter().map(Rows::input).collect();
    values.push(reals(rows, 999, 0.2, 0.8).input());
    if let Err(error) = check_operation(&*composite.operation(), &values, target) {
        panic!("{name}: {error}");
    }
}

fn r(q: QDyn) -> skoro_core::Q<Rotation> {
    q.narrow()
}

fn p(q: QDyn) -> skoro_core::Q<Pose> {
    q.narrow()
}

fn e(q: QDyn) -> skoro_core::Q<ExtendedPose> {
    q.narrow()
}

/// The rotation `-1`, a constant: `q * minus()` is `-q`, smoothly in `q`.
fn minus() -> skoro_core::Q<Rotation> {
    constant(Rotation, [-1.0, 0.0, 0.0, 0.0])
}

fn minus_pose() -> skoro_core::Q<Pose> {
    constant(Pose, ([-1.0, 0.0, 0.0, 0.0], [0.0; 3]))
}

fn minus_extended() -> skoro_core::Q<ExtendedPose> {
    constant(ExtendedPose, ([-1.0, 0.0, 0.0, 0.0], [0.0; 3], [0.0; 3]))
}

/// `|v|`, for `v ≠ 0`.
fn length(v: QDyn) -> QDyn {
    ((v * v).sum().log() * 0.5).exp()
}

skoro_core::precision_tests! {
    fn composites_pass_check_operation(target) {
        let qs = quaternions(4, 501);
        let n = qs.len();
        let others = random_quaternions(n, 502);
        let rot_rows = Rows::new(rotation_space(), qs.concat());
        let other_rows = Rows::new(rotation_space(), others.concat());
        let v3 = Rows::vectors(&vectors::<3>(n, 503, -1.5, 1.5));
        let v3b = Rows::vectors(&vectors::<3>(n, 504, -1.5, 1.5));
        let omega = twists(3, n, 505);
        let t = reals(n, 506, -0.5, 1.5);
        let angle = reals(n, 507, -3.0, 3.0);
        let poses = rows_of(Pose.space(), with_translations(&qs, 1, 508));
        let poses_b = rows_of(Pose.space(), with_translations(&others, 1, 509));
        let extended = rows_of(ExtendedPose.space(), with_translations(&qs, 2, 510));
        let extended_b = rows_of(ExtendedPose.space(), with_translations(&others, 2, 511));
        let xi6 = twists(6, n, 512);
        let xi9 = twists(9, n, 513);

        // The rows of the discrete maps, ten single-precision steps from their jumps.
        let away = |keep: &dyn Fn(&[f64; 4], &[f64; 4]) -> bool, k: usize| {
            let kept: Vec<([f64; 4], [f64; 4])> = qs
                .iter()
                .zip(&others)
                .filter(|(a, b)| keep(a, b))
                .map(|(a, b)| (*a, *b))
                .collect();
            let a: Vec<[f64; 4]> = kept.iter().map(|(a, _)| *a).collect();
            let b: Vec<[f64; 4]> = kept.iter().map(|(_, b)| *b).collect();
            let space = match k {
                0 => rotation_space(),
                1 => Pose.space(),
                _ => ExtendedPose.space(),
            };
            let rows = a.len();
            (
                Rows::new(space.clone(), with_translations(&a, k, 520)),
                Rows::new(space, with_translations(&b, k, 521)),
                reals(rows, 522, -0.5, 1.5),
            )
        };
        let relative_w = |a: &[f64; 4], b: &[f64; 4]| qmul(conj(*a), *b)[0];
        let shorter_rows = |k| away(&|a, b| relative_w(a, b).abs() > 0.1, k);
        let own_w = |k| away(&|a, _| a[0].abs() > 0.1, k);

        let mut index = 0;
        let mut run = |name: &str, inputs: &[Rows], f: Map| {
            index += 1;
            check(name, index, inputs, f, &target);
        };

        // The rotations.
        run("compose", &[rot_rows.clone(), other_rows.clone()], Box::new(|q| (r(q[0]) * r(q[1])).untyped()));
        run("inverse", std::slice::from_ref(&rot_rows), Box::new(|q| r(q[0]).inverse().untyped()));
        run("between", &[rot_rows.clone(), other_rows.clone()], Box::new(|q| r(q[0]).between(r(q[1])).untyped()));
        run("rotate", &[rot_rows.clone(), v3.clone()], Box::new(|q| (r(q[0]) * q[1].narrow::<Vector<3>>()).untyped()));
        run("unrotate", &[rot_rows.clone(), v3.clone()], Box::new(|q| r(q[0]).unrotate(q[1].narrow()).untyped()));
        run("matrix", std::slice::from_ref(&rot_rows), Box::new(|q| r(q[0]).matrix().untyped()));
        run("exp", std::slice::from_ref(&omega), Box::new(|q| rotation::exp(q[0].narrow()).untyped()));
        run("log", std::slice::from_ref(&rot_rows), Box::new(|q| r(q[0]).log().untyped()));
        let (a, _, _) = own_w(0);
        run("log_shorter", &[a], Box::new(|q| r(q[0]).log_shorter().untyped()));
        run("jl", std::slice::from_ref(&omega), Box::new(|q| so3::jl(q[0].narrow()).untyped()));
        run("jl_inv", std::slice::from_ref(&omega), Box::new(|q| so3::jl_inv(q[0].narrow()).untyped()));
        run("about", &[v3.clone(), angle.clone()], Box::new(|q| rotation::about(q[0].narrow(), q[1].narrow()).untyped()));
        let (a, _, _) = away(&|a, _| (1.0 - a[0] * a[0]).sqrt() > 0.1, 0);
        run("axis_angle", &[a], Box::new(|q| {
            let (axis, angle) = r(q[0]).axis_angle();
            concatenate(&[axis.untyped(), angle.untyped()], Space::Vector(4))
        }));
        run("from_two_vectors", &[v3.clone(), v3b.clone()], Box::new(|q| rotation::from_two_vectors(q[0].narrow(), q[1].narrow()).untyped()));
        let (a, b, _) = away(&|a, b| common::host::dot(a, b).abs() > 0.1, 0);
        run("align", &[a, b], Box::new(|q| r(q[0]).aligned_with(r(q[1])).untyped()));
        let decided_rows: Vec<[f64; 4]> = qs.iter().filter(|q| decided(q, 0.1)).copied().collect();
        let matrices: Vec<f64> = decided_rows
            .iter()
            .enumerate()
            .flat_map(|(i, q)| stored(rot(*q)).into_iter().enumerate().map(move |(j, v)| v + 0.01 * ((i * 9 + j) as f64).sin()))
            .collect();
        run("from_matrix", &[Rows::new(Space::Matrix(3, 3), matrices)], Box::new(|q| rotation::from_matrix(q[0].narrow()).untyped()));
        run("slerp", &[rot_rows.clone(), other_rows.clone(), t.clone()], Box::new(|q| r(q[0]).slerp(r(q[1]), q[2].narrow()).untyped()));
        let (a, b, tt) = shorter_rows(0);
        run("slerp_shorter", &[a.clone(), b.clone(), tt], Box::new(|q| r(q[0]).slerp_shorter(r(q[1]), q[2].narrow()).untyped()));
        let (a, b, _) = away(&|a, b| {
            let w = relative_w(a, b);
            w.abs() > 0.1 && (1.0 - w * w).sqrt() > 0.1
        }, 0);
        run("angle", &[a, b], Box::new(|q| r(q[0]).angle_to(r(q[1])).untyped()));
        for sequence in Sequence::ALL {
            let convention = Euler { sequence, frame: Frame::Intrinsic };
            run("from_euler", std::slice::from_ref(&v3), Box::new(move |q| rotation::from_euler(q[0].narrow(), convention).untyped()));
            // Away from the lock, and from α or γ at ±π, where the angles wrap.
            let (a, _, _) = away(&|a, _| {
                let angles = host::euler(*a, convention);
                let pi = std::f64::consts::PI;
                lock_distance(sequence, *a) > 0.01 && angles[0].abs() < pi - 0.1 && angles[2].abs() < pi - 0.1
            }, 0);
            run("euler", &[a], Box::new(move |q| r(q[0]).euler(convention).untyped()));
        }

        // The poses.
        run("pose.compose", &[poses.clone(), poses_b.clone()], Box::new(|q| (p(q[0]) * p(q[1])).untyped()));
        run("pose.transform", &[poses.clone(), v3.clone()], Box::new(|q| (p(q[0]) * q[1].narrow::<Vector<3>>()).untyped()));
        run("pose.inverse", std::slice::from_ref(&poses), Box::new(|q| p(q[0]).inverse().untyped()));
        run("pose.between", &[poses.clone(), poses_b.clone()], Box::new(|q| p(q[0]).between(p(q[1])).untyped()));
        run("pose.rotate", &[poses.clone(), v3.clone()], Box::new(|q| p(q[0]).rotate(q[1].narrow()).untyped()));
        run("pose.untransform", &[poses.clone(), v3.clone()], Box::new(|q| p(q[0]).untransform(q[1].narrow()).untyped()));
        run("pose.matrix", std::slice::from_ref(&poses), Box::new(|q| p(q[0]).matrix().untyped()));
        run("pose.adjoint", std::slice::from_ref(&poses), Box::new(|q| p(q[0]).adjoint().untyped()));
        run("pose.exp", std::slice::from_ref(&xi6), Box::new(|q| pose::exp(q[0].narrow()).untyped()));
        run("pose.log", std::slice::from_ref(&poses), Box::new(|q| p(q[0]).log().untyped()));
        let (a, _, _) = own_w(1);
        run("pose.log_shorter", &[a], Box::new(|q| p(q[0]).log_shorter().untyped()));
        run("pose.jl", std::slice::from_ref(&xi6), Box::new(|q| pose::jl(q[0].narrow()).untyped()));
        run("pose.jl_inv", std::slice::from_ref(&xi6), Box::new(|q| pose::jl_inv(q[0].narrow()).untyped()));
        run("pose.interpolate", &[poses.clone(), poses_b.clone(), t.clone()], Box::new(|q| p(q[0]).interpolate(p(q[1]), q[2].narrow()).untyped()));
        run("pose.interpolate_parts", &[poses.clone(), poses_b.clone(), t.clone()], Box::new(|q| p(q[0]).interpolate_parts(p(q[1]), q[2].narrow()).untyped()));
        let (a, b, tt) = shorter_rows(1);
        run("pose.interpolate_shorter", &[a.clone(), b.clone(), tt.clone()], Box::new(|q| p(q[0]).interpolate_shorter(p(q[1]), q[2].narrow()).untyped()));
        run("pose.interpolate_parts_shorter", &[a, b, tt], Box::new(|q| p(q[0]).interpolate_parts_shorter(p(q[1]), q[2].narrow()).untyped()));

        // The extended poses.
        run("extended_pose.compose", &[extended.clone(), extended_b.clone()], Box::new(|q| (e(q[0]) * e(q[1])).untyped()));
        run("extended_pose.inverse", std::slice::from_ref(&extended), Box::new(|q| e(q[0]).inverse().untyped()));
        run("extended_pose.between", &[extended.clone(), extended_b.clone()], Box::new(|q| e(q[0]).between(e(q[1])).untyped()));
        run("extended_pose.matrix", std::slice::from_ref(&extended), Box::new(|q| e(q[0]).matrix().untyped()));
        run("extended_pose.adjoint", std::slice::from_ref(&extended), Box::new(|q| e(q[0]).adjoint().untyped()));
        run("extended_pose.exp", std::slice::from_ref(&xi9), Box::new(|q| extended_pose::exp(q[0].narrow()).untyped()));
        run("extended_pose.log", std::slice::from_ref(&extended), Box::new(|q| e(q[0]).log().untyped()));
        let (a, _, _) = own_w(2);
        run("extended_pose.log_shorter", &[a], Box::new(|q| e(q[0]).log_shorter().untyped()));
        run("extended_pose.jl", std::slice::from_ref(&xi9), Box::new(|q| extended_pose::jl(q[0].narrow()).untyped()));
        run("extended_pose.jl_inv", std::slice::from_ref(&xi9), Box::new(|q| extended_pose::jl_inv(q[0].narrow()).untyped()));
        run("extended_pose.interpolate", &[extended.clone(), extended_b.clone(), t.clone()], Box::new(|q| e(q[0]).interpolate(e(q[1]), q[2].narrow()).untyped()));
        let (a, b, tt) = shorter_rows(2);
        run("extended_pose.interpolate_shorter", &[a, b, tt], Box::new(|q| e(q[0]).interpolate_shorter(e(q[1]), q[2].narrow()).untyped()));
    }
}

/// The offset from a jump: closer than a central difference's step at the precision, farther than
/// its rounding.
fn offset(target: &Target) -> f64 {
    match target.precision() {
        Precision::Double => 5e-7,
        Precision::Single => 5e-3,
    }
}

/// A unit quaternion whose `w` is `w`, its vector part along a fixed direction.
fn with_w(w: f64, direction: [f64; 3]) -> [f64; 4] {
    let u = common::unit(direction).map(|v| v * (1.0 - w * w).sqrt());
    [w, u[0], u[1], u[2]]
}

/// Holds `map` to `alone`, every pass, at `inputs`.
fn agree(name: &str, inputs: &[Rows], map: Map, alone: Map, target: &Target) {
    assert!(inputs[0].rows() > 0, "{name}: no rows");
    let values: Vec<(Space, HostValue)> = inputs.iter().map(Rows::input).collect();
    if let Err(error) = check_agreement(|q| map(q), |q| alone(q), &values, target) {
        panic!("{name}: {error}");
    }
}

/// Rows of the intrinsic `sequence` on either side of the wrap of angle `k`, `π - 10⁻⁵` and
/// `-π + 10⁻⁵`: the angle is near `π` on one side and near `-π` on the other, the other two agree,
/// and the tangent, cotangent and second-order passes agree across the cut.
fn wrap_is_a_jump_of_the_value_alone(sequence: Sequence, k: usize, middle: f64, target: &Target) {
    let pi = std::f64::consts::PI;
    let side = 1e-5;
    let convention = Euler {
        sequence,
        frame: Frame::Intrinsic,
    };
    let angles = |a: f64| {
        let mut angles = [0.3, middle, 0.3];
        angles[k] = a;
        angles
    };
    let qs = [angles(pi - side), angles(-pi + side)].map(|a| host::from_euler(a, convention));
    let rows = [rows_of(rotation_space(), qs.concat())];
    let name = format!("euler {sequence:?}, angle {k}, across the wrap");
    let tol = common::tolerance(target);

    let values = common::evaluate(|q| r(q[0]).euler(convention).untyped(), &rows, target);
    let (below, above) = (&values[0..3], &values[3..6]);
    assert!(
        (below[k] - (pi - side)).abs() < tol.max(1e-6),
        "{name}: {values:?}"
    );
    assert!(
        (above[k] - (-pi + side)).abs() < tol.max(1e-6),
        "{name}: {values:?}"
    );
    for other in (0..3).filter(|&o| o != k) {
        assert!(
            (below[other] - above[other]).abs() < tol.max(1e-6),
            "{name}: {values:?}"
        );
    }

    // The passes change by the order of the rows' distance, 2·10⁻⁵, and do not jump.
    let across = |name: &str, passes: &[f64]| {
        let n = passes.len() / 2;
        let scale = passes.iter().fold(1.0f64, |m, v| m.max(v.abs()));
        for (a, b) in passes[..n].iter().zip(&passes[n..]) {
            assert!((a - b).abs() < 1e-3 * scale, "{name}: {passes:?}");
        }
    };
    let tangent = common::jacobian(|q| r(q[0]).euler(convention).untyped(), &rows, 0, target);
    across(&format!("{name}, tangent"), &tangent);
    let scalar = |q: &[QDyn]| {
        let weights = constant(Vector::<3>, [0.7, -0.4, 0.9]);
        (r(q[0]).euler(convention) * weights).sum().untyped()
    };
    let (cotangent, second) = common::second_order(scalar, &rows, 0, target);
    across(&format!("{name}, cotangent"), &cotangent);
    across(&format!("{name}, second order"), &second);
}

/// The statement euler's general formula alone, for the intrinsic `sequence`, on the storage of
/// a rotation: the angles on every row off the lock.
fn euler_alone(sequence: Sequence, x: QDyn) -> QDyn {
    let (i, j, l, epsilon) = frame(sequence);
    let at = |a: usize| x.slice(a..a + 1).reshape(Space::Real);
    let (w, ui, uj, ul) = if sequence.is_proper() {
        (at(0), at(1 + i), at(1 + j), at(1 + l))
    } else {
        (
            at(0) - at(1 + j),
            at(1 + i) - at(1 + l) * epsilon,
            at(1 + j) + at(0),
            at(1 + l) + at(1 + i) * epsilon,
        )
    };
    let root = |v: QDyn| (v.log() * 0.5).exp();
    let (na, nb) = (w * w + ui * ui, uj * uj + ul * ul);
    let beta = root(nb).atan2(root(na)) * 2.0;
    let (sigma, delta) = (ui.atan2(w), (ul * epsilon).atan2(uj));
    let wrap = |a: QDyn| a.sin().atan2(a.cos());
    let (alpha, gamma) = (sigma + delta, sigma - delta);
    let (beta, gamma) = if sequence.is_proper() {
        (beta, gamma)
    } else {
        (beta - std::f64::consts::FRAC_PI_2, gamma * -epsilon)
    };
    concatenate(&[wrap(alpha), beta, wrap(gamma)], Space::Vector(3))
}

/// `P(c_m(X))` alone, of a stored `Matrix(3, 3)`.
fn candidate(m: usize, x: QDyn) -> QDyn {
    let at = |i: usize, j: usize| {
        x.slice((i - 1) + 3 * (j - 1)..i + 3 * (j - 1))
            .reshape(Space::Real)
    };
    let (x11, x22, x33) = (at(1, 1), at(2, 2), at(3, 3));
    let c = match m {
        0 => [
            x11 + x22 + x33 + 1.0,
            at(3, 2) - at(2, 3),
            at(1, 3) - at(3, 1),
            at(2, 1) - at(1, 2),
        ],
        1 => [
            at(3, 2) - at(2, 3),
            x11 - x22 - x33 + 1.0,
            at(1, 2) + at(2, 1),
            at(1, 3) + at(3, 1),
        ],
        2 => [
            at(1, 3) - at(3, 1),
            at(1, 2) + at(2, 1),
            x22 - x11 - x33 + 1.0,
            at(2, 3) + at(3, 2),
        ],
        _ => [
            at(2, 1) - at(1, 2),
            at(1, 3) + at(3, 1),
            at(2, 3) + at(3, 2),
            x33 - x11 - x22 + 1.0,
        ],
    };
    project(Rotation, concatenate(&c, Space::Vector(4))).untyped()
}

skoro_core::precision_tests! {
    fn discretizations_jump_cleanly(target) {
        let h = offset(&target);
        let directions = vectors::<3>(4, 601, -1.0, 1.0);
        let starts = random_quaternions(4, 602);
        let rotations = |qs: &[[f64; 4]]| Rows::new(rotation_space(), qs.concat());
        let ts = |rows: usize| reals(rows, 603, 0.2, 0.8);

        // The shorter turn, at w = ±h: Log of q, and of -q.
        for (sign, k) in [(1.0, 0), (-1.0, 0), (1.0, 1), (-1.0, 1), (1.0, 2), (-1.0, 2)] {
            let qs: Vec<[f64; 4]> = directions.iter().map(|d| with_w(sign * h, *d)).collect();
            let space = match k { 0 => rotation_space(), 1 => Pose.space(), _ => ExtendedPose.space() };
            let rows = Rows::new(space, with_translations(&qs, k, 604));
            let (map, alone): (Map, Map) = match (k, sign > 0.0) {
                (0, true) => (Box::new(|q| r(q[0]).log_shorter().untyped()), Box::new(|q| r(q[0]).log().untyped())),
                (0, false) => (Box::new(|q| r(q[0]).log_shorter().untyped()), Box::new(|q| (r(q[0]) * minus()).log().untyped())),
                (1, true) => (Box::new(|q| p(q[0]).log_shorter().untyped()), Box::new(|q| p(q[0]).log().untyped())),
                (1, false) => (Box::new(|q| p(q[0]).log_shorter().untyped()), Box::new(|q| (p(q[0]) * minus_pose()).log().untyped())),
                (_, true) => (Box::new(|q| e(q[0]).log_shorter().untyped()), Box::new(|q| e(q[0]).log().untyped())),
                (_, false) => (Box::new(|q| e(q[0]).log_shorter().untyped()), Box::new(|q| (e(q[0]) * minus_extended()).log().untyped())),
            };
            agree(&format!("log_shorter of {k} translations at w = {}", sign * h), &[rows], map, alone, &target);
        }

        // The shorter interpolations and the angle, at w(q̄₁ q₂) = ±h.
        for sign in [1.0, -1.0] {
            let relative: Vec<[f64; 4]> = directions.iter().map(|d| with_w(sign * h, *d)).collect();
            let ends: Vec<[f64; 4]> = starts.iter().zip(&relative).map(|(a, r)| qmul(*a, *r)).collect();
            let near = sign > 0.0;
            let label = |name: &str| format!("{name} at w = {}", sign * h);
            let rows = [rotations(&starts), rotations(&ends), ts(4)];
            let alone: Map = if near {
                Box::new(|q| r(q[0]).slerp(r(q[1]), q[2].narrow()).untyped())
            } else {
                Box::new(|q| r(q[0]).slerp(r(q[1]) * minus(), q[2].narrow()).untyped())
            };
            agree(&label("slerp_shorter"), &rows, Box::new(|q| r(q[0]).slerp_shorter(r(q[1]), q[2].narrow()).untyped()), alone, &target);
            let alone: Map = if near {
                Box::new(|q| length(r(q[0]).between(r(q[1])).log().untyped()))
            } else {
                Box::new(|q| length((r(q[0]).between(r(q[1])) * minus()).log().untyped()))
            };
            agree(&label("angle"), &rows[..2], Box::new(|q| r(q[0]).angle_to(r(q[1])).untyped()), alone, &target);
            for k in [1, 2] {
                let space = if k == 1 { Pose.space() } else { ExtendedPose.space() };
                let rows = [
                    Rows::new(space.clone(), with_translations(&starts, k, 605)),
                    Rows::new(space, with_translations(&ends, k, 606)),
                    ts(4),
                ];
                if k == 1 {
                    let alone: Map = if near {
                        Box::new(|q| p(q[0]).interpolate(p(q[1]), q[2].narrow()).untyped())
                    } else {
                        Box::new(|q| p(q[0]).interpolate(p(q[1]) * minus_pose(), q[2].narrow()).untyped())
                    };
                    agree(&label("pose.interpolate_shorter"), &rows, Box::new(|q| p(q[0]).interpolate_shorter(p(q[1]), q[2].narrow()).untyped()), alone, &target);
                    let alone: Map = if near {
                        Box::new(|q| p(q[0]).interpolate_parts(p(q[1]), q[2].narrow()).untyped())
                    } else {
                        Box::new(|q| p(q[0]).interpolate_parts(p(q[1]) * minus_pose(), q[2].narrow()).untyped())
                    };
                    agree(&label("pose.interpolate_parts_shorter"), &rows, Box::new(|q| p(q[0]).interpolate_parts_shorter(p(q[1]), q[2].narrow()).untyped()), alone, &target);
                } else {
                    let alone: Map = if near {
                        Box::new(|q| e(q[0]).interpolate(e(q[1]), q[2].narrow()).untyped())
                    } else {
                        Box::new(|q| e(q[0]).interpolate(e(q[1]) * minus_extended(), q[2].narrow()).untyped())
                    };
                    agree(&label("extended_pose.interpolate_shorter"), &rows, Box::new(|q| e(q[0]).interpolate_shorter(e(q[1]), q[2].narrow()).untyped()), alone, &target);
                }
            }
        }

        // Alignment, at ⟨q, r⟩ = ±h: q, and -q.
        for sign in [1.0, -1.0] {
            let relative: Vec<[f64; 4]> = directions.iter().map(|d| with_w(sign * h, *d)).collect();
            let qs: Vec<[f64; 4]> = starts.iter().zip(&relative).map(|(a, r)| qmul(*a, *r)).collect();
            let alone: Map = if sign > 0.0 {
                Box::new(|q| (r(q[0]) * r(q[1]).between(r(q[1]))).untyped())
            } else {
                Box::new(|q| (r(q[0]) * minus() * r(q[1]).between(r(q[1]))).untyped())
            };
            agree("align", &[rotations(&qs), rotations(&starts)], Box::new(|q| r(q[0]).aligned_with(r(q[1])).untyped()), alone, &target);
        }

        // The axis and angle: near the identity, Log's direction and length; at it, finite.
        let near: Vec<[f64; 4]> = directions.iter().map(|d| with_w((1.0 - h * h).sqrt(), *d)).collect();
        let axis_angle: fn(&[QDyn]) -> QDyn = |q| {
            let (axis, angle) = r(q[0]).axis_angle();
            concatenate(&[axis.untyped(), angle.untyped()], Space::Vector(4))
        };
        let alone: Map = Box::new(|q| {
            let v = r(q[0]).log().untyped();
            let n = length(v);
            let axis = ((v * v).sum().log() * -0.5).exp() * v;
            concatenate(&[axis, n], Space::Vector(4))
        });
        agree("axis_angle near the identity", &[rotations(&near)], Box::new(axis_angle), alone, &target);
        let identity = rotations(&[[1.0, 0.0, 0.0, 0.0]]);
        agree("axis_angle at the identity", &[identity], Box::new(axis_angle), Box::new(axis_angle), &target);

        // The reading of a matrix, where its choice changes: the candidate each side selects.
        for (m, other) in [(0, 1), (2, 3), (0, 2), (1, 3)] {
            for (winner, loser) in [(m, other), (other, m)] {
                let qs: Vec<[f64; 4]> = directions
                    .iter()
                    .map(|d| {
                        let mut q = [0.0; 4];
                        let rest: Vec<usize> = (0..4).filter(|c| *c != m && *c != other).collect();
                        q[rest[0]] = 0.2 * d[0];
                        q[rest[1]] = 0.2 * d[1];
                        q[winner] = 0.6 * (1.0 + h);
                        q[loser] = 0.6 * d[2].signum();
                        common::unit(q)
                    })
                    .collect();
                let matrices: Vec<f64> = qs.iter().flat_map(|q| stored(rot(*q))).collect();
                let rows = [Rows::new(Space::Matrix(3, 3), matrices)];
                let alone: Map = Box::new(move |q| candidate(winner, q[0]));
                agree(&format!("from_matrix choosing {winner} over {loser}"), &rows, Box::new(|q| rotation::from_matrix(q[0].narrow()).untyped()), alone, &target);
            }
        }

        // The Euler angles: near the lock, the statement's formula; at an exact lock, finite.
        for sequence in Sequence::ALL {
            let convention = Euler { sequence, frame: Frame::Intrinsic };
            let lock = if sequence.is_proper() { [0.0, std::f64::consts::PI] } else {
                [-std::f64::consts::FRAC_PI_2, std::f64::consts::FRAC_PI_2]
            };
            let near: Vec<[f64; 4]> = lock
                .iter()
                .flat_map(|beta| {
                    let inside = if *beta > 0.0 { -2.0 * h } else { 2.0 * h };
                    [[0.4, beta + inside, -1.1], [-2.0, beta + inside, 0.7]]
                })
                .map(|angles| host::from_euler(angles, convention))
                .collect();
            let map: Map = Box::new(move |q| r(q[0]).euler(convention).untyped());
            let alone: Map = Box::new(move |q| euler_alone(sequence, r(q[0]).include().untyped()));
            agree(&format!("euler {sequence:?} near the lock"), &[rotations(&near)], map, alone, &target);

            // Across the wrap of α and of γ at ±π, away from the lock.
            let middle = if sequence.is_proper() { 1.2 } else { 0.4 };
            let pi = std::f64::consts::PI;
            let wrapped: Vec<[f64; 4]> = [
                [pi - h, middle, 0.3],
                [-pi + h, middle, 0.3],
                [0.3, middle, pi - h],
                [0.3, middle, -pi + h],
            ]
            .into_iter()
            .map(|angles| host::from_euler(angles, convention))
            .collect();
            let map: Map = Box::new(move |q| r(q[0]).euler(convention).untyped());
            let alone: Map = Box::new(move |q| euler_alone(sequence, r(q[0]).include().untyped()));
            agree(&format!("euler {sequence:?} across the wrap"), &[rotations(&wrapped)], map, alone, &target);
            for k in [0, 2] {
                wrap_is_a_jump_of_the_value_alone(sequence, k, middle, &target);
            }

            let (i, j, l, epsilon) = frame(sequence);
            let exact: Vec<[f64; 4]> = if sequence.is_proper() {
                let mut a = [0.6, 0.0, 0.0, 0.0];
                a[1 + i] = 0.8;
                let mut b = [0.0; 4];
                b[1 + j] = 0.6;
                b[1 + l] = 0.8;
                vec![[1.0, 0.0, 0.0, 0.0], a, b]
            } else {
                let mut a = [0.5; 4];
                a[1 + l] = epsilon * 0.5;
                let mut b = [0.5; 4];
                b[1 + j] = -0.5;
                b[1 + l] = -epsilon * 0.5;
                vec![a, b]
            };
            let map: Map = Box::new(move |q| r(q[0]).euler(convention).untyped());
            let again: Map = Box::new(move |q| r(q[0]).euler(convention).untyped());
            agree(&format!("euler {sequence:?} at the lock"), &[rotations(&exact)], map, again, &target);
        }
    }
}
