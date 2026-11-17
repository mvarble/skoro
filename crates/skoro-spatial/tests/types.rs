#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)] // a test crate
//! The three declarations: their boundaries and volumes, the chart, and the registry.

mod common;

use std::f64::consts::PI;
use std::sync::Arc;

use common::host::{hat, identity3, mm, rot};
use common::{Rows, close, quaternions, run, unit, with_translations};
use skoro_core::testing::{check_embedded, check_mass, check_volume};
use skoro_core::{Embedded, Error, Key, Param, Registry, Space};
use skoro_spatial::{EXTENDED_POSE, POSE, ROTATION, register};

fn declarations() -> [(&'static Arc<Embedded>, &'static str, usize, usize); 3] {
    [
        (&ROTATION, "skoro.spatial.rotation", 4, 3),
        (&POSE, "skoro.spatial.pose", 7, 6),
        (&EXTENDED_POSE, "skoro.spatial.extended_pose", 10, 9),
    ]
}

#[test]
fn types_are_declared() {
    for (declaration, name, n, d) in declarations() {
        assert_eq!(declaration.name(), name);
        assert_eq!((declaration.storage(), declaration.dimension()), (n, d));
        assert!(declaration.has_projection() && declaration.volume().is_some());
    }
    let mut registry = Registry::core();
    register(&mut registry).unwrap();
    for (declaration, name, _, _) in declarations() {
        assert_eq!(*registry.embedded(name, &[]).unwrap(), **declaration);
        assert!(registry.embedded(name, &[Param::Integer(1)]).is_err());
    }
    assert!(register(&mut registry).is_err(), "registered twice");

    // Every map of the crate, applied once, defines every composite.
    let mut writer = skoro_core::Writer::new();
    writer.build(every_map).unwrap();
}

/// Applies every map of the crate once.
fn every_map() {
    use skoro_core::scope::quantity;
    use skoro_core::{Matrix, Real, Vector};
    use skoro_spatial::{
        Axis, Euler, ExtendedPose, ExtendedPoseMaps, Frame, Pose, PoseMaps, Rotation, RotationMaps,
        Sequence, extended_pose, pose, rotation, so3,
    };
    let (a, b) = (quantity("a", Rotation), quantity("b", Rotation));
    let x = quantity("x", Vector::<3>);
    let m = quantity("m", Matrix::<3, 4>);
    let angle = quantity("angle", Real);
    let _ = (
        a * b,
        a * x,
        a * x.widen(),
        a * m,
        a.inverse(),
        a.between(b),
    );
    let _ = (a.unrotate(x), a.matrix(), a.adjoint(), rotation::identity());
    let _ = so3::vee(so3::hat(x));
    let _ = (rotation::exp(x), a.log());
    let _ = (so3::jl(x), so3::jr(x), so3::jl_inv(x), so3::jr_inv(x));
    let _ = (
        rotation::about(x, angle),
        rotation::elementary(Axis::Y, angle),
    );
    let _ = (
        a.axis_angle(),
        a.aligned_with(b),
        rotation::from_two_vectors(x, x),
    );
    let _ = rotation::from_matrix(a.matrix());
    let _ = (a.angle_to(b), a.slerp(b, angle));
    for sequence in Sequence::ALL {
        for frame in [Frame::Intrinsic, Frame::Extrinsic] {
            let convention = Euler { sequence, frame };
            let _ = (a.euler(convention), rotation::from_euler(x, convention));
        }
    }

    let (s, t) = (quantity("s", Pose), quantity("t", Pose));
    let _ = (
        s * t,
        s * x,
        s.inverse(),
        s.between(t),
        s.rotate(x),
        s.untransform(x),
    );
    let _ = (s.matrix(), s.adjoint(), pose::from_product(s.as_product()));
    let _ = (
        pose::identity(),
        pose::from_parts(s.rotation(), s.position()),
    );
    let xi6 = quantity("xi6", Vector::<6>);
    let _ = (pose::exp(xi6), s.log());
    let _ = (s.interpolate(t, angle), s.interpolate_parts(t, angle));
    let _ = (
        pose::jl(xi6),
        pose::jr(xi6),
        pose::jl_inv(xi6),
        pose::jr_inv(xi6),
    );

    let (u, v) = (quantity("u", ExtendedPose), quantity("v", ExtendedPose));
    let _ = (
        u * v,
        u.inverse(),
        u.between(v),
        u.matrix(),
        u.adjoint(),
        u.pose(),
    );
    let _ = extended_pose::from_parts(u.rotation(), u.velocity(), u.position());
    let _ = extended_pose::identity();
    let xi9 = quantity("xi9", Vector::<9>);
    let _ = (extended_pose::exp(xi9), u.log());
    let _ = (extended_pose::jl(xi9), extended_pose::jr(xi9));
    let _ = (extended_pose::jl_inv(xi9), extended_pose::jr_inv(xi9));
    let _ = u.interpolate(v, angle);
}

/// The points of a type with `k` translations: the fixed rows and random ones.
fn points(k: usize) -> Rows {
    let n = 4 + 3 * k;
    Rows::new(
        Space::Vector(n),
        with_translations(&quaternions(4, 11), k, 12),
    )
}

skoro_core::precision_tests! {
    fn types_pass_the_core_checks(target) {
        for (k, (declaration, ..)) in declarations().into_iter().enumerate() {
            let points = points(k).host();
            if let Err(error) = check_embedded(declaration, &points, &target) {
                panic!("{}: {error}", declaration.name());
            }
            if let Err(error) = check_volume(declaration, &points, &target) {
                panic!("{}: {error}", declaration.name());
            }
        }
    }
}

#[test]
fn chart_is_cayley() {
    let target = skoro_tensor::Target::host();
    let x = unit([0.4, -0.3, 0.8, 0.2]);
    let minus = |q: [f64; 4]| q.map(|v| -v);
    let retract = |x: [f64; 4], t: [f64; 3]| {
        run(
            ROTATION.retraction(),
            &[Rows::vectors(&[x]), Rows::vectors(&[t])],
            &target,
        )
    };
    let local = |x: [f64; 4], y: [f64; 4]| {
        run(
            ROTATION.inverse(),
            &[Rows::vectors(&[x]), Rows::vectors(&[y])],
            &target,
        )
    };
    let quaternion = |y: Vec<f64>| [y[0], y[1], y[2], y[3]];
    let direction = unit([0.6, 0.2, -0.7]);

    // Q(x, R(x, t)) = t, out to a length of 10; the chart reaches every rotation by less than a
    // half-turn, and Q reads neither sign.
    for length in [1e-3, 0.1, 1.0, 3.0, 10.0] {
        let t = direction.map(|v| v * length);
        let y = quaternion(retract(x, t));
        close(&local(x, y), &t, 1e-9);
        close(&local(minus(x), y), &t, 1e-9);
        close(&local(x, minus(y)), &t, 1e-9);
        // (-x) ⊕ t = -(x ⊕ t).
        close(&retract(minus(x), t), &minus(y), 1e-14);
    }

    // The angle of γ(t) is 2 arctan(|t| / 2).
    for length in [0.0, 0.5, 4.0, 40.0] {
        let d = retract([1.0, 0.0, 0.0, 0.0], direction.map(|v| v * length));
        let n = (d[1] * d[1] + d[2] * d[2] + d[3] * d[3]).sqrt();
        let angle = 2.0 * n.atan2(d[0]);
        close(&[angle], &[2.0 * (length / 2.0).atan()], 1e-12);
    }

    // π(x ⊕ t) = π(x) (I + [t]×) + O(|t|²): the coordinates are the rotation vector in the frame of
    // x, to first order.
    let t = direction.map(|v| v * 1e-3);
    let y = retract(x, t);
    let moved = rot([y[0], y[1], y[2], y[3]]);
    let first_order = mm(rot(x), {
        let (mut a, h) = (identity3(), hat(t));
        for i in 0..3 {
            for j in 0..3 {
                a[i][j] += h[i][j];
            }
        }
        a
    });
    for i in 0..3 {
        for j in 0..3 {
            assert!((moved[i][j] - first_order[i][j]).abs() < 2e-6, "{moved:?}");
        }
    }

    // The base point is normalized.
    let t = [0.3, -1.2, 0.5];
    close(&retract(x.map(|v| 1.01 * v), t), &retract(x, t), 1e-14);

    // The cut: Q is non-finite exactly at a half-turn, at either precision, and of length
    // 2 tan(φ/2) short of it.
    for precision in [
        skoro_tensor::Precision::Single,
        skoro_tensor::Precision::Double,
    ] {
        let target = skoro_core::testing::flex(precision).unwrap();
        let at_cut = run(
            ROTATION.inverse(),
            &[
                Rows::vectors(&[[1.0, 0.0, 0.0, 0.0]]),
                Rows::vectors(&[[0.0, 1.0, 0.0, 0.0]]),
            ],
            &target,
        );
        assert!(at_cut.iter().any(|v| !v.is_finite()), "{at_cut:?}");
    }
    let phi: f64 = 3.0;
    let near = local(
        [1.0, 0.0, 0.0, 0.0],
        [(phi / 2.0).cos(), (phi / 2.0).sin(), 0.0, 0.0],
    );
    close(&near, &[2.0 * (phi / 2.0).tan(), 0.0, 0.0], 1e-12);
}

#[test]
fn volume_has_the_stated_mass() {
    // ∫ e^{λ(t)} dt over ℝ³ in polar coordinates, with r = 2 tan θ for θ in (0, π/2): the area of
    // the unit sphere, 4π, times ∫ r² e^{λ(r)} dr/dθ dθ. The mass of the Haar measure of SO(3)
    // normalized at the chart's centre is 8π², half that of the sphere S³ it is stored on.
    let target = skoro_tensor::Target::host();
    let steps = 4000;
    let direction = unit([0.2, 0.9, -0.4]);
    let thetas: Vec<f64> = (0..steps)
        .map(|i| (i as f64 + 0.5) * (PI / 2.0) / steps as f64)
        .collect();
    let radii: Vec<f64> = thetas.iter().map(|theta| 2.0 * theta.tan()).collect();
    let tangents: Vec<[f64; 3]> = radii.iter().map(|r| direction.map(|v| v * r)).collect();
    let points = vec![unit([0.5, 0.1, -0.3, 0.7]); steps];
    let lambda = run(
        ROTATION.volume().unwrap().log_volume(),
        &[Rows::vectors(&points), Rows::vectors(&tangents)],
        &target,
    );
    let h = (PI / 2.0) / steps as f64;
    let radial: f64 = thetas
        .iter()
        .zip(&radii)
        .zip(&lambda)
        .map(|((theta, r), lambda)| r * r * lambda.exp() * 2.0 / theta.cos().powi(2) * h)
        .sum();
    // The total mass of the chart's volume, against the mass the type declares.
    let declared = ROTATION.mass().unwrap().log_mass().exp();
    close(&[4.0 * PI * radial], &[declared], 1e-10);
    assert_eq!(ROTATION.mass().unwrap().log_mass(), (8.0 * PI * PI).ln());
}

/// The draw, the Cayley volume and the mass are one law, at the points the volume is checked at.
#[test]
fn rotation_mass_checks() {
    if let Err(error) = check_mass(&ROTATION, &points(0).host(), 1.0) {
        panic!("{error}");
    }
}

#[test]
fn uniform_rotation_is_shoemakes_map() {
    let draw = ROTATION.mass().unwrap().draw();
    let mut q = [f64::NAN; 4];
    draw(&[0.25, 0.5, 0.75], &mut q);
    // (√(1 - u₁) cos 2πu₂, √(1 - u₁) sin 2πu₂, √u₁ cos 2πu₃, √u₁ sin 2πu₃), as (w, x, y, z).
    close(&q, &[-0.75f64.sqrt(), 0.0, 0.0, -0.5], 1e-15);
    let grid = [1e-9, 0.1, 0.25, 0.5, 0.8, 1.0 - 1e-9];
    for u1 in grid {
        for u2 in grid {
            for u3 in grid {
                draw(&[u1, u2, u3], &mut q);
                let norm = q.iter().map(|v| v * v).sum::<f64>().sqrt();
                assert!((norm - 1.0).abs() < 1e-15, "{q:?} at ({u1}, {u2}, {u3})");
            }
        }
    }
}

/// From a centre that is not the identity, the angle of a uniform rotation has the Haar law,
/// `(θ - sin θ) / π` on `(0, π)`, and its axis no preferred direction.
#[test]
fn uniform_rotations_have_the_haar_angle() {
    let draws = 65_536;
    let draw = ROTATION.mass().unwrap().draw();
    let drawn: Vec<[f64; 4]> = Key::from_seed(31)
        .split(draws)
        .iter()
        .map(|key| {
            let mut q = [0.0; 4];
            draw(&key.uniforms(3), &mut q);
            q
        })
        .collect();
    let centre = vec![unit([0.5, 0.1, -0.3, 0.7]); draws];
    // x ⊖ μ, by the chart's route: Q(μ, x).
    let coordinates = run(
        ROTATION.inverse(),
        &[Rows::vectors(&centre), Rows::vectors(&drawn)],
        &skoro_tensor::Target::host(),
    );
    let count = draws as f64;
    let mut angles = Vec::with_capacity(draws);
    let mut direction = [0.0; 3];
    for t in coordinates.chunks(3) {
        let length = t.iter().map(|v| v * v).sum::<f64>().sqrt();
        assert!(length.is_finite() && length > 0.0, "{t:?}");
        angles.push(2.0 * (length / 2.0).atan());
        for (sum, v) in direction.iter_mut().zip(t) {
            *sum += v / length / count;
        }
    }
    // Kolmogorov's statistic against the Haar law, at the 1 % level.
    angles.sort_by(f64::total_cmp);
    let distance = angles
        .iter()
        .enumerate()
        .map(|(i, theta)| {
            let law = (theta - theta.sin()) / PI;
            (law - i as f64 / count)
                .abs()
                .max((law - (i + 1) as f64 / count).abs())
        })
        .fold(0.0, f64::max);
    assert!(
        distance <= 1.63 / count.sqrt(),
        "the angles are {distance} from the Haar law"
    );
    // A component of a uniform direction has variance 1/3.
    for mean in direction {
        assert!(mean.abs() <= 4.0 / (3.0 * count).sqrt(), "{direction:?}");
    }
}

#[test]
fn poses_have_no_mass() {
    for declaration in [&POSE, &EXTENDED_POSE] {
        assert!(declaration.mass().is_none());
        let space = Space::Embedded((*declaration).clone());
        let no_mass = Error::NoMass {
            space: space.clone(),
        };
        assert_eq!(space.log_mass(), Err(no_mass.clone()));
        let mut point = vec![0.0; space.storage()];
        let variates = vec![0.5; space.dimension().unwrap()];
        assert_eq!(space.draw_uniform(&variates, &mut point), Err(no_mass));
    }
    let rotation = Space::Embedded(ROTATION.clone());
    assert_eq!(rotation.log_mass(), Ok((8.0 * PI * PI).ln()));
}

#[test]
fn host_values_are_normalized_or_refused() {
    use skoro_core::IntoHostValue;
    use skoro_spatial::{ExtendedPose, Pose, Rotation};
    use skoro_tensor::HostData;
    let data = |value: Option<skoro_tensor::HostValue>| match value.unwrap().data() {
        HostData::Continuous(data) => data.clone(),
        HostData::Discrete(_) => panic!("a continuous value"),
    };
    let rotation = IntoHostValue::<Rotation>::into_host_value([0.0, 3.0, 0.0, 4.0]);
    close(&data(rotation), &[0.0, 0.6, 0.0, 0.8], 1e-15);
    assert!(IntoHostValue::<Rotation>::into_host_value([0.0; 4]).is_none());
    assert!(IntoHostValue::<Rotation>::into_host_value([f64::NAN, 1.0, 0.0, 0.0]).is_none());
    let pose = IntoHostValue::<Pose>::into_host_value(([2.0, 0.0, 0.0, 0.0], [1.0, 2.0, 3.0]));
    close(&data(pose), &[1.0, 0.0, 0.0, 0.0, 1.0, 2.0, 3.0], 1e-15);
    let stored = [0.0, 0.0, 2.0, 0.0, 1.0, 2.0, 3.0];
    close(
        &data(IntoHostValue::<Pose>::into_host_value(stored)),
        &[0.0, 0.0, 1.0, 0.0, 1.0, 2.0, 3.0],
        1e-15,
    );
    let infinite = ([1.0, 0.0, 0.0, 0.0], [f64::INFINITY, 0.0, 0.0]);
    assert!(IntoHostValue::<Pose>::into_host_value(infinite).is_none());
    let extended = IntoHostValue::<ExtendedPose>::into_host_value((
        [0.0, 0.0, 0.0, -5.0],
        [1.0, 2.0, 3.0],
        [4.0, 5.0, 6.0],
    ));
    close(
        &data(extended),
        &[0.0, 0.0, 0.0, -1.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        1e-15,
    );
    let stored = [1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 7.0, 8.0, 9.0];
    close(
        &data(IntoHostValue::<ExtendedPose>::into_host_value(stored)),
        &[0.5, 0.5, 0.5, 0.5, 0.0, 0.0, 0.0, 7.0, 8.0, 9.0],
        1e-15,
    );
}
