#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)] // a test crate
//! The three declarations: their boundaries and volumes, the chart, and the registry.

mod common;

use std::f64::consts::PI;
use std::sync::Arc;

use common::host::{hat, identity3, mm, rot};
use common::{Rows, close, quaternions, run, unit, with_translations};
use skoro_core::testing::{check_embedded, check_volume};
use skoro_core::{Embedded, Param, Registry, Space};
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
    use skoro_core::{Matrix, Vector};
    use skoro_spatial::{Rotation, RotationMaps, rotation, so3};
    let (a, b) = (quantity("a", Rotation), quantity("b", Rotation));
    let x = quantity("x", Vector::<3>);
    let m = quantity("m", Matrix::<3, 4>);
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
    let _ = (rotation::exp(x), a.log(), a.log_shorter());
    let _ = (so3::jl(x), so3::jr(x), so3::jl_inv(x), so3::jr_inv(x));
    let angle = quantity("angle", skoro_core::Real);
    let _ = (
        rotation::about(x, angle),
        rotation::elementary(skoro_spatial::Axis::Y, angle),
    );
    let _ = (
        a.axis_angle(),
        a.aligned_with(b),
        rotation::from_two_vectors(x, x),
    );
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
fn chart_is_stereographic() {
    let target = skoro_tensor::Target::host();
    let x = unit([0.4, -0.3, 0.8, 0.2]);
    let retract = |x: [f64; 4], t: [f64; 3]| {
        run(
            ROTATION.retraction(),
            &[Rows::vectors(&[x]), Rows::vectors(&[t])],
            &target,
        )
    };
    let direction = unit([0.6, 0.2, -0.7]);

    // Q(x, R(x, t)) = t, out to lengths of a thousand.
    for length in [1e-3, 0.1, 1.0, 10.0, 100.0, 1000.0] {
        let t = direction.map(|v| v * length);
        let y = retract(x, t);
        let back = run(
            ROTATION.inverse(),
            &[Rows::vectors(&[x]), Rows::new(Space::Vector(4), y)],
            &target,
        );
        close(&back, &t, 1e-9);
    }

    // The angle of δ(t) is 4 arctan(|t| / 4).
    for length in [0.0, 0.5, 4.0, 40.0] {
        let d = retract([1.0, 0.0, 0.0, 0.0], direction.map(|v| v * length));
        let n = (d[1] * d[1] + d[2] * d[2] + d[3] * d[3]).sqrt();
        let angle = 2.0 * n.atan2(d[0]);
        close(&[angle], &[4.0 * (length / 4.0).atan()], 1e-12);
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
}

#[test]
fn volume_has_the_stated_mass() {
    // ∫ e^{λ(t)} dt over ℝ³ in polar coordinates, with r = 4 tan θ for θ in (0, π/2): the area of
    // the unit sphere, 4π, times ∫ r² e^{λ(r)} dr/dθ dθ.
    let target = skoro_tensor::Target::host();
    let steps = 4000;
    let direction = unit([0.2, 0.9, -0.4]);
    let thetas: Vec<f64> = (0..steps)
        .map(|i| (i as f64 + 0.5) * (PI / 2.0) / steps as f64)
        .collect();
    let radii: Vec<f64> = thetas.iter().map(|theta| 4.0 * theta.tan()).collect();
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
        .map(|((theta, r), lambda)| r * r * lambda.exp() * 4.0 / theta.cos().powi(2) * h)
        .sum();
    close(&[4.0 * PI * radial], &[16.0 * PI * PI], 1e-10);
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
