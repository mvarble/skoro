#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)] // a test crate
//! The sphere of the acceptance test, declared from `skoro`'s public interface alone.

mod common;

use std::sync::Arc;

use common::sphere::{
    EXPONENTIAL, Near, SPHERE, Sphere, ball_of_radius_pi, basis, exponential_log_volume,
    exponential_sphere, measured_exponential_sphere, points,
};
use skoro::scope::{project, quantity};
use skoro::tensor::{Frame, HostData, HostValue, Value};
use skoro::testing::{check_agreement, check_embedded, check_operation, check_volume};
use skoro::{
    Composite, Embedded, Frames, Q, QDyn, Query, Space, Target, Values, Vector, VectorDyn, Volume,
    Writer,
};

fn value(data: &[f64], rows: usize, target: &Target) -> Value {
    HostValue::new(
        Some(rows),
        data.len() / rows,
        HostData::Continuous(data.to_vec()),
    )
    .unwrap()
    .load(rows, target)
    .unwrap()
    .unwrap()
}

fn host(values: &Values, q: impl Into<skoro::NodeId>) -> Vec<f64> {
    match values.get(q).unwrap() {
        Value::Continuous(points) => points.to_host(),
        Value::Discrete(_) => panic!("a continuous value"),
    }
}

#[test]
fn sphere_basis_is_the_derivative_of_its_retraction() {
    let target = Target::host();
    let program = SPHERE.retraction().compile(&target).unwrap();
    let free = program.free();
    let xs = points(4);
    let flat: Vec<f64> = xs.iter().flatten().copied().collect();
    let at = Values::new()
        .with(free[0].node, value(&flat, 4, &target))
        .with(free[1].node, value(&[0.0; 8], 4, &target));
    let identity: Vec<f64> = (0..4).flat_map(|_| [1.0, 0.0, 0.0, 1.0]).collect();
    let frames = Frames::new().with(free[1].node, Frame::from_host(&identity, 4, 2, 2, &target));
    let (_, tangent) = program.tangent(&at, &frames).unwrap();
    let b = tangent.get(program.targets()[0]).unwrap().to_host();
    for (row, x) in xs.iter().enumerate() {
        let expected = basis(*x);
        let at = |i: usize, c: usize| b[(row * 3 + i) * 2 + c];
        for (i, expected) in expected.iter().enumerate() {
            for (c, expected) in expected.iter().enumerate() {
                assert!((at(i, c) - expected).abs() < 1e-12, "B({x:?})");
            }
        }
        for (a, c) in [(0, 0), (0, 1), (1, 1)] {
            let product: f64 = (0..3).map(|i| at(i, a) * at(i, c)).sum();
            let identity = if a == c { 1.0 } else { 0.0 };
            assert!((product - identity).abs() < 1e-12, "orthonormal columns");
        }
        for c in 0..2 {
            let normal: f64 = (0..3).map(|i| x[i] * at(i, c)).sum();
            assert!(normal.abs() < 1e-12, "columns orthogonal to x");
        }
    }
}

#[test]
fn sphere_retraction_reaches_the_far_side() {
    let target = Target::host();
    let mut writer = Writer::new();
    let (s, t, y, back) = writer
        .build(|| {
            let s = quantity("s", Sphere);
            let t = quantity("t", VectorDyn(2));
            let y = s.retract(t);
            (s, t, y, y.local(s))
        })
        .unwrap();
    let program = writer
        .snapshot()
        .compile(&Query::of([QDyn::from(y), QDyn::from(back)]), &target)
        .unwrap();
    let x = points(1)[0];
    let at = Values::new()
        .with(s, value(&x, 1, &target))
        .with(t, value(&[6.0, 8.0], 1, &target));
    let out = program.forward(&at).unwrap();
    let y = host(&out, y);
    let norm: f64 = y.iter().map(|v| v * v).sum::<f64>().sqrt();
    assert!((norm - 1.0).abs() < 1e-12, "on the sphere: {y:?}");
    let back = host(&out, back);
    assert!(
        (back[0] - 6.0).abs() < 1e-9 && (back[1] - 8.0).abs() < 1e-9,
        "{back:?}"
    );
}

#[test]
fn removable_singularity_is_reported() {
    let target = Target::host();
    let rows = 3;
    let flat: Vec<f64> = points(rows).iter().flatten().copied().collect();
    let points = HostValue::new(Some(rows), 3, HostData::Continuous(flat)).unwrap();

    let closed = check_embedded(&exponential_sphere(Near::Closed), &points, &target).unwrap_err();
    assert!(
        closed.contains("skoro.retract") && closed.contains("pass is not finite"),
        "{closed}"
    );

    let selected =
        check_embedded(&exponential_sphere(Near::Selected), &points, &target).unwrap_err();
    assert!(
        selected.contains("cotangent pass is not finite"),
        "{selected}"
    );

    if let Err(error) = check_embedded(&exponential_sphere(Near::Guarded), &points, &target) {
        panic!("{error}");
    }
}

/// The points of `points(rows)`, as a host value of one row each.
fn sphere_points(rows: usize) -> HostValue {
    let data: Vec<f64> = points(rows).into_iter().flatten().collect();
    HostValue::new(Some(rows), 3, HostData::Continuous(data)).unwrap()
}

/// Tangent coordinates of moderate size, one row each.
fn tangents(rows: usize) -> HostValue {
    let data: Vec<f64> = (0..2 * rows)
        .map(|i| 0.6 * (1.7 * i as f64).sin())
        .collect();
    HostValue::new(Some(rows), 2, HostData::Continuous(data)).unwrap()
}

/// The normalized midpoint of two points of the sphere: computed in storage, projected back.
fn midpoint(a: Q<Sphere>, b: Q<Sphere>) -> Q<Sphere> {
    project(Sphere, a.include() + b.include())
}

skoro::precision_tests! {
    fn sphere_passes_check_volume(target) {
        if let Err(error) = check_volume(&SPHERE, &sphere_points(4), &target) {
            panic!("{error}");
        }
    }

    fn sphere_passes_check_embedded(target) {
        if let Err(error) = check_embedded(&SPHERE, &sphere_points(4), &target) {
            panic!("{error}");
        }
    }

    fn sphere_projection_agrees_with_its_retraction(target) {
        let inputs = [
            (Space::Embedded(SPHERE.clone()), sphere_points(3)),
            (Space::Vector(2), tangents(3)),
        ];
        let checked = check_agreement(
            |q| q[0].retract(q[1]),
            |q| {
                let stored = SPHERE.retraction().call(&[q[0].include(), q[1]]);
                project(Sphere, stored).untyped()
            },
            &inputs,
            &target,
        );
        if let Err(error) = checked {
            panic!("{error}");
        }
    }

    fn midpoint_is_a_map_into_the_sphere(target) {
        let sphere = Space::Embedded(SPHERE.clone());
        let composite = Composite::define(
            "test.sphere.midpoint",
            Vec::new(),
            &[sphere.clone(), sphere.clone()],
            |q| midpoint(q[0].narrow::<Sphere>(), q[1].narrow::<Sphere>()).untyped(),
        )
        .unwrap();
        let mut others = points(3);
        others.rotate_left(1);
        let others: Vec<f64> = others.into_iter().flatten().collect();
        let others = HostValue::new(Some(3), 3, HostData::Continuous(others)).unwrap();
        let inputs = [(sphere.clone(), sphere_points(3)), (sphere, others)];
        if let Err(error) = check_operation(&*composite.operation(), &inputs, &target) {
            panic!("{error}");
        }
    }
}

#[test]
fn projection_returns_drift_to_the_sphere() {
    let target = Target::host();
    let mut writer = Writer::new();
    let (w, y) = writer
        .build(|| {
            let w = quantity("w", Vector::<3>);
            (w, project(Sphere, w))
        })
        .unwrap();
    let program = writer
        .snapshot()
        .compile(&Query::of([QDyn::from(y)]), &target)
        .unwrap();
    let xs = points(4);
    let drifted: Vec<f64> = xs.iter().flatten().map(|x| x * (1.0 + 1e-6)).collect();
    let values = program
        .forward(&Values::new().with(w, value(&drifted, 4, &target)))
        .unwrap();
    let projected = host(&values, y);
    for (row, x) in xs.iter().enumerate() {
        let p = &projected[row * 3..row * 3 + 3];
        let norm: f64 = p.iter().map(|v| v * v).sum::<f64>().sqrt();
        assert!((norm - 1.0).abs() < 1e-14, "|P(w)| = {norm}");
        for (p, x) in p.iter().zip(x) {
            assert!((p - x).abs() < 1e-14, "P(w) = {p:?}, from {x:?}");
        }
    }
}

/// The points of `points(rows)`, each moved to the next row's: pairs that are never antipodal.
fn other_points(rows: usize) -> HostValue {
    let mut others = points(rows);
    others.rotate_left(1);
    let others: Vec<f64> = others.into_iter().flatten().collect();
    HostValue::new(Some(rows), 3, HostData::Continuous(others)).unwrap()
}

#[test]
fn sphere_log_volume_in_storage() {
    let sphere = Space::Embedded(SPHERE.clone());
    let inputs = [
        (sphere.clone(), sphere_points(5)),
        (sphere, other_points(5)),
    ];
    // Λ(x, y ⊖ x) = 2 log((1 + xᵀ y) / 2).
    check_agreement(
        |q| q[0].log_volume(q[1].local(q[0])),
        |q| (((q[0].include() * q[1].include()).sum() + 1.0) * 0.5).log() * 2.0,
        &inputs,
        &Target::host(),
    )
    .unwrap();
}

skoro::precision_tests! {
    fn exponential_sphere_passes_check_volume(target) {
        if let Err(error) = check_volume(&EXPONENTIAL, &sphere_points(4), &target) {
            panic!("{error}");
        }
    }
}

/// The sphere's retraction with `log_volume` declared in place of its own.
fn stereographic_sphere(name: &'static str, log_volume: Arc<Composite>) -> Arc<Embedded> {
    Embedded::declare(
        name,
        Vec::new(),
        3,
        2,
        common::sphere::retraction(),
        common::sphere::inverse(),
        None,
    )
    .and_then(|sphere| sphere.with_volume(Volume::global(log_volume)))
    .unwrap()
}

fn wrong_log_volume(name: &'static str, body: fn(QDyn) -> QDyn) -> Arc<Composite> {
    Composite::define(
        name,
        Vec::new(),
        &[Space::Vector(3), Space::Vector(2)],
        move |q| body((q[1] * q[1]).sum()),
    )
    .unwrap()
}

#[test]
fn check_volume_reports_a_wrong_declaration() {
    let (points, target) = (sphere_points(3), Target::host());
    let fails = |declaration: Arc<Embedded>, expected: &str| {
        let error = check_volume(&declaration, &points, &target).unwrap_err();
        assert!(error.contains(expected), "{}: {error}", declaration.name());
    };
    // Off by a constant: the chart's volume at its centre is not 1.
    let shifted = wrong_log_volume("sphere.log_volume.shifted", |tt| {
        (tt * 0.25 + 1.0).log() * -2.0 + 0.1
    });
    fails(
        stereographic_sphere("sphere.shifted", shifted),
        "Λ(x, 0) is not 0",
    );
    // The normalization chart's volume, on the stereographic chart.
    let gnomonic = wrong_log_volume("sphere.log_volume.gnomonic", |tt| (tt + 1.0).log() * -1.5);
    fails(
        stereographic_sphere("sphere.gnomonic", gnomonic),
        "the area formula",
    );
    // The exponential chart declared global: its inverse fails beyond π.
    fails(
        measured_exponential_sphere(
            "sphere.exp.global",
            Volume::global(exponential_log_volume(true)),
        ),
        "Q(x, R(x, t)) is not t",
    );
    // Its log-volume as written: 0 / 0 at t = 0.
    fails(
        measured_exponential_sphere(
            "sphere.exp.unguarded",
            Volume::within(exponential_log_volume(false), ball_of_radius_pi()),
        ),
        "is not finite",
    );
}

/// `Λ(x, r e₁)` of `declaration` at each radius, at the point `x`, on the host.
fn along_a_ray(declaration: &Embedded, x: [f64; 3], radii: &[f64]) -> Vec<f64> {
    let target = Target::host();
    let rows = radii.len();
    let program = declaration
        .volume()
        .unwrap()
        .log_volume()
        .compile(&target)
        .unwrap();
    let free = program.free();
    let xs: Vec<f64> = (0..rows).flat_map(|_| x).collect();
    let ts: Vec<f64> = radii.iter().flat_map(|&r| [r, 0.0]).collect();
    let at = Values::new()
        .with(free[0].node, value(&xs, rows, &target))
        .with(free[1].node, value(&ts, rows, &target));
    let out = program.forward(&at).unwrap();
    match out.get(program.targets()[0]).unwrap() {
        Value::Continuous(points) => points.to_host(),
        Value::Discrete(_) => panic!("a log-volume is real"),
    }
}

/// The trapezoidal rule for `f` on `[a, b]` with `n` intervals.
fn trapezoid(f: impl Fn(&[f64]) -> Vec<f64>, a: f64, b: f64, n: usize) -> f64 {
    let h = (b - a) / n as f64;
    let nodes: Vec<f64> = (0..=n).map(|i| a + h * i as f64).collect();
    let values = f(&nodes);
    let ends = 0.5 * (values[0] + values[n]);
    h * (ends + values[1..n].iter().sum::<f64>())
}

#[test]
fn sphere_charts_have_the_mass_of_the_sphere() {
    let four_pi = 4.0 * std::f64::consts::PI;
    for x in points(3) {
        // The stereographic chart over all of ℝ², in u = atan(r / 2): r = 2 tan u, dr = 2 sec²u du.
        let stereographic = trapezoid(
            |us| {
                let u_max = std::f64::consts::FRAC_PI_2;
                let us: Vec<f64> = us.iter().map(|u| u.min(u_max - 1e-9)).collect();
                let radii: Vec<f64> = us.iter().map(|u| 2.0 * u.tan()).collect();
                let lambda = along_a_ray(&SPHERE, x, &radii);
                us.iter()
                    .zip(&radii)
                    .zip(lambda)
                    .map(|((u, r), l)| {
                        2.0 * std::f64::consts::PI * r * l.exp() * 2.0 / u.cos().powi(2)
                    })
                    .collect()
            },
            0.0,
            std::f64::consts::FRAC_PI_2,
            20_000,
        );
        assert!(
            (stereographic - four_pi).abs() < 1e-6,
            "the stereographic chart's mass is {stereographic}"
        );
        // The exponential chart over the ball of radius π.
        let exponential = trapezoid(
            |radii| {
                let lambda = along_a_ray(&EXPONENTIAL, x, radii);
                radii
                    .iter()
                    .zip(lambda)
                    .map(|(r, l)| 2.0 * std::f64::consts::PI * r * l.exp())
                    .collect()
            },
            0.0,
            std::f64::consts::PI,
            20_000,
        );
        assert!(
            (exponential - four_pi).abs() < 1e-6,
            "the exponential chart's mass is {exponential}"
        );
    }
}

#[test]
fn exponential_domain_ends_at_the_antipode() {
    let target = Target::host();
    let program = ball_of_radius_pi().compile(&target).unwrap();
    let free = program.free();
    let pi = std::f64::consts::PI;
    let lengths = [0.0, 1.0, pi - 1e-6, pi + 1e-6, 4.0, 10.0];
    let expected = [1, 1, 1, 0, 0, 0];
    for (k, angle) in [0.0, 0.7, 2.9].into_iter().enumerate() {
        let rows = lengths.len();
        let xs: Vec<f64> = (0..rows).flat_map(|_| points(3)[k]).collect();
        let ts: Vec<f64> = lengths
            .iter()
            .flat_map(|l| [l * f64::cos(angle), l * f64::sin(angle)])
            .collect();
        let at = Values::new()
            .with(free[0].node, value(&xs, rows, &target))
            .with(free[1].node, value(&ts, rows, &target));
        let out = program.forward(&at).unwrap();
        let Some(Value::Discrete(inside)) = out.get(program.targets()[0]) else {
            panic!("a domain is a boolean");
        };
        assert_eq!(inside.to_host(), expected, "at the angle {angle}");
    }
}

#[test]
fn pushed_density_integrates_to_one_on_the_sphere() {
    let target = Target::host();
    let mut writer = Writer::new();
    // ε standard normal on ℝ², pushed through the chart at μ: log f(x ⊖ μ) - λ_μ(x ⊖ μ).
    let (mu, x, with, without) = writer
        .build(|| {
            let mu = quantity("mu", Sphere);
            let x = quantity("x", Sphere);
            let t = x.local(mu).untyped();
            let gaussian = -(t * t).sum() * 0.5 - (2.0 * std::f64::consts::PI).ln();
            let with = gaussian - mu.log_volume(t).untyped();
            (mu, x, with, gaussian)
        })
        .unwrap();
    let program = writer
        .snapshot()
        .compile(&Query::of([with, without]), &target)
        .unwrap();
    // The midpoint rule in the polar angle θ and the azimuth φ, with dσ = sin θ dθ dφ.
    let (n, m) = (800, 200);
    let pi = std::f64::consts::PI;
    let (h_theta, h_phi) = (pi / n as f64, 2.0 * pi / m as f64);
    let mut grid = Vec::with_capacity(3 * n * m);
    let mut weights = Vec::with_capacity(n * m);
    for i in 0..n {
        let theta = (i as f64 + 0.5) * h_theta;
        for j in 0..m {
            let phi = (j as f64 + 0.5) * h_phi;
            grid.extend([
                theta.sin() * phi.cos(),
                theta.sin() * phi.sin(),
                theta.cos(),
            ]);
            weights.push(theta.sin() * h_theta * h_phi);
        }
    }
    let rows = n * m;
    for centre in points(2) {
        let centres: Vec<f64> = (0..rows).flat_map(|_| centre).collect();
        let at = Values::new()
            .with(mu, value(&centres, rows, &target))
            .with(x, value(&grid, rows, &target));
        let out = program.forward(&at).unwrap();
        let integral = |q: QDyn| -> f64 {
            host(&out, q)
                .iter()
                .zip(&weights)
                .map(|(l, w)| l.exp() * w)
                .sum()
        };
        let (mass, uncorrected) = (integral(with), integral(without));
        assert!(
            (mass - 1.0).abs() < 1e-4,
            "the pushed density's mass is {mass}"
        );
        assert!(
            (uncorrected - 1.0).abs() > 0.05,
            "without the log-volume the mass is {uncorrected}"
        );
    }
}
