#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)] // a test crate
//! The sphere of the acceptance test, declared from `skoro`'s public interface alone.

mod common;

use std::sync::Arc;

use common::sphere::{
    COS_SERIES, EXP_CUT, EXPONENTIAL, LOG_VOLUME_SERIES, Near, SINC_SERIES, SPHERE, Sphere,
    ball_of_radius_pi, basis, exponential_log_volume, exponential_sphere,
    measured_exponential_sphere, points,
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

/// `ζ(s)` for even `s ≥ 2`: exact at `2` and `4`, where a direct sum converges far too slowly,
/// and beyond them summed to `J = 1000` with the Euler–Maclaurin tail
/// `J^{1-s}/(s-1) - J^{-s}/2`, whose own error, below `s J^{-s-1}/12`, is far below rounding.
fn zeta(s: u32) -> f64 {
    use std::f64::consts::PI;
    match s {
        2 => PI * PI / 6.0,
        4 => PI.powi(4) / 90.0,
        _ => {
            let (s, j) = (s as i32, 1000.0f64);
            let sum: f64 = (1..=1000).rev().map(|i| f64::from(i).powi(-s)).sum();
            sum + j.powi(1 - s) / f64::from(s - 1) - 0.5 * j.powi(-s)
        }
    }
}

/// `n!` as a float.
fn factorial(n: u32) -> f64 {
    (2..=n).map(f64::from).product()
}

/// `Σ cₖ xᵏ` in double precision on the host.
fn horner_on_host(coefficients: &[f64], x: f64) -> f64 {
    coefficients.iter().rev().fold(0.0, |sum, c| sum * x + c)
}

/// The truncation condition of archive/wgpu-accuracy/math: the tails of the exponential sphere's
/// series beyond their tables, at its cutoff, are below double precision's rounding of the
/// function there, in its value and its first two derivatives in `r²`.
#[test]
fn exponential_series_are_truncated_below_rounding() {
    use std::f64::consts::PI;
    let x = EXP_CUT;
    let r = x.sqrt();
    // Each series as (its first power of r² past the table, a bound on its k-th coefficient, the
    // function's value at the cutoff). Λ's coefficients are ζ(2n) / (n π^{2n}) in magnitude, with
    // ζ(s) ≤ 1 + 2^{-s} + 2^{1-s}/(s-1): geometric in r²/π².
    fn zeta_bound(s: i32) -> f64 {
        1.0 + 2f64.powi(-s) + 2f64.powi(1 - s) / f64::from(s - 1)
    }
    type Series = (&'static str, u32, fn(u32) -> f64, f64);
    let series: [Series; 3] = [
        (
            "cos",
            COS_SERIES.len() as u32,
            |k| 1.0 / factorial(2 * k),
            r.cos(),
        ),
        (
            "sinc",
            SINC_SERIES.len() as u32,
            |k| 1.0 / factorial(2 * k + 1),
            r.sin() / r,
        ),
        (
            "Λ",
            LOG_VOLUME_SERIES.len() as u32 + 1,
            |n| zeta_bound(2 * n as i32) / (f64::from(n) * PI.powi(2 * n as i32)),
            (r.sin() / r).ln().abs(),
        ),
    ];
    for (name, first, bound, value) in series {
        for m in 0..=2u32 {
            let tail: f64 = (first..first + 40)
                .map(|k| {
                    let falling: f64 = (0..m).map(|j| f64::from(k - j)).product();
                    bound(k) * falling * x.powi((k - m) as i32)
                })
                .sum();
            assert!(
                tail <= 2f64.powi(-53) * value,
                "{name}: the tail of its {m}-th derivative at r² = {x}, {tail:e}, is above \
                 rounding, {:e}",
                2f64.powi(-53) * value
            );
        }
    }
}

/// The exponential sphere's series against their closed forms just below its cutoff, where the
/// closed forms are accurate in double precision, and Λ's coefficients against `-ζ(2n) / (n
/// π^{2n})`, independent of the Bernoulli numbers they are built from.
#[test]
fn exponential_series_meet_closed_forms() {
    use std::f64::consts::PI;
    let x = 0.9 * EXP_CUT;
    let r = x.sqrt();
    for (name, got, want) in [
        ("cos", horner_on_host(&COS_SERIES, x), r.cos()),
        ("sinc", horner_on_host(&SINC_SERIES, x), r.sin() / r),
        (
            "Λ",
            x * horner_on_host(&LOG_VOLUME_SERIES, x),
            (r.sin() / r).ln(),
        ),
    ] {
        assert!(
            (got - want).abs() <= 1e-15,
            "{name} at r² = {x}: {got} against {want}"
        );
    }
    for (i, &c) in LOG_VOLUME_SERIES.iter().enumerate() {
        let n = i as u32 + 1;
        let want = -zeta(2 * n) / (f64::from(n) * PI.powi(2 * n as i32));
        assert!(
            (c - want).abs() <= 1e-14 * want.abs(),
            "Λ's coefficient of r^{}: {c} against {want}",
            2 * n
        );
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

/// `R(μ, t)` on the host: `((1 - s) μ + B(μ) t) / (1 + s)`, `s = |t|² / 4`.
fn retract_on_host(mu: [f64; 3], t: [f64; 2]) -> [f64; 3] {
    let b = basis(mu);
    let s = (t[0] * t[0] + t[1] * t[1]) / 4.0;
    [0, 1, 2].map(|i| ((1.0 - s) * mu[i] + b[i][0] * t[0] + b[i][1] * t[1]) / (1.0 + s))
}

/// `y ⊖ μ` on the host: `2 B(μ)ᵀ y / (1 + μᵀ y)`.
fn local_on_host(mu: [f64; 3], y: [f64; 3]) -> [f64; 2] {
    let b = basis(mu);
    let dot = mu[0] * y[0] + mu[1] * y[1] + mu[2] * y[2];
    let bty = |c: usize| (0..3).map(|i| b[i][c] * y[i]).sum::<f64>();
    [2.0 * bty(0) / (1.0 + dot), 2.0 * bty(1) / (1.0 + dot)]
}

#[test]
fn improper_factor_on_the_sphere() {
    use skoro_normal::Stacked;
    let target = Target::host();
    let anchor = {
        let v = [0.3f64, 0.5, 0.8];
        let n = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
        [v[0] / n, v[1] / n, v[2] / n]
    };
    let real_anchor = 0.4;
    // Two rows over D = 3, row by row, with a linear part.
    let a = [[1.0, -0.5, 0.3], [0.2, 0.7, -1.1]];
    let (b, eta, c) = ([0.4, -0.6], [0.1, -0.2, 0.5], -1.3);
    let columns: Vec<f64> = (0..3)
        .flat_map(|j| a.iter().map(move |row| row[j]))
        .collect();
    let factor = |delta: [f64; 3]| {
        let squares: f64 = (0..2)
            .map(|i| (0..3).map(|j| a[i][j] * delta[j]).sum::<f64>() - b[i])
            .map(|r| r * r)
            .sum();
        -0.5 * squares - (0..3).map(|j| eta[j] * delta[j]).sum::<f64>() + c
    };
    let at_host = |s: [f64; 3], r: f64| {
        let t = local_on_host(anchor, s);
        factor([t[0], t[1], r - real_anchor])
    };

    let mut writer = Writer::new();
    let (s, r, log_density) = writer
        .build(|| {
            let s = quantity("s", Sphere);
            let r = quantity("r", skoro::Real);
            let improper = Stacked::new()
                .block(s, Sphere, anchor)
                .block(r, skoro::Real, real_anchor)
                .improper(2, columns.as_slice(), &b[..], c)
                .linear(&eta[..]);
            (s, r, improper.log_density())
        })
        .unwrap();
    let program = writer
        .snapshot()
        .compile(&Query::of([log_density.untyped()]), &target)
        .unwrap();
    let ts = [[0.3, -0.2], [-0.5, 0.4], [0.9, 0.7]];
    let spheres: Vec<[f64; 3]> = ts.iter().map(|&t| retract_on_host(anchor, t)).collect();
    let reals = [1.1, -0.3, 0.6];
    let rows = ts.len();
    let flat: Vec<f64> = spheres.iter().flatten().copied().collect();
    let at = Values::new()
        .with(s, value(&flat, rows, &target))
        .with(r, value(&reals, rows, &target));

    // The value is the definition, with δ read by the sphere's inverse retraction.
    let (values, gradient) = program.gradient(&at).unwrap();
    let found = host(&values, log_density.id());
    for row in 0..rows {
        let want = at_host(spheres[row], reals[row]);
        assert!((found[row] - want).abs() < 1e-12, "{found:?} at row {row}");
    }

    // The gradient, in the sphere's chart at each point, against central differences, where
    // D δ moves with the point.
    let gs = gradient.get(s).unwrap().to_host();
    let gr = gradient.get(r).unwrap().to_host();
    let h = 1e-6;
    for row in 0..rows {
        let (point, real) = (spheres[row], reals[row]);
        for j in 0..2 {
            let mut t = [0.0; 2];
            t[j] = h;
            let up = at_host(retract_on_host(point, t), real);
            t[j] = -h;
            let down = at_host(retract_on_host(point, t), real);
            let want = (up - down) / (2.0 * h);
            assert!((gs[row * 2 + j] - want).abs() < 1e-6, "∂s{j} at row {row}");
        }
        let want = (at_host(point, real + h) - at_host(point, real - h)) / (2.0 * h);
        assert!((gr[row] - want).abs() < 1e-6, "∂r at row {row}");
    }
}
