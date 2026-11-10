#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)] // a test crate
//! The retractions of the core's spaces, built as composites.

use skoro_core::scope::quantity_in;
use skoro_core::testing::random;
use skoro_core::{Error, Frames, Positive, QDyn, Query, Space, Values, Vector, Writer};
use skoro_tensor::{Frame, HostData, HostValue, Precision, Target, Value};

const ROWS: usize = 3;

fn spaces() -> Vec<Space> {
    vec![
        Space::Real,
        Space::Vector(3),
        Space::Matrix(2, 2),
        Space::Positive(2),
        Space::product([Space::Real, Space::Positive(2), Space::Matrix(1, 2)]).unwrap(),
    ]
}

fn load(value: &HostValue, target: &Target) -> Value {
    value.load(ROWS, target).unwrap().unwrap()
}

fn host(values: &Values, q: QDyn) -> Vec<f64> {
    match values.get(q).unwrap() {
        Value::Continuous(points) => points.to_host(),
        Value::Discrete(_) => panic!("a continuous value"),
    }
}

/// Rounding, and nothing more.
fn rounding(target: &Target) -> f64 {
    match target.precision() {
        Precision::Double => 1e-12,
        Precision::Single => 1e-5,
    }
}

fn close(found: &[f64], expected: &[f64], relative: f64, what: &str) {
    assert_eq!(found.len(), expected.len(), "{what}");
    for (f, e) in found.iter().zip(expected) {
        assert!(
            (f - e).abs() <= relative * e.abs().max(1.0),
            "{what}: {found:?} against {expected:?}"
        );
    }
}

/// `(x ⊕ t) ⊖ x` and `x ⊖ (x ⊕ t)` for `x` and `t` drawn at random.
fn identities(space: &Space, target: &Target) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    let mut writer = Writer::new();
    let tangent = Space::Vector(space.dimension().unwrap());
    let (x, t, round, mirror) = writer
        .build(|| {
            let x = quantity_in("x", space.clone());
            let t = quantity_in("t", tangent.clone());
            let y = x.retract(t);
            (x, t, y.local(x), x.local(y))
        })
        .unwrap();
    let (xv, tv) = (
        random(space, ROWS, 1, 0.5, 2.0),
        random(&tangent, ROWS, 2, -1.0, 1.0),
    );
    let program = writer
        .snapshot()
        .compile(&Query::of([round, mirror]), target)
        .unwrap();
    let at = Values::new()
        .with(x, load(&xv, target))
        .with(t, load(&tv, target));
    let out = program.forward(&at).unwrap();
    let HostData::Continuous(t) = tv.data().clone() else {
        panic!("a continuous value")
    };
    (t, host(&out, round), host(&out, mirror))
}

skoro_core::precision_tests! {
    fn retraction_round_trips(target) {
        for space in spaces() {
            let (t, round, _) = identities(&space, &target);
            close(&round, &t, rounding(&target), &format!("(x ⊕ t) ⊖ x on {space:?}"));
        }
    }

    fn positive_mirror_identity_is_exact(target) {
        for space in spaces() {
            let (t, _, mirror) = identities(&space, &target);
            let minus: Vec<f64> = t.iter().map(|t| -t).collect();
            close(&mirror, &minus, rounding(&target), &format!("x ⊖ (x ⊕ t) on {space:?}"));
        }
    }

    fn positive_chart_hessian_equals_the_pullback_hessian(target) {
        // g(x) = Σ a_i x_i + Σ x_i², and its pull-back h(u) = g(x ⊙ e^u), whose Hessian at u = 0
        // is diag(a_i x_i + 4 x_i²); the gradient a_i x_i + 2 x_i² is not zero.
        let a = [0.7, -1.3];
        let mut writer = Writer::new();
        let (x, g) = writer
            .build(|| {
                let x = skoro_core::scope::quantity("x", Positive::<2>);
                let real = x.as_real();
                let weights = skoro_core::scope::constant(Vector::<2>, a);
                (x, (weights * real + real * real).sum())
            })
            .unwrap();
        let program = writer.snapshot().compile(&Query::of([g.into()]), &target).unwrap();
        let point = [1.4, 0.6];
        let value = HostValue::new(Some(1), 2, HostData::Continuous(point.to_vec())).unwrap();
        let at = Values::new().with(x, value.load(1, &target).unwrap().unwrap());
        let identity = Frames::new().with(x, Frame::from_host(&[1.0, 0.0, 0.0, 1.0], 1, 2, 2, &target));
        let second = program.hessian_products(&at, &identity).unwrap();
        let gradient: Vec<f64> = (0..2).map(|i| a[i] * point[i] + 2.0 * point[i] * point[i]).collect();
        let hessian: Vec<f64> = (0..2)
            .flat_map(|i| (0..2).map(move |j| if i == j { a[i] * point[i] + 4.0 * point[i] * point[i] } else { 0.0 }))
            .collect();
        let tolerance = skoro_core::testing::Tolerance::AGREEMENT;
        close(&second.alpha.get(x).unwrap().to_host(), &gradient, tolerance, "the gradient in the chart");
        close(&second.beta.get(x).unwrap().to_host(), &hessian, tolerance, "the chart Hessian");
    }
}

#[test]
fn discrete_spaces_have_no_retraction() {
    for space in [Space::Integer, Space::Categorical(3)] {
        assert_eq!(
            space.retraction().unwrap_err(),
            Error::NotContinuous {
                space: space.clone()
            }
        );
    }
    // One composite per space, cached.
    let a = Space::Positive(2).retraction().unwrap();
    let b = Space::Positive(2).retraction().unwrap();
    assert!(std::sync::Arc::ptr_eq(&a, &b));
}

/// A circle declared by stand-in composites of the right signatures, with the log-volume
/// `log_volume` of the point's storage and the tangent.
fn circle_with(
    name: &'static str,
    log_volume: fn(&[QDyn]) -> QDyn,
) -> std::sync::Arc<skoro_core::Embedded> {
    use skoro_core::{Composite, Embedded, Volume};
    let (n, d) = (Space::Vector(2), Space::Vector(1));
    let r = Composite::define(
        "test.volume.retract",
        Vec::new(),
        &[n.clone(), d.clone()],
        |q| q[0],
    )
    .unwrap();
    let q = Composite::define(
        "test.volume.local",
        Vec::new(),
        &[n.clone(), n.clone()],
        |q| q[1].slice(0..1) - q[0].slice(0..1),
    )
    .unwrap();
    let lambda = Composite::define(name, Vec::new(), &[n, d], log_volume).unwrap();
    Embedded::declare(
        "test.volume.circle",
        vec![skoro_core::Param::Text(name.into())],
        2,
        1,
        r,
        q,
        None,
    )
    .and_then(|circle| circle.with_volume(Volume::global(lambda)))
    .unwrap()
}

#[test]
fn volume_reads_point_by_space() {
    for space in [Space::Real, Space::Vector(3), Space::Matrix(2, 2)] {
        assert_eq!(space.volume_reads_point(), Ok(false), "{space:?}");
    }
    assert_eq!(Space::Positive(2).volume_reads_point(), Ok(true));
    let mixed = Space::product([Space::Vector(2), Space::Positive(1)]).unwrap();
    assert_eq!(mixed.volume_reads_point(), Ok(true));
    let flat = Space::product([Space::Vector(2), Space::Matrix(1, 2)]).unwrap();
    assert_eq!(flat.volume_reads_point(), Ok(false));
    // The sphere's and the rotations' log-volumes read the tangent alone.
    let invariant = Space::Embedded(circle_with("test.volume.tangent", |q| {
        ((q[1] * q[1]).sum() * 0.25 + 1.0).log() * -2.0
    }));
    assert_eq!(invariant.volume_reads_point(), Ok(false));
    let product = Space::product([invariant, Space::Vector(1)]).unwrap();
    assert_eq!(product.volume_reads_point(), Ok(false));
    let moving = Space::Embedded(circle_with("test.volume.point", |q| {
        q[0].sum() * 0.0 + q[1].sum()
    }));
    assert_eq!(
        moving.volume_reads_point(),
        Ok(true),
        "read, even if multiplied by zero"
    );
    assert!(matches!(
        Space::Integer.volume_reads_point(),
        Err(Error::NotContinuous { .. })
    ));
    let unmeasured = circle_with("test.volume.unmeasured", |q| q[1].sum());
    let unmeasured = skoro_core::Embedded::declare(
        "test.volume.none",
        Vec::new(),
        2,
        1,
        unmeasured.retraction().clone(),
        unmeasured.inverse().clone(),
        None,
    )
    .unwrap();
    assert!(matches!(
        Space::Embedded(unmeasured).volume_reads_point(),
        Err(Error::NoVolume { .. })
    ));
    let bare = skoro_core::Embedded::declare_flat("test.volume.flat", Vec::new(), 2).unwrap();
    assert_eq!(Space::Embedded(bare).volume_reads_point(), Ok(false));
}
