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
