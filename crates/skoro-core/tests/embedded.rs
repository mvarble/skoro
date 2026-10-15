#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)] // a test crate
//! Embedded types, on the circle: storage 2 and dimension 1, its retraction a rotation.

use std::sync::{Arc, OnceLock};

use skoro_core::primitive::{Include, Retract};
use skoro_core::scope::{apply, constant, quantity};
use skoro_core::testing::check_embedded;
use skoro_core::{
    Composite, Declares, Embedded, Error, IntoHostValue, Lowered, Lowering, Operation, Param, QDyn,
    Query, Space, Typed, Values, Vector, Writer,
};
use skoro_tensor::{HostData, HostValue, Target, Value};

/// `R(x, t)`: `x` turned by the angle `t`.
fn rotation(name: &'static str) -> Arc<Composite> {
    Composite::define(
        name,
        Vec::new(),
        &[Space::Vector(2), Space::Vector(1)],
        |q| {
            let (x, t) = (q[0], q[1]);
            let (c, s) = (t.cos().reshape(Space::Real), t.sin().reshape(Space::Real));
            let (x1, x2) = (x.slice(0..1), x.slice(1..2));
            let r1 = c * x1 - s * x2;
            let r2 = s * x1 + c * x2;
            skoro_core::scope::concatenate(&[r1, r2], Space::Vector(2))
        },
    )
    .unwrap()
}

/// `Q(x, y)`: the angle from `x` to `y`, times `scale`.
fn angle(scale: f64) -> Arc<Composite> {
    Composite::define(
        "test.circle.angle",
        vec![Param::Float(scale)],
        &[Space::Vector(2), Space::Vector(2)],
        move |q| {
            let (x, y) = (q[0], q[1]);
            let (x1, x2, y1, y2) = (x.slice(0..1), x.slice(1..2), y.slice(0..1), y.slice(1..2));
            (x1 * y2 - x2 * y1).atan2(x1 * y1 + x2 * y2) * scale
        },
    )
    .unwrap()
}

fn circle() -> Arc<Embedded> {
    Embedded::declare(
        "test.circle",
        Vec::new(),
        2,
        1,
        rotation("test.circle.rotation"),
        angle(1.0),
    )
    .unwrap()
}

/// The circle with the retraction of a step along the tangent line, normalized:
/// `R(x, t) = (x + t B(x)) / √(1 + t²)`, whose inverse is `Q(x, y) = B(x)·y / x·y`. In the angle
/// chart it moves θ to θ + atan t, which is no translation, so its second-order maps are not zero.
fn tangent_circle() -> Arc<Embedded> {
    let r = Composite::define(
        "test.circle.tangent.retraction",
        Vec::new(),
        &[Space::Vector(2), Space::Vector(1)],
        |q| {
            let (x, t) = (q[0], q[1].reshape(Space::Real));
            let (x1, x2) = (x.slice(0..1), x.slice(1..2));
            let bt = skoro_core::scope::concatenate(&[-(t * x2), t * x1], Space::Vector(2));
            let inverse = (-((t * t + 1.0).log() * 0.5)).exp();
            inverse * (x + bt)
        },
    )
    .unwrap();
    let q = Composite::define(
        "test.circle.tangent.inverse",
        Vec::new(),
        &[Space::Vector(2), Space::Vector(2)],
        |q| {
            let (x, y) = (q[0], q[1]);
            let (x1, x2, y1, y2) = (x.slice(0..1), x.slice(1..2), y.slice(0..1), y.slice(1..2));
            let along = x1 * y2 - x2 * y1;
            let inverse = (-(x1 * y1 + x2 * y2).log()).exp();
            along * inverse
        },
    )
    .unwrap();
    Embedded::declare("test.circle.tangent", Vec::new(), 2, 1, r, q).unwrap()
}

/// The circle with a basis whose length varies along it, `B(x) = ρ(x) J x` for
/// `ρ(x) = 1 + x₁ / 2` and `J x = (-x₂, x₁)`: `R(x, t) = (x + t B(x)) / √(1 + t² ρ²)` and
/// `Q(x, y) = J x · y / (ρ(x) x·y)`. On a curve whose basis has unit length, `D B(x)[ẋ]` and
/// `D L(y)[ẏ]` are normal to it, and the terms of the second-order maps they enter vanish; here
/// they do not.
fn scaled_circle() -> Arc<Embedded> {
    let rho = |x: QDyn| x.slice(0..1).reshape(Space::Real) * 0.5 + 1.0;
    let r = Composite::define(
        "test.circle.scaled.retraction",
        Vec::new(),
        &[Space::Vector(2), Space::Vector(1)],
        move |q| {
            let (x, t) = (q[0], q[1].reshape(Space::Real));
            let (x1, x2) = (x.slice(0..1), x.slice(1..2));
            let step = t * rho(x);
            let bt = skoro_core::scope::concatenate(&[-(step * x2), step * x1], Space::Vector(2));
            let inverse = (-((step * step + 1.0).log() * 0.5)).exp();
            inverse * (x + bt)
        },
    )
    .unwrap();
    let q = Composite::define(
        "test.circle.scaled.inverse",
        Vec::new(),
        &[Space::Vector(2), Space::Vector(2)],
        move |q| {
            let (x, y) = (q[0], q[1]);
            let (x1, x2, y1, y2) = (x.slice(0..1), x.slice(1..2), y.slice(0..1), y.slice(1..2));
            let along = x1 * y2 - x2 * y1;
            let dot = (x1 * y1 + x2 * y2).reshape(Space::Real);
            let inverse = (-(dot.log() + rho(x).log())).exp();
            along * inverse
        },
    )
    .unwrap();
    Embedded::declare("test.circle.scaled", Vec::new(), 2, 1, r, q).unwrap()
}

/// Points of the circle, at angles spread round it.
fn points(rows: usize) -> HostValue {
    let data: Vec<f64> = (0..rows)
        .flat_map(|i| {
            let theta = 0.3 + 2.1 * i as f64;
            [theta.cos(), theta.sin()]
        })
        .collect();
    HostValue::new(Some(rows), 2, HostData::Continuous(data)).unwrap()
}

skoro_core::precision_tests! {
    fn circle_passes_check_embedded(target) {
        for declaration in [circle(), tangent_circle(), scaled_circle()] {
            if let Err(error) = check_embedded(&declaration, &points(3), &target) {
                panic!("{error}");
            }
        }
    }
}

#[test]
fn a_wrong_inverse_fails_check_embedded() {
    let wrong = Embedded::declare(
        "test.circle.wrong",
        Vec::new(),
        2,
        1,
        rotation("test.circle.rotation"),
        angle(2.0),
    )
    .unwrap();
    let error = check_embedded(&wrong, &points(3), &Target::host()).unwrap_err();
    assert!(error.contains("Q(x, R(x, t)) is not t"), "{error}");
}

#[test]
fn declarations_of_the_wrong_signature_are_refused() {
    let refused = Embedded::declare("test.bad", Vec::new(), 2, 1, angle(1.0), angle(1.0));
    assert!(matches!(refused, Err(Error::InvalidDeclaration { .. })));
    let refused = Embedded::declare("test.bad", Vec::new(), 2, 3, rotation("r"), angle(1.0));
    assert!(matches!(refused, Err(Error::InvalidDeclaration { .. })));
}

#[test]
fn storage_and_dimension_of_each_space() {
    let circle = Space::Embedded(circle());
    let cases = [
        (Space::Real, 1, Some(1)),
        (Space::Vector(3), 3, Some(3)),
        (Space::Matrix(2, 3), 6, Some(6)),
        (Space::Positive(2), 2, Some(2)),
        (Space::Categorical(4), 1, None),
        (Space::Integer, 1, None),
        (circle.clone(), 2, Some(1)),
    ];
    for (space, storage, dimension) in cases {
        assert_eq!(
            (space.storage(), space.dimension()),
            (storage, dimension),
            "{space:?}"
        );
    }
    // A product with an embedded factor: its storage and tangent ranges differ.
    let product = Space::product([Space::Real, circle, Space::Vector(2)]).unwrap();
    assert_eq!((product.storage(), product.dimension()), (5, Some(4)));
    let blocks = product.blocks();
    let storage: Vec<_> = blocks.iter().map(|b| b.storage.clone()).collect();
    let tangent: Vec<_> = blocks.iter().map(|b| b.tangent.clone()).collect();
    assert_eq!(storage, [0..1, 1..3, 3..5]);
    assert_eq!(tangent, [0..1, 1..2, 2..4]);
    // Two declarations of one name and parameters are one space.
    assert_eq!(
        Space::Embedded(self::circle()),
        Space::Embedded(self::circle())
    );
}

/// A marker for the cyclic declaration, read off a static once it is declared.
#[derive(Clone, Copy, Debug)]
struct Cyclic;

static CYCLIC: OnceLock<Arc<Embedded>> = OnceLock::new();

impl Typed for Cyclic {
    fn admits(space: &Space) -> bool {
        CYCLIC
            .get()
            .is_some_and(|e| *space == Space::Embedded(e.clone()))
    }
}

impl Declares for Cyclic {
    fn space(&self) -> Space {
        Space::Embedded(CYCLIC.get().unwrap().clone())
    }
}

impl IntoHostValue<Cyclic> for [f64; 2] {
    fn into_host_value(self) -> Option<HostValue> {
        HostValue::new(None, 2, HostData::Continuous(self.to_vec()))
    }
}

/// An operation in the cyclic `R`, whose fallback retracts on the cyclic declaration itself.
#[derive(Debug)]
struct Turn;

impl Operation for Turn {
    fn name(&self) -> &'static str {
        "test.turn"
    }

    fn params(&self) -> Vec<Param> {
        Vec::new()
    }

    fn output(&self, inputs: &[Space]) -> Result<Space, Error> {
        Ok(inputs[0].clone())
    }

    fn lower(&self, inputs: &[Space], _: &mut Lowering<'_>) -> Result<Lowered, Error> {
        let declaration = CYCLIC.get().unwrap().clone();
        let fallback = Composite::define("test.turn.fallback", Vec::new(), inputs, move |q| {
            let c = constant(Cyclic, [1.0, 0.0]).untyped();
            let y = apply(Arc::new(Retract::new(declaration.clone())), &[c, q[1]]);
            apply(Arc::new(Include::new(declaration)), &[y]) + q[0] * 0.0
        })
        .map_err(|failure| failure.error)?;
        Ok(Lowered::Composite(fallback))
    }
}

#[test]
fn cycle_through_a_boundary_kernel_is_refused() {
    let r = Composite::define(
        "test.cyclic.retraction",
        Vec::new(),
        &[Space::Vector(2), Space::Vector(1)],
        |q| apply(Arc::new(Turn), &[q[0], q[1]]),
    )
    .unwrap();
    let declaration = Embedded::declare("test.cyclic", Vec::new(), 2, 1, r, angle(1.0)).unwrap();
    CYCLIC.set(declaration).unwrap();
    let mut writer = Writer::new();
    let y = writer
        .build(|| {
            let x = quantity("x", Cyclic);
            let t = quantity("t", Vector::<1>);
            x.retract(t)
        })
        .unwrap();
    let error = writer
        .snapshot()
        .compile(&Query::of([QDyn::from(y)]), &Target::host())
        .unwrap_err();
    let Error::CompositeCycle { chain } = error else {
        panic!("{error:?}")
    };
    assert_eq!(
        chain[chain.len() - 3..],
        [
            "test.cyclic.retraction",
            "test.turn.fallback",
            "test.cyclic.retraction"
        ],
        "{chain:?}"
    );
}

/// A marker for the circle, read off a static.
#[derive(Clone, Copy, Debug)]
struct Circle;

static CIRCLE: std::sync::LazyLock<Arc<Embedded>> = std::sync::LazyLock::new(circle);

impl Typed for Circle {
    fn admits(space: &Space) -> bool {
        *space == Space::Embedded(CIRCLE.clone())
    }
}

impl Declares for Circle {
    fn space(&self) -> Space {
        Space::Embedded(CIRCLE.clone())
    }
}

#[test]
fn include_reads_an_embedded_value_as_storage() {
    let mut writer = Writer::new();
    let (x, y) = writer
        .build(|| {
            let x = quantity("x", Circle);
            (x, x.include())
        })
        .unwrap();
    let program = writer
        .snapshot()
        .compile(&Query::of([QDyn::from(y)]), &Target::host())
        .unwrap();
    let point = HostValue::new(Some(1), 2, HostData::Continuous(vec![0.6, 0.8]))
        .unwrap()
        .load(1, &Target::host())
        .unwrap()
        .unwrap();
    let values = program.forward(&Values::new().with(x, point)).unwrap();
    let Value::Continuous(points) = values.get(y).unwrap() else {
        panic!("a continuous value")
    };
    assert_eq!(points.to_host(), [0.6, 0.8]);

    let failure = writer
        .build(|| quantity("r", skoro_core::Real).include())
        .unwrap_err();
    assert!(
        matches!(failure.error, Error::SpaceMismatch { .. }),
        "{failure:?}"
    );
}
