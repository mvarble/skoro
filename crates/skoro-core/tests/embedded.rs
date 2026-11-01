#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)] // a test crate
//! Embedded types, on the circle: storage 2 and dimension 1, its retraction a rotation.

use std::sync::{Arc, OnceLock};

use skoro_core::primitive::{Include, Neg, Project, Retract};
use skoro_core::scope::{apply, constant, project, quantity};
use skoro_core::testing::{check_embedded, lower};
use skoro_core::{
    Composite, Declares, Domain, Embedded, Error, FirstOrder, IntoHostValue, Kernel, Lowered,
    Lowering, Operation, Order, Param, QDyn, Query, Space, Typed, Values, Vector, Volume, Writer,
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

/// `P(w) = w / |w|`.
fn normalize() -> Arc<Composite> {
    Composite::define(
        "test.circle.normalize",
        Vec::new(),
        &[Space::Vector(2)],
        |q| {
            let w = q[0];
            let norm = (w * w).sum();
            (-(norm.log() * 0.5)).exp() * w
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
        Some(normalize()),
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
    Embedded::declare(
        "test.circle.tangent",
        Vec::new(),
        2,
        1,
        r,
        q,
        Some(normalize()),
    )
    .unwrap()
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
    Embedded::declare(
        "test.circle.scaled",
        Vec::new(),
        2,
        1,
        r,
        q,
        Some(normalize()),
    )
    .unwrap()
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
        None,
    )
    .unwrap();
    let error = check_embedded(&wrong, &points(3), &Target::host()).unwrap_err();
    assert!(error.contains("Q(x, R(x, t)) is not t"), "{error}");
}

#[test]
fn declarations_of_the_wrong_signature_are_refused() {
    let refused = Embedded::declare("test.bad", Vec::new(), 2, 1, angle(1.0), angle(1.0), None);
    assert!(matches!(refused, Err(Error::InvalidDeclaration { .. })));
    let refused = Embedded::declare(
        "test.bad",
        Vec::new(),
        2,
        3,
        rotation("r"),
        angle(1.0),
        None,
    );
    assert!(matches!(refused, Err(Error::InvalidDeclaration { .. })));
    let refused = Embedded::declare(
        "test.bad",
        Vec::new(),
        2,
        1,
        rotation("r"),
        angle(1.0),
        Some(rotation("p")),
    );
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
    let declaration =
        Embedded::declare("test.cyclic", Vec::new(), 2, 1, r, angle(1.0), None).unwrap();
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

/// The circle with no projection.
#[derive(Clone, Copy, Debug)]
struct Bare;

static BARE: std::sync::LazyLock<Arc<Embedded>> = std::sync::LazyLock::new(|| {
    let r = rotation("test.circle.bare.rotation");
    Embedded::declare("test.circle.bare", Vec::new(), 2, 1, r, angle(1.0), None).unwrap()
});

impl Typed for Bare {
    fn admits(space: &Space) -> bool {
        *space == Space::Embedded(BARE.clone())
    }
}

impl Declares for Bare {
    fn space(&self) -> Space {
        Space::Embedded(BARE.clone())
    }
}

#[test]
fn project_without_a_declared_projection_fails() {
    assert!(!BARE.has_projection() && CIRCLE.has_projection());
    let mut writer = Writer::new();
    let failure = writer
        .build(|| {
            let w = quantity("w", Vector::<2>);
            let projected = project(Bare, w);
            assert!(projected.is_failed());
        })
        .unwrap_err();
    assert!(
        matches!(&failure.error, Error::NoProjection { space } if *space == Space::Embedded(BARE.clone())),
        "{failure:?}"
    );
}

/// A primitive, its kernel lowered as first-order only.
#[derive(Debug)]
struct FirstOnly(Neg);

impl Operation for FirstOnly {
    fn name(&self) -> &'static str {
        "test.first_only"
    }

    fn params(&self) -> Vec<Param> {
        Vec::new()
    }

    fn output(&self, inputs: &[Space]) -> Result<Space, Error> {
        self.0.output(inputs)
    }

    fn lower(&self, inputs: &[Space], lowering: &mut Lowering<'_>) -> Result<Lowered, Error> {
        Ok(match self.0.lower(inputs, lowering)? {
            Lowered::Kernel(Kernel::Two(kernel)) => {
                let kernel: Arc<dyn FirstOrder> = kernel;
                Lowered::Kernel(Kernel::One(kernel))
            }
            lowered => lowered,
        })
    }
}

/// `-(-c)` with the negations first-order only.
fn first_order_identity(c: QDyn) -> QDyn {
    let neg = |q: QDyn| apply(Arc::new(FirstOnly(Neg)), &[q]);
    neg(neg(c))
}

#[test]
fn boundary_orders_follow_the_composites() {
    let orders = |declaration: &Arc<Embedded>| {
        let space = Space::Embedded(declaration.clone());
        let order = |operation: &dyn Operation, inputs: &[Space]| {
            lower(operation, inputs, &Target::host()).unwrap().order()
        };
        [
            order(
                &Include::new(declaration.clone()),
                std::slice::from_ref(&space),
            ),
            order(
                &Retract::new(declaration.clone()),
                &[space, Space::Vector(1)],
            ),
            order(&Project::new(declaration.clone()), &[Space::Vector(2)]),
        ]
    };
    assert_eq!(orders(&CIRCLE), [Order::Two; 3]);

    let q = Composite::define(
        "test.circle.first_order_angle",
        Vec::new(),
        &[Space::Vector(2), Space::Vector(2)],
        |q| first_order_identity(angle(1.0).call(q)),
    )
    .unwrap();
    let rough_q = Embedded::declare(
        "test.circle.rough_q",
        Vec::new(),
        2,
        1,
        rotation("test.circle.rotation"),
        q,
        Some(normalize()),
    )
    .unwrap();
    assert_eq!(orders(&rough_q), [Order::Two, Order::One, Order::One]);

    let p = Composite::define(
        "test.circle.first_order_normalize",
        Vec::new(),
        &[Space::Vector(2)],
        |q| first_order_identity(normalize().call(q)),
    )
    .unwrap();
    let rough_p = Embedded::declare(
        "test.circle.rough_p",
        Vec::new(),
        2,
        1,
        rotation("test.circle.rotation"),
        angle(1.0),
        Some(p),
    )
    .unwrap();
    assert_eq!(orders(&rough_p), [Order::Two, Order::Two, Order::One]);
}

/// A composite of `inputs` whose output is `output`, made from the first input: a stand-in
/// for a log-volume or a domain of a given signature.
fn stand_in(name: &'static str, inputs: &[Space], output: &Space) -> Arc<Composite> {
    let output = output.clone();
    Composite::define(name, Vec::new(), inputs, move |q| match output {
        Space::Real => (q[1] * q[1]).sum(),
        Space::Categorical(2) => (q[1] * q[1]).sum().is_nonnegative(),
        _ => q[1] * 2.0,
    })
    .unwrap()
}

fn is_invalid(result: Result<Arc<Embedded>, Error>) -> bool {
    matches!(result, Err(Error::InvalidDeclaration { .. }))
}

#[test]
fn with_volume_refuses_a_wrong_signature() {
    let (n, d) = (Space::Vector(2), Space::Vector(1));
    let real = stand_in("test.volume.real", &[n.clone(), d.clone()], &Space::Real);
    let vector = stand_in("test.volume.vector", &[n.clone(), d.clone()], &d);
    assert!(is_invalid(circle().with_volume(Volume::global(vector))));
    let wide = stand_in(
        "test.volume.wide",
        &[Space::Vector(3), d.clone()],
        &Space::Real,
    );
    assert!(is_invalid(circle().with_volume(Volume::global(wide))));
    let not_boolean = stand_in("test.domain.real", &[n.clone(), d.clone()], &Space::Real);
    assert!(is_invalid(
        circle().with_volume(Volume::within(real.clone(), not_boolean))
    ));
    let boolean = stand_in("test.domain.boolean", &[n, d], &Space::BOOLEAN);
    let declared = circle().with_volume(Volume::within(real, boolean)).unwrap();
    assert!(matches!(
        declared.volume().map(|v| v.domain()),
        Some(Domain::Within(_))
    ));
}

#[test]
fn with_volume_refuses_a_second_volume_and_a_flat_declaration() {
    let real = stand_in(
        "test.volume.real",
        &[Space::Vector(2), Space::Vector(1)],
        &Space::Real,
    );
    let once = circle().with_volume(Volume::global(real.clone())).unwrap();
    assert!(is_invalid(once.with_volume(Volume::global(real))));
    let flat = Embedded::declare_flat("test.flat.volume", Vec::new(), 2).unwrap();
    let zero = stand_in(
        "test.volume.flat",
        &[Space::Vector(2), Space::Vector(2)],
        &Space::Real,
    );
    assert!(is_invalid(flat.with_volume(Volume::global(zero))));
}

#[test]
fn flat_declaration_has_a_global_zero_volume() {
    let flat = Embedded::declare_flat("test.flat.measured", Vec::new(), 2).unwrap();
    let volume = flat.volume().expect("a flat declaration has a volume");
    assert!(matches!(volume.domain(), Domain::Global));
    let space = Space::Embedded(flat);
    assert!(space.has_measure());
    assert!(matches!(space.domain(), Ok(Domain::Global)));
    let inputs = [(space.clone(), points(3)), (Space::Vector(2), points(3))];
    skoro_core::testing::check_agreement(
        |q| q[0].log_volume(q[1]) + (q[0].include() * 0.0).sum() + (q[1] * 0.0).sum(),
        |q| (q[0].include() * 0.0).sum() + (q[1] * 0.0).sum(),
        &inputs,
        &Target::host(),
    )
    .unwrap();
}

#[test]
fn declaration_without_volume_has_no_measure() {
    let space = Space::Embedded(circle());
    let product = Space::product([Space::Real, space.clone()]).unwrap();
    for space in [space, product] {
        assert!(!space.has_measure(), "{space:?}");
        let no_volume = Error::NoVolume {
            space: space.clone(),
            quantity: None,
        };
        assert_eq!(space.log_volume().err(), Some(no_volume.clone()));
        assert!(matches!(space.domain(), Err(e) if e == no_volume));
    }
}

/// The circle with a volume of zero and the domain `|t| ≤ radius`, under a name of its own.
fn ball_circle(name: &'static str, radius: f64) -> Arc<Embedded> {
    let (n, d) = (Space::Vector(2), Space::Vector(1));
    let zero = Composite::define(
        "test.circle.zero_volume",
        Vec::new(),
        &[n.clone(), d.clone()],
        |_| constant(skoro_core::Real, 0.0).untyped(),
    )
    .unwrap();
    let ball = Composite::define(
        "test.circle.ball",
        vec![Param::Float(radius)],
        &[n, d],
        move |q| (-(q[1] * q[1]).sum() + radius * radius).is_nonnegative(),
    )
    .unwrap();
    Embedded::declare(name, Vec::new(), 2, 1, rotation(name), angle(1.0), None)
        .unwrap()
        .with_volume(Volume::within(zero, ball))
        .unwrap()
}

#[test]
fn product_domain_is_the_conjunction_of_its_factors() {
    let (a, b) = (
        ball_circle("test.circle.ball.one", 1.0),
        ball_circle("test.circle.ball.two", 2.0),
    );
    let space = Space::product([Space::Embedded(a), Space::Real, Space::Embedded(b)]).unwrap();
    let Ok(Domain::Within(domain)) = space.domain() else {
        panic!("a product with bounded factors has a bounded domain");
    };
    // Rows of tangents (t₁, s, t₂): inside both balls, outside the first, outside the second,
    // outside both, and on the boundary of each.
    let tangents = [
        [0.5, 9.0, 1.5],
        [1.5, 0.0, 1.5],
        [0.5, 0.0, 2.5],
        [-1.5, 0.0, -2.5],
        [1.0, -9.0, -2.0],
    ];
    let expected = [1.0, 0.0, 0.0, 0.0, 1.0];
    let rows = tangents.len();
    let x: Vec<f64> = (0..rows).flat_map(|_| [1.0, 0.0, 0.3, 0.0, 1.0]).collect();
    let t: Vec<f64> = tangents.iter().flatten().copied().collect();
    let program = domain.compile(&Target::host()).unwrap();
    let free = program.free();
    let load = |data: Vec<f64>, storage| {
        HostValue::new(Some(rows), storage, HostData::Continuous(data))
            .unwrap()
            .load(rows, &Target::host())
            .unwrap()
            .unwrap()
    };
    let at = Values::new()
        .with(free[0].node, load(x, 5))
        .with(free[1].node, load(t, 3));
    let out = program.forward(&at).unwrap();
    let Some(Value::Discrete(inside)) = out.get(program.targets()[0]) else {
        panic!("a domain is a boolean");
    };
    let inside: Vec<f64> = inside.to_host().into_iter().map(|i| i as f64).collect();
    assert_eq!(inside, expected);
}
