//! The pose operations, `compose`, `invert`, `transform` and `untransform`, against forward
//! differences in the body-frame chart `(r, p) ⊕ (a, b) = (r Exp(a), p + r b)`.
//!
//! Every input is checked, with both mixed HVP blocks where there are two: several diagonal blocks
//! are identically zero, so a test that seeds and reads the same input cannot see a wrong term.
//! The poses are far from the identity and far from the origin, which makes the lever-arm terms
//! (`a × p`, `a × u`) large.

use skoro_backend::Backend;
use skoro_graph::quantity::ContinuousQuantity;
use skoro_model::{
    hash_map_builders::points,
    model::ModelClient,
    q::Q,
    test_utils::{
        tensor, test_hvp_cross_numerically, test_hvp_numerically, test_jvp_numerically,
        test_vjp_numerically,
    },
};
use skoro_operation::ValueMap;
use skoro_riemann::{euclidean::EuclideanD, matrix::EuclideanMatrixD};
use skoro_se3::{
    ops::{SE3ComposeOps, SE3InvertOps, SE3TransformOps},
    quantity::SE3,
};

/// `(Exp((0.3, -0.2, 0.5)), (5, 6, 7))`.
const X0: [f64; 12] = [
    0.8595338985586632,
    -0.497991537002922,
    -0.11491695393636675,
    0.43986763295823095,
    0.8353156052067086,
    -0.3297943376922551,
    0.26022671404809444,
    0.23292116428443663,
    0.937032437284918,
    5.0,
    6.0,
    7.0,
];

/// `(Exp((-0.7, 1.1, 0.25)), (2.5, -4, 0.75))`.
const X1: [f64; 12] = [
    0.45187853044804804,
    -0.5144418593332605,
    0.7288040663208806,
    -0.14890357336615478,
    0.7620140574243981,
    0.6302081419074148,
    -0.8795643919343841,
    -0.39329905880048077,
    0.2677355613058403,
    2.5,
    -4.0,
    0.75,
];

/// A point, and a `3 × 2` matrix whose columns are two points.
const U: [f64; 3] = [1.5, -2.0, 0.75];
const M: [f64; 6] = [1.5, -0.5, -2.0, 1.25, 0.75, 2.0];

fn check_unary<B, A, O>(a: &Q<B, A>, out: &Q<B, O>, point: ValueMap<B>, delta: f64, epsilon: f64)
where
    B: Backend,
    A: ContinuousQuantity<B>,
    O: ContinuousQuantity<B>,
{
    test_jvp_numerically(a, out, point.clone(), delta, epsilon);
    test_vjp_numerically(a, out, point.clone(), delta, epsilon);
    test_hvp_numerically(a, out, point, delta, epsilon);
}

fn check_binary<B, A, C, O>(
    a: &Q<B, A>,
    b: &Q<B, C>,
    out: &Q<B, O>,
    point: ValueMap<B>,
    delta: f64,
    epsilon: f64,
) where
    B: Backend,
    A: ContinuousQuantity<B>,
    C: ContinuousQuantity<B>,
    O: ContinuousQuantity<B>,
{
    check_unary(a, out, point.clone(), delta, epsilon);
    check_unary(b, out, point.clone(), delta, epsilon);
    test_hvp_cross_numerically(a, b, out, point.clone(), delta, epsilon);
    test_hvp_cross_numerically(b, a, out, point, delta, epsilon);
}

pub fn compose_impl<B: Backend>(delta: f64, epsilon: f64)
where
    SE3: ContinuousQuantity<B>,
{
    let model = ModelClient::<B>::default();
    let x0 = model.add_parameter(SE3);
    let x1 = model.add_parameter(SE3);
    let out = x0.compose(&x1).unwrap();
    let point = points()
        .with_quantity(&x0, tensor::<B, 2>([X0], &Default::default()))
        .with_quantity(&x1, tensor::<B, 2>([X1], &Default::default()))
        .build();
    check_binary(&x0, &x1, &out, point, delta, epsilon);
}

pub fn invert_impl<B: Backend>(delta: f64, epsilon: f64)
where
    SE3: ContinuousQuantity<B>,
{
    let model = ModelClient::<B>::default();
    let x = model.add_parameter(SE3);
    let out = x.invert().unwrap();
    let point = points()
        .with_quantity(&x, tensor::<B, 2>([X1], &Default::default()))
        .build();
    check_unary(&x, &out, point, delta, epsilon);
}

/// `transform` (or, with `inverse`, `untransform`) of a point and of a matrix of points.
pub fn transform_impl<B: Backend>(inverse: bool, delta: f64, epsilon: f64)
where
    SE3: ContinuousQuantity<B>,
{
    let model = ModelClient::<B>::default();
    let x = model.add_parameter(SE3);
    let u = model.add_parameter(EuclideanD(3));
    let m = model.add_parameter(EuclideanMatrixD(3, 2));
    let point = points()
        .with_quantity(&x, tensor::<B, 2>([X0], &Default::default()))
        .with_quantity(&u, tensor::<B, 2>([U], &Default::default()))
        .with_quantity(&m, tensor::<B, 2>([M], &Default::default()))
        .build();
    let (out_u, out_m) = if inverse {
        (x.untransform(&u).unwrap(), x.untransform(&m).unwrap())
    } else {
        (x.transform(&u).unwrap(), x.transform(&m).unwrap())
    };
    check_binary(&x, &u, &out_u, point.clone(), delta, epsilon);
    check_binary(&x, &m, &out_m, point, delta, epsilon);
}

/// Forward values against the group laws: `x ∘ x⁻¹` is the identity, and `untransform` undoes
/// `transform`.
pub fn group_laws_impl<B: Backend>()
where
    SE3: ContinuousQuantity<B>,
{
    let model = ModelClient::<B>::default();
    let x = model.add_parameter(SE3);
    let u = model.add_parameter(EuclideanD(3));
    let point = points()
        .with_quantity(&x, tensor::<B, 2>([X0], &Default::default()))
        .with_quantity(&u, tensor::<B, 2>([U], &Default::default()))
        .build();
    let identity = x.compose(&x.invert().unwrap()).unwrap();
    let round_trip = x.untransform(&x.transform(&u).unwrap()).unwrap();
    let values = |q: skoro_core::id::Id| -> Vec<f64> {
        model
            .evaluate_ids(&point, &[q])
            .unwrap()
            .evaluations
            .get(&q)
            .unwrap()
            .0
            .clone()
            .unwrap_float()
            .into_data()
            .convert::<f64>()
            .to_vec::<f64>()
            .unwrap()
    };
    let want = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0];
    for (got, want) in values(identity.id()).iter().zip(want) {
        assert!((got - want).abs() < 1e-4, "x ∘ x⁻¹: {got} vs {want}");
    }
    for (got, want) in values(round_trip.id()).iter().zip(U) {
        assert!(
            (got - want).abs() < 1e-4,
            "untransform ∘ transform: {got} vs {want}"
        );
    }
}
