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
use skoro_se3::{ops::SE3ComposeOps, quantity::SE3};

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
