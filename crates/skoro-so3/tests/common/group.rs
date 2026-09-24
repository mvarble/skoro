//! The group operations, `compose`, `invert`, `rotate` and `unrotate`, against forward
//! differences in the body-frame chart `r ⊕ a = r Exp(a)`.
//!
//! Each is checked on every input, and on both mixed HVP blocks where it has two inputs: a test
//! that seeds and reads the same input sees only a diagonal block, and several of these are
//! identically zero. Rotations are far from the identity, since `rᵀ` and `r` are indistinguishable
//! near it.

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

use skoro_so3::quantity::SO3;

use super::retraction::R0;

/// `Exp((-0.7, 0.4, 0.9))`.
pub const R1: [f64; 9] = [
    0.5712102635432366,
    -0.8202167393050774,
    0.03103764466921849,
    0.5726680254743688,
    0.4253333428929975,
    0.70081586741651,
    -0.5880222507883133,
    -0.3825389496341701,
    0.7126666714464989,
];

/// Every derivative of `out` with respect to `a` and `b`, including both mixed HVP blocks.
#[allow(unused)]
pub fn check_binary<B, A, C, O>(
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
    test_jvp_numerically(a, out, point.clone(), delta, epsilon);
    test_vjp_numerically(a, out, point.clone(), delta, epsilon);
    test_hvp_numerically(a, out, point.clone(), delta, epsilon);
    test_jvp_numerically(b, out, point.clone(), delta, epsilon);
    test_vjp_numerically(b, out, point.clone(), delta, epsilon);
    test_hvp_numerically(b, out, point.clone(), delta, epsilon);
    test_hvp_cross_numerically(a, b, out, point.clone(), delta, epsilon);
    test_hvp_cross_numerically(b, a, out, point, delta, epsilon);
}

#[allow(unused)]
pub fn compose<B: Backend>(delta: f64, epsilon: f64)
where
    SO3: ContinuousQuantity<B>,
{
    use skoro_so3::ops::SO3ComposeOps;

    let model = ModelClient::<B>::default();
    let r0 = model.add_parameter(SO3);
    let r1 = model.add_parameter(SO3);
    let out = r0.compose(&r1).unwrap();
    let point = points()
        .with_quantity(&r0, tensor::<B, 2>([R0], &Default::default()))
        .with_quantity(&r1, tensor::<B, 2>([R1], &Default::default()))
        .build();
    check_binary(&r0, &r1, &out, point, delta, epsilon);
}

/// A vector operand, and a `3 × 2` matrix operand whose columns are rotated alike.
pub const V: [f64; 3] = [1.5, -2.0, 0.75];
pub const M: [f64; 6] = [1.5, -0.5, -2.0, 1.25, 0.75, 2.0];

#[allow(unused)]
pub fn rotate<B: Backend>(delta: f64, epsilon: f64)
where
    SO3: ContinuousQuantity<B>,
{
    use skoro_riemann::{euclidean::EuclideanD, matrix::EuclideanMatrixD};
    use skoro_so3::ops::SO3RotateOps;

    let model = ModelClient::<B>::default();
    let r = model.add_parameter(SO3);
    let v = model.add_parameter(EuclideanD(3));
    let m = model.add_parameter(EuclideanMatrixD(3, 2));
    let point = points()
        .with_quantity(&r, tensor::<B, 2>([R1], &Default::default()))
        .with_quantity(&v, tensor::<B, 2>([V], &Default::default()))
        .with_quantity(&m, tensor::<B, 2>([M], &Default::default()))
        .build();
    let out = r.rotate(&v).unwrap();
    check_binary(&r, &v, &out, point.clone(), delta, epsilon);
    let out = r.rotate(&m).unwrap();
    check_binary(&r, &m, &out, point, delta, epsilon);
}

#[allow(unused)]
pub fn unrotate<B: Backend>(delta: f64, epsilon: f64)
where
    SO3: ContinuousQuantity<B>,
{
    use skoro_riemann::{euclidean::EuclideanD, matrix::EuclideanMatrixD};
    use skoro_so3::ops::SO3RotateOps;

    let model = ModelClient::<B>::default();
    let r = model.add_parameter(SO3);
    let v = model.add_parameter(EuclideanD(3));
    let m = model.add_parameter(EuclideanMatrixD(3, 2));
    let point = points()
        .with_quantity(&r, tensor::<B, 2>([R1], &Default::default()))
        .with_quantity(&v, tensor::<B, 2>([V], &Default::default()))
        .with_quantity(&m, tensor::<B, 2>([M], &Default::default()))
        .build();
    let out = r.unrotate(&v).unwrap();
    check_binary(&r, &v, &out, point.clone(), delta, epsilon);
    let out = r.unrotate(&m).unwrap();
    check_binary(&r, &m, &out, point, delta, epsilon);
}

#[macro_export]
macro_rules! test_group {
    ($delta:literal, $epsilon:literal) => {
        use super::B;
        use super::common::group;

        /// `r₀ r₁`, whose chart derivative is `r₁ᵀ a₀ + a₁`.
        #[test]
        pub fn compose() {
            group::compose::<B>($delta, $epsilon);
        }

        /// `r v`, for a vector and for the columns of a matrix.
        #[test]
        pub fn rotate() {
            group::rotate::<B>($delta, $epsilon);
        }

        /// `rᵀ v`, for a vector and for the columns of a matrix.
        #[test]
        pub fn unrotate() {
            group::unrotate::<B>($delta, $epsilon);
        }
    };
}
