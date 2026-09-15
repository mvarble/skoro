use skoro_backend::Backend;
use skoro_graph::quantity::ContinuousQuantity;
use skoro_model::{
    hash_map_builders::points,
    model::ModelClient,
    ops::AssociatedTangentVector,
    test_utils::{
        tensor, test_hvp_cross_numerically, test_hvp_numerically, test_jvp_numerically,
        test_vjp_numerically,
    },
};
use skoro_riemann::euclidean::Euclidean;

use skoro_so3::quantity::SO3;

use super::retraction::R0;

/// `Exp((-0.15, 0.45, -0.25))`; ~1.09 rad away from [`R0`], so `Δr` is far enough from the
/// identity that a stray transpose changes the answer.
const R1: [f64; 9] = [
    0.8706442127616613,
    0.2052427441778379,
    0.4470504118631115,
    -0.2711409754124633,
    0.9585085210744951,
    0.08799992318156921,
    -0.4104402833994308,
    -0.1978303085726115,
    0.8901696146089577,
];

#[allow(unused)]
pub fn inverse_retract<B: Backend>(delta: f64, epsilon: f64)
where
    SO3: ContinuousQuantity<B> + AssociatedTangentVector<Euclidean<3>, B>,
{
    let model = ModelClient::<B>::default();
    let attitude = model.add_parameter(SO3);
    let other_attitude = model.add_parameter(SO3);
    let out = attitude
        .inverse_retract::<Euclidean<3>>(&other_attitude)
        .unwrap();
    let point = points()
        .with_quantity(&attitude, tensor::<B, 2>([R0], &Default::default()))
        .with_quantity(&other_attitude, tensor::<B, 2>([R1], &Default::default()))
        .build();
    test_jvp_numerically(&attitude, &out, point.clone(), delta, epsilon);
    test_vjp_numerically(&attitude, &out, point.clone(), delta, epsilon);
    test_hvp_numerically(&attitude, &out, point.clone(), delta, epsilon);
    test_jvp_numerically(&other_attitude, &out, point.clone(), delta, epsilon);
    test_vjp_numerically(&other_attitude, &out, point.clone(), delta, epsilon);
    test_hvp_numerically(&other_attitude, &out, point.clone(), delta, epsilon);
    test_hvp_cross_numerically(
        &attitude,
        &other_attitude,
        &out,
        point.clone(),
        delta,
        epsilon,
    );
    test_hvp_cross_numerically(
        &other_attitude,
        &attitude,
        &out,
        point.clone(),
        delta,
        epsilon,
    );
}

#[macro_export]
macro_rules! test_inverse_retract {
    ($delta:literal, $epsilon:literal) => {
        use super::B;
        use super::common::inverse_retraction;

        #[test]
        pub fn inverse_retract() {
            inverse_retraction::inverse_retract::<B>($delta, $epsilon);
        }
    };
}
