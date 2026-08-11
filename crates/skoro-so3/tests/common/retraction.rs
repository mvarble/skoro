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

/// `Exp((0.3, -0.2, 0.5))`.
pub const R0: [f64; 9] = [
    0.8595338985586632,
    -0.497991537002922,
    -0.1149169539363668,
    0.4398676329582309,
    0.8353156052067086,
    -0.3297943376922551,
    0.2602267140480944,
    0.2329211642844366,
    0.937032437284918,
];

#[allow(unused)]
pub fn retract<B: Backend>(delta: B::FloatElem, epsilon: B::FloatElem)
where
    SO3: ContinuousQuantity<B> + AssociatedTangentVector<Euclidean<3>, B>,
    B::FloatElem: approx::AbsDiffEq<B::FloatElem, Epsilon = B::FloatElem> + Copy,
{
    let model = ModelClient::<B>::default();
    let attitude = model.add_parameter(SO3);
    let tangent = model.add_parameter(Euclidean::<3>);
    let out = attitude.retract(&tangent).unwrap();
    let point = points()
        .with_quantity(&attitude, tensor::<B, 2>([R0], &Default::default()))
        .with_tensor(
            tangent.id(),
            tensor::<B, 2>([[0.4, -0.3, 0.6]], &Default::default()),
        )
        .build();
    test_jvp_numerically(&attitude, &out, point.clone(), delta, epsilon);
    test_vjp_numerically(&attitude, &out, point.clone(), delta, epsilon);
    test_hvp_numerically(&attitude, &out, point.clone(), delta, epsilon);
    test_jvp_numerically(&tangent, &out, point.clone(), delta, epsilon);
    test_vjp_numerically(&tangent, &out, point.clone(), delta, epsilon);
    test_hvp_numerically(&tangent, &out, point.clone(), delta, epsilon);
    test_hvp_cross_numerically(&tangent, &attitude, &out, point.clone(), delta, epsilon);
    test_hvp_cross_numerically(&attitude, &tangent, &out, point.clone(), delta, epsilon);
}

#[macro_export]
macro_rules! test_retract {
    ($delta:literal, $epsilon:literal) => {
        use super::B;
        use super::common::retraction;

        #[test]
        pub fn retract() {
            retraction::retract::<B>($delta, $epsilon);
        }
    };
}
