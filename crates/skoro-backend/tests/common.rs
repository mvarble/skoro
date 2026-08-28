use skoro_backend::{
    Backend, CotangentFrame, CotangentVector, DiscretePoint, Point, TangentFrame, TangentVector,
    Tensor, UnbatchedDiscretePoint, UnbatchedPoint,
};

pub fn exercise_api<B: Backend>() {
    let _ = Tensor::<B, 1>::from_floats([1.0, 2.0, 3.0], &Default::default());
    let _ = UnbatchedPoint::<B>::from_floats([1.0, 2.0, 3.0], &Default::default());
    let _ = UnbatchedDiscretePoint::<B>::from_ints([1, 2, 3], &Default::default());

    let _ = Tensor::<B, 2>::from_floats([[1.0, 2.0], [3.0, 4.0]], &Default::default());
    let _ = Point::<B>::from_floats([[1.0, 2.0], [3.0, 4.0]], &Default::default());
    let _ = TangentVector::<B>::from_floats([[1.0, 2.0], [3.0, 4.0]], &Default::default());
    let _ = CotangentVector::<B>::from_floats([[1.0, 2.0], [3.0, 4.0]], &Default::default());
    let _ = DiscretePoint::<B>::from_ints([[1, 2], [3, 4]], &Default::default());

    let _ = Tensor::<B, 3>::from_floats(
        [[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]],
        &Default::default(),
    );
    let _ = TangentFrame::<B>::from_floats(
        [[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]],
        &Default::default(),
    );
    let _ = CotangentFrame::<B>::from_floats(
        [[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]],
        &Default::default(),
    );
}

#[allow(unused)]
#[macro_export]
macro_rules! exercise_api {
    ($B:ident) => {
        #[test]
        fn exercise_api() {
            common::exercise_api::<B>();
        }
    };
}
