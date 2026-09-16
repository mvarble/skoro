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

/// The binary's float policy is `dtype`, and a host read and write keeps both the dtype and, at
/// `f64`, full precision.
pub fn float_policy_is<B: Backend>(dtype: skoro_backend::DType) {
    use skoro_backend::{TensorData, host};

    let device = Default::default();
    assert_eq!(Tensor::<B, 2>::zeros([2, 3], &device).dtype(), dtype);

    let value = 1.0 + 1e-12;
    let tensor = Tensor::<B, 1>::from_data(TensorData::new(vec![value, 2.0], [2]), &device);
    assert_eq!(tensor.dtype(), dtype);
    let back = host::write::<B, f64, 1>(host::read(tensor.clone()), [2], &device, tensor.dtype());
    assert_eq!(back.dtype(), dtype);
    let read: Vec<f64> = host::read(back);
    if dtype == skoro_backend::DType::F64 {
        assert_eq!(read, vec![value, 2.0]);
    }
}

#[allow(unused)]
#[macro_export]
macro_rules! float_policy_is {
    ($B:ident, $dtype:ident) => {
        #[test]
        fn float_policy_is() {
            common::float_policy_is::<$B>(skoro_backend::DType::$dtype);
        }
    };
}
