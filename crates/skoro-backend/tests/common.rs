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

/// `abs` and `recip` on the two layouts burn-ndarray's SIMD unary path used to get wrong: a
/// `select` along a trailing axis and a contiguous `swap_dims`. A second handle is kept on each
/// input so no kernel can take an in-place path, and there are enough rows to clear a 32-element
/// SIMD threshold. Kept as a backend-agnostic layout check.
pub fn unary_kernels_on_non_standard_layouts<B: Backend>() {
    use skoro_backend::{Int, TensorData};

    let device = Default::default();
    let rows = 64;
    // Signs alternate and magnitudes stay away from zero, so `abs` is visibly not the identity
    // and `recip` stays well conditioned.
    let value = |i: usize| {
        let magnitude = 0.5 + (i as f64 * 0.53).sin().abs();
        if i.is_multiple_of(2) {
            magnitude
        } else {
            -magnitude
        }
    };
    let check = |what: &str, got: Vec<f64>, inputs: &[f64], f: fn(f64) -> f64| {
        assert_eq!(got.len(), inputs.len(), "{what}");
        for (i, (g, x)) in got.iter().zip(inputs).enumerate() {
            let want = f(*x);
            assert!(
                (g - want).abs() <= 1e-5 * want.abs().max(1.0),
                "{what}: entry {i} is {g}, expected {want}"
            );
        }
    };
    let to_host = |t: Tensor<B, 2>| t.into_data().convert::<f64>().to_vec::<f64>().unwrap();
    let to_host3 = |t: Tensor<B, 3>| t.into_data().convert::<f64>().to_vec::<f64>().unwrap();

    // A `select` of columns 0 and 2 from a `[rows, 3]` tensor.
    let packed: Vec<f64> = (0..3 * rows).map(value).collect();
    let tensor = Tensor::<B, 2>::from_data(TensorData::new(packed.clone(), [rows, 3]), &device);
    let indices = Tensor::<B, 1, Int>::from_data(
        TensorData::new(vec![0i64, 2], [2]).convert::<B::IntElem>(),
        &device,
    );
    let selected = tensor.select(1, indices);
    let host_selected: Vec<f64> = packed.chunks(3).flat_map(|r| [r[0], r[2]]).collect();
    let _held = selected.clone();
    check(
        "abs of a select",
        to_host(selected.clone().abs()),
        &host_selected,
        f64::abs,
    );
    check(
        "recip of a select",
        to_host(selected.recip()),
        &host_selected,
        f64::recip,
    );

    // A contiguous `[rows, 2, 2]` tensor with its last two axes swapped.
    let data: Vec<f64> = (0..4 * rows).map(value).collect();
    let tensor = Tensor::<B, 3>::from_data(TensorData::new(data.clone(), [rows, 2, 2]), &device);
    let swapped = tensor.swap_dims(1, 2);
    let mut host_swapped = vec![0.0; 4 * rows];
    for s in 0..rows {
        for r in 0..2 {
            for c in 0..2 {
                host_swapped[4 * s + 2 * r + c] = data[4 * s + 2 * c + r];
            }
        }
    }
    let _held = swapped.clone();
    check(
        "abs of a swap_dims",
        to_host3(swapped.clone().abs()),
        &host_swapped,
        f64::abs,
    );
    check(
        "recip of a swap_dims",
        to_host3(swapped.recip()),
        &host_swapped,
        f64::recip,
    );
}

#[allow(unused)]
#[macro_export]
macro_rules! unary_kernels_on_non_standard_layouts {
    ($B:ident) => {
        #[test]
        fn unary_kernels_on_non_standard_layouts() {
            common::unary_kernels_on_non_standard_layouts::<$B>();
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
