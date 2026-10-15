#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)] // a test crate
//! The tensor wrappers keep their target's precision and integers, and host values load as they
//! say. Each check runs on the Flex device at both precisions and, with the `wgpu` feature, on the
//! test adapter at single precision.

use skoro_tensor::burn::{DType, Int, Tensor};
use skoro_tensor::{
    Device, Frame, HostData, HostValue, Integers, Points, Precision, Target, TensorError, Value,
};

fn targets() -> Vec<Target> {
    let flex = [Precision::Single, Precision::Double]
        .map(|precision| Target::new(Device::flex(), precision).unwrap());
    #[cfg(feature = "wgpu")]
    let wgpu = Some(
        Target::new(
            skoro_tensor::test_adapter(skoro_tensor::wgpu_devices()).unwrap(),
            Precision::Single,
        )
        .unwrap(),
    );
    #[cfg(not(feature = "wgpu"))]
    let wgpu = None;
    flex.into_iter().chain(wgpu).collect()
}

fn points(data: &[f64], rows: usize, storage: usize, target: &Target) -> Tensor<2> {
    Points::from_host(data, rows, storage, target).into_burn()
}

fn assert_close(found: &[f64], expected: &[f64]) {
    assert_eq!(found.len(), expected.len());
    for (a, b) in found.iter().zip(expected) {
        assert!(
            (a - b).abs() <= 1e-5 * b.abs().max(1.0),
            "{found:?} != {expected:?}"
        );
    }
}

#[test]
fn two_precisions_share_a_device() {
    let single = Target::new(Device::flex(), Precision::Single).unwrap();
    let double = Target::new(Device::flex(), Precision::Double).unwrap();
    let a = points(&[1.0, 2.0], 1, 2, &single);
    let b = points(&[1.0, 2.0], 1, 2, &double);
    let a = Points::from_burn(a.clone().exp() + a, &single).unwrap();
    let b = Points::from_burn(b.clone().exp() + b, &double).unwrap();
    let expected = [1.0f64.exp() + 1.0, 2.0f64.exp() + 2.0];
    assert_close(&a.to_host(), &expected);
    assert_eq!(b.to_host(), expected);
    assert_eq!(a.into_burn().dtype(), DType::F32);
    assert_eq!(b.into_burn().dtype(), DType::F64);
}

#[test]
fn arithmetic_keeps_its_precision() {
    for target in targets() {
        let x = points(&[1.0, 2.0, 3.0, 4.0], 2, 2, &target);
        let y = points(&[0.5, 0.25, 2.0, 1.5], 2, 2, &target);
        let mask = x.clone().greater_equal_elem(2.5);
        let indices = Tensor::<2, Int>::from_data(
            burn_data(&[1, 0, 0, 1]),
            (target.device(), target.int_dtype()),
        );
        let results: Vec<(&str, Tensor<2>)> = vec![
            ("add", x.clone() + y.clone()),
            ("sub", x.clone() - y.clone()),
            ("neg", -x.clone()),
            ("mul", x.clone() * y.clone()),
            ("div", x.clone() / y.clone()),
            ("matmul", x.clone().matmul(y.clone())),
            ("mul_scalar", x.clone().mul_scalar(3.0)),
            ("exp", x.clone().exp()),
            ("log", x.clone().log()),
            ("sin", x.clone().sin()),
            ("cos", x.clone().cos()),
            ("atan2", x.clone().atan2(y.clone())),
            ("recip", x.clone().recip()),
            ("sum_dim", x.clone().sum_dim(1)),
            ("sum", x.clone().sum().reshape([1, 1])),
            ("slice", x.clone().slice([0..2, 1..2])),
            ("cat", Tensor::cat(vec![x.clone(), y.clone()], 1)),
            ("swap_dims", x.clone().swap_dims(0, 1)),
            ("reshape", x.clone().reshape([1, 4])),
            ("expand", x.clone().slice([0..1, 0..2]).expand([3, 2])),
            ("gather", x.clone().gather(1, indices)),
            ("mask_where", x.clone().mask_where(mask, y.clone())),
            ("zeros_like", x.zeros_like()),
            ("ones_like", x.ones_like()),
        ];
        for (name, result) in results {
            assert_eq!(result.dtype(), target.float_dtype(), "{name} on {target:?}");
            assert!(Points::from_burn(result, &target).is_ok(), "{name}");
        }
    }
}

fn burn_data(data: &[i64]) -> skoro_tensor::burn::TensorData {
    skoro_tensor::burn::TensorData::new(data.to_vec(), [2, 2])
}

#[test]
fn integers_to_points_takes_the_target_precision() {
    for target in targets() {
        // 2^24 + 1 has no single-precision representation: a conversion through single precision
        // would lose it in a double-precision target.
        let values = [16_777_217, -3, 0, 7];
        let integers = Integers::from_host(&values, 2, 2, &target).unwrap();
        let points = integers.to_points(&target);
        let expected: Vec<f64> = values.iter().map(|&v| v as f64).collect();
        if target.precision() == Precision::Double {
            assert_eq!(points.to_host(), expected);
        } else {
            assert_close(&points.to_host(), &expected);
        }
        assert_eq!(points.into_burn().dtype(), target.float_dtype());
    }
}

#[test]
fn comparison_takes_the_target_integers() {
    for target in targets() {
        let x = points(&[-1.0, 0.0, 2.0, -0.5], 2, 2, &target);
        let integers = Integers::from_mask(x.greater_equal_elem(0.0), &target).unwrap();
        assert_eq!(integers.to_host(), vec![0, 1, 1, 0]);
        assert_eq!(integers.into_burn().dtype(), target.int_dtype());
    }
}

#[test]
fn from_burn_refuses_a_foreign_tensor() {
    let single = Target::new(Device::flex(), Precision::Single).unwrap();
    let double = Target::new(Device::flex(), Precision::Double).unwrap();
    for (made, taken) in [(&single, &double), (&double, &single)] {
        let tensor = points(&[1.0], 1, 1, made);
        assert!(matches!(
            Points::from_burn(tensor, taken),
            Err(TensorError::ForeignTensor { .. })
        ));
        let frame = Frame::zeros(1, 1, 1, made).into_burn();
        assert!(matches!(
            Frame::from_burn(frame, taken),
            Err(TensorError::ForeignTensor { .. })
        ));
    }
    #[cfg(feature = "wgpu")]
    {
        let wgpu = targets().pop().unwrap();
        let tensor = points(&[1.0], 1, 1, &wgpu);
        assert!(matches!(
            Points::from_burn(tensor, &single),
            Err(TensorError::ForeignTensor { .. })
        ));
        let integers = Integers::zeros(1, 1, &single).into_burn();
        assert!(matches!(
            Integers::from_burn(integers, &wgpu),
            Err(TensorError::ForeignTensor { .. })
        ));
    }
}

#[test]
fn host_round_trip_is_exact_in_double() {
    let target = Target::host();
    let data = [
        0.1,
        -0.0,
        f64::MIN_POSITIVE / 3.0,
        f64::MAX,
        -1.0e-300,
        std::f64::consts::PI,
    ];
    let round = Points::from_host(&data, 3, 2, &target).to_host();
    let bits = |v: &[f64]| v.iter().map(|x| x.to_bits()).collect::<Vec<_>>();
    assert_eq!(bits(&round), bits(&data));

    let frame = Frame::from_host(&data, 1, 3, 2, &target).to_host();
    assert_eq!(bits(&frame), bits(&data));

    let integers = [i64::MIN, -1, 0, i64::MAX];
    let round = Integers::from_host(&integers, 2, 2, &target)
        .unwrap()
        .to_host();
    assert_eq!(round, integers);
}

#[test]
fn narrow_integers_refuse_what_does_not_fit() {
    for target in targets() {
        let result = Integers::from_host(&[1, i64::from(i32::MAX) + 1], 1, 2, &target);
        if target.int_dtype() == DType::I32 {
            assert!(matches!(
                result,
                Err(TensorError::IntegerOutOfRange { value }) if value == i64::from(i32::MAX) + 1
            ));
        } else {
            assert!(result.is_ok());
        }
    }
}

#[test]
fn shared_value_expands_to_the_batch() {
    for target in targets() {
        let shared = HostValue::new(None, 2, HostData::Continuous(vec![1.5, -2.0])).unwrap();
        let Some(Value::Continuous(points)) = shared.load(3, &target).unwrap() else {
            panic!("a continuous value");
        };
        assert_eq!((points.rows(), points.storage()), (3, 2));
        assert_eq!(points.to_host(), vec![1.5, -2.0, 1.5, -2.0, 1.5, -2.0]);

        let shared = HostValue::new(None, 1, HostData::Discrete(vec![4])).unwrap();
        let Some(Value::Discrete(integers)) = shared.load(2, &target).unwrap() else {
            panic!("a discrete value");
        };
        assert_eq!(integers.to_host(), vec![4, 4]);
    }
}

#[test]
fn per_row_value_pins_the_batch() {
    for target in targets() {
        let rows = HostValue::new(Some(2), 1, HostData::Continuous(vec![1.0, 2.0])).unwrap();
        let value = rows.load(2, &target).unwrap().unwrap();
        assert_eq!(value.rows(), 2);
        assert!(rows.load(3, &target).unwrap().is_none());

        // One row is a value for a batch of one, never broadcast to several.
        let one = HostValue::new(Some(1), 2, HostData::Discrete(vec![1, 2])).unwrap();
        assert!(one.load(1, &target).unwrap().is_some());
        assert!(one.load(4, &target).unwrap().is_none());
    }
}

#[test]
fn host_value_holds_rows_times_storage() {
    assert!(HostValue::new(None, 3, HostData::Continuous(vec![1.0, 2.0])).is_none());
    assert!(HostValue::new(Some(2), 2, HostData::Discrete(vec![1, 2, 3])).is_none());
    let value = HostValue::new(Some(2), 0, HostData::Continuous(Vec::new())).unwrap();
    assert_eq!((value.rows(), value.storage()), (Some(2), 0));
}
