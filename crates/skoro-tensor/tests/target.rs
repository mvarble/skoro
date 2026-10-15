#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)] // a test crate
//! The checks a target makes when it is constructed.

use skoro_tensor::{Device, Precision, Target, TensorError};

#[test]
fn target_refuses_an_autodiff_device() {
    let result = Target::new(Device::flex().autodiff(), Precision::Double);
    assert!(
        matches!(result, Err(TensorError::AutodiffDevice { .. })),
        "{result:?}"
    );
}

#[test]
// burn deprecates its NdArray backend, which is what makes it a backend skoro refuses.
#[allow(deprecated)]
fn target_refuses_an_unsupported_backend() {
    let result = Target::new(Device::ndarray(), Precision::Double);
    assert!(
        matches!(result, Err(TensorError::UnsupportedDevice { .. })),
        "{result:?}"
    );
}

#[test]
fn flex_computes_at_both_precisions() {
    for precision in [Precision::Single, Precision::Double] {
        let target = Target::new(Device::flex(), precision).unwrap();
        assert_eq!(target.precision(), precision);
    }
}

#[cfg(feature = "wgpu")]
mod wgpu {
    use super::*;

    fn adapter() -> Device {
        skoro_tensor::test_adapter(skoro_tensor::wgpu_devices()).unwrap()
    }

    #[test]
    fn target_refuses_an_unsupported_precision() {
        let device = adapter();
        let result = Target::new(device.clone(), Precision::Double);
        if device.supports_dtype(Precision::Double.float_dtype()) {
            assert!(result.is_ok(), "{result:?}");
        } else {
            assert!(
                matches!(result, Err(TensorError::UnsupportedPrecision { .. })),
                "{result:?}"
            );
        }
        assert!(Target::new(device, Precision::Single).is_ok());
    }

    #[test]
    fn wgpu_tests_fail_without_an_adapter() {
        let error = skoro_tensor::test_adapter(Vec::new()).unwrap_err();
        assert!(error.contains("no wgpu adapter"), "{error}");
    }
}
