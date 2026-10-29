use std::fmt;

use burn::tensor::{DType, Device};
use burn_dispatch::DispatchDevice;

use crate::TensorError;

/// The precision of a target's floating-point values.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum Precision {
    /// `f32`.
    Single,
    /// `f64`.
    Double,
}

impl Precision {
    /// The data type of a float tensor at this precision.
    pub fn float_dtype(self) -> DType {
        match self {
            Precision::Single => DType::F32,
            Precision::Double => DType::F64,
        }
    }
}

impl fmt::Display for Precision {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Precision::Single => "single",
            Precision::Double => "double",
        })
    }
}

/// Where a program computes, and at what precision. Checked once, here.
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct Target {
    device: Device,
    precision: Precision,
    wide_integers: bool,
}

impl Target {
    /// Refuses a device that is not Flex or wgpu (`UnsupportedDevice`), an autodiff device
    /// (`AutodiffDevice`), and a precision the device does not compute in (`UnsupportedPrecision`).
    pub fn new(device: Device, precision: Precision) -> Result<Target, TensorError> {
        let name = format!("{device:?}");
        // Checked first: an autodiff device wraps a device that may otherwise be supported.
        if device.is_autodiff() {
            return Err(TensorError::AutodiffDevice { device: name });
        }
        if !supported(&device) {
            return Err(TensorError::UnsupportedDevice { device: name });
        }
        if !device.supports_dtype(precision.float_dtype()) {
            return Err(TensorError::UnsupportedPrecision {
                device: name,
                precision,
            });
        }
        let wide_integers = device.supports_dtype(DType::I64);
        Ok(Target {
            device,
            precision,
            wide_integers,
        })
    }

    /// The Flex device at double precision: where the model folds constants.
    pub fn host() -> Target {
        // The Flex device computes at double precision with 64-bit integers, which
        // `wide_integers_on_the_host` checks against `new`.
        Target {
            device: Device::flex(),
            precision: Precision::Double,
            wide_integers: true,
        }
    }

    pub fn device(&self) -> &Device {
        &self.device
    }

    pub fn precision(&self) -> Precision {
        self.precision
    }

    /// The data type of this target's float tensors.
    pub fn float_dtype(&self) -> DType {
        self.precision.float_dtype()
    }

    /// The data type of this target's integer tensors: `I64` where the device has it.
    pub fn int_dtype(&self) -> DType {
        if self.wide_integers {
            DType::I64
        } else {
            DType::I32
        }
    }
}

/// The Flex device and wgpu devices of the cubecl backend. Every other cubecl runtime is refused
/// by having no arm here: none can be built without an LLVM toolchain, so none is tested.
fn supported(device: &Device) -> bool {
    match device.as_dispatch() {
        DispatchDevice::Flex(_) => true,
        #[cfg(feature = "wgpu")]
        DispatchDevice::Cube(burn_dispatch::devices::CubeDevice::Wgpu(_)) => true,
        // Unreachable when burn is built with Flex and wgpu alone, but Cargo unifies features, so
        // another crate in a build may turn on a backend skoro does not test.
        #[allow(unreachable_patterns)]
        _ => false,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn wide_integers_on_the_host() {
        let host = Target::host();
        let checked = Target::new(Device::flex(), Precision::Double).unwrap();
        assert_eq!(host, checked);
    }
}
