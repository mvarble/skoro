//! The tensor layer of skoro: the tensor shapes skoro computes with, each created at an explicit
//! precision on a device skoro supports.

mod error;
mod host;
mod target;
mod value;

/// The burn the core is built with, for a feature crate's kernels.
pub use ::burn::tensor as burn;
pub use ::burn::tensor::Device;
pub use error::TensorError;
pub use host::{HostData, HostValue};
pub use target::{Precision, Target};
pub use value::{Frame, Integers, Points, Value};

/// Finds the wgpu device the tests run on, and fails naming the missing adapter when there is
/// none, so that a run with the `wgpu` feature on and no adapter cannot pass with nothing run.
#[cfg(feature = "wgpu")]
#[doc(hidden)]
pub fn test_adapter(devices: Vec<Device>) -> Result<Device, String> {
    devices.into_iter().next().ok_or_else(|| {
        "no wgpu adapter was found: the wgpu tests need a machine with one".to_string()
    })
}

/// The wgpu devices burn enumerates on this machine.
#[cfg(feature = "wgpu")]
#[doc(hidden)]
pub fn wgpu_devices() -> Vec<Device> {
    Device::enumerate(::burn::tensor::DeviceType::Wgpu)
        .into_iter()
        .collect()
}
