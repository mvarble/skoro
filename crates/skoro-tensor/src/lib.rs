//! The tensor layer of skoro: the tensor shapes skoro computes with, each created at an explicit
//! precision on a device skoro supports.

mod error;
mod target;

pub use burn::tensor::Device;
pub use error::TensorError;
pub use target::{Precision, Target};

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
    Device::enumerate(burn::tensor::DeviceType::Wgpu)
        .into_iter()
        .collect()
}
