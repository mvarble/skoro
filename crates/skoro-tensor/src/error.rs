use std::fmt;

/// What the tensor layer refuses.
#[derive(Clone, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum TensorError {
    /// A device that is neither the Flex device nor a wgpu device of the cubecl backend.
    UnsupportedDevice { device: String },
    /// A device with automatic differentiation on: skoro differentiates by its own passes.
    AutodiffDevice { device: String },
    /// A precision the device does not compute in.
    UnsupportedPrecision {
        device: String,
        precision: crate::Precision,
    },
    /// A burn tensor of another data type or device than the target's.
    ForeignTensor { expected: String, found: String },
    /// A discrete value that does not fit the target's integers.
    IntegerOutOfRange { value: i64 },
}

impl fmt::Display for TensorError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            TensorError::UnsupportedDevice { device } => write!(
                f,
                "{device} is not a device skoro supports: use the Flex device or a wgpu device"
            ),
            TensorError::AutodiffDevice { device } => write!(
                f,
                "{device} has automatic differentiation on, which skoro never runs a model with"
            ),
            TensorError::UnsupportedPrecision { device, precision } => {
                write!(f, "{device} does not compute at {precision} precision")
            }
            TensorError::ForeignTensor { expected, found } => {
                write!(
                    f,
                    "a tensor of {found} was given where {expected} was expected"
                )
            }
            TensorError::IntegerOutOfRange { value } => {
                write!(f, "{value} does not fit the target's integers")
            }
        }
    }
}

impl std::error::Error for TensorError {}
