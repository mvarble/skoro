use std::fmt;
use std::panic::Location;

use skoro_tensor::TensorError;

use crate::{Address, NodeId, Space};

/// What the core refuses, with what a caller needs to name the fault.
#[derive(Clone, Debug, PartialEq)]
#[non_exhaustive]
pub enum Error {
    /// The tensor layer refused a device, a precision, a tensor or a value.
    Tensor(TensorError),
    /// A continuous space was needed: a product's factors, for one.
    NotContinuous { space: Space },
    /// An operation does not take inputs of these spaces, or of this many.
    SpaceMismatch {
        operation: &'static str,
        inputs: Vec<Space>,
    },
    /// An operation takes inputs of these kinds of space, but not of these sizes.
    ShapeMismatch {
        operation: &'static str,
        inputs: Vec<Space>,
    },
    /// An operation was attempted with no building scope open, and its handle was then used in
    /// one.
    OutsideScope,
    /// A handle from a scope that failed was used in another.
    FailedOperand,
    /// A handle the model does not hold: from another model, or a retired quantity.
    UnknownNode { id: NodeId },
    /// A quantity already lives at this address.
    AddressTaken { address: Address },
    /// No quantity lives at this address.
    UnknownAddress { address: Address },
    /// A value that does not fit its space: its storage, its kind, or its range.
    ValueShape { space: Space },
    /// Values that pin different batches, or a batch other than the one a pass runs at.
    BatchSize { expected: usize, found: usize },
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Error::Tensor(error) => write!(f, "{error}"),
            Error::NotContinuous { space } => write!(f, "{space:?} is not a continuous space"),
            Error::SpaceMismatch { operation, inputs } => {
                write!(f, "{operation} does not take inputs of {inputs:?}")
            }
            Error::OutsideScope => f.write_str(
                "an operation was attempted with no building scope open, and its handle used here",
            ),
            Error::FailedOperand => f.write_str("a handle from a scope that failed was used here"),
            Error::UnknownNode { id } => write!(f, "the model does not hold node {id:?}"),
            Error::AddressTaken { address } => write!(f, "a quantity already lives at {address}"),
            Error::UnknownAddress { address } => write!(f, "no quantity lives at {address}"),
            Error::ValueShape { space } => write!(f, "the value does not fit {space:?}"),
            Error::BatchSize { expected, found } => {
                write!(f, "a batch of {found} rows where {expected} were expected")
            }
            Error::ShapeMismatch { operation, inputs } => {
                write!(
                    f,
                    "{operation} takes inputs of these kinds, but not of the sizes {inputs:?}"
                )
            }
        }
    }
}

impl std::error::Error for Error {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Error::Tensor(error) => Some(error),
            _ => None,
        }
    }
}

impl From<TensorError> for Error {
    fn from(error: TensorError) -> Error {
        Error::Tensor(error)
    }
}

/// The first error a building scope recorded, and the line of user code that caused it.
#[derive(Clone, Debug, PartialEq)]
pub struct Failure {
    pub error: Error,
    pub location: &'static Location<'static>,
}

impl fmt::Display for Failure {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{} (at {})", self.error, self.location)
    }
}

impl std::error::Error for Failure {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        Some(&self.error)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn failure_names_its_error_and_line() {
        let error = Error::from(TensorError::IntegerOutOfRange { value: 1 << 40 });
        let failure = Failure {
            error: error.clone(),
            location: Location::caller(),
        };
        let shown = failure.to_string();
        assert!(shown.starts_with(&error.to_string()), "{shown}");
        assert!(
            shown.contains(&format!("error.rs:{}", failure.location.line())),
            "{shown}"
        );
        let source = std::error::Error::source(&failure).map(|s| s.to_string());
        assert_eq!(source, Some(error.to_string()));
    }
}
