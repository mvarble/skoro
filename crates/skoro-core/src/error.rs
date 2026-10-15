use std::fmt;
use std::panic::Location;

use skoro_tensor::TensorError;

use crate::Space;

/// What the core refuses, with what a caller needs to name the fault.
#[derive(Clone, Debug, PartialEq)]
#[non_exhaustive]
pub enum Error {
    /// The tensor layer refused a device, a precision, a tensor or a value.
    Tensor(TensorError),
    /// A continuous space was needed: a product's factors, for one.
    NotContinuous { space: Space },
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Error::Tensor(error) => write!(f, "{error}"),
            Error::NotContinuous { space } => write!(f, "{space:?} is not a continuous space"),
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
