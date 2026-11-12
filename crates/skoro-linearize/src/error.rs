use std::fmt;

use skoro_core::NodeId;

/// What a linearization refuses, naming the term or the quantity.
#[derive(Clone, Debug, PartialEq)]
#[non_exhaustive]
pub enum Error {
    /// The core refused: a free discrete quantity, an integral over a quantity without a volume,
    /// a value of the wrong shape.
    Core(skoro_core::Error),
    /// The elimination refused something other than an undetermined block.
    Elimination(skoro_elimination::Error),
    /// A query of nodes, where a density query was needed.
    NotADensity,
    /// A factor that binds no residual.
    NoResidual { term: NodeId },
    /// A factor of the residual linearization whose bound scale is reached from a free quantity.
    ScaleNotFixed { term: NodeId, quantity: NodeId },
    /// A factor of the Fisher linearization whose scale is free and which binds no information
    /// root, or no tangent, which the rows of a free scale both need.
    NoInformationRoot { term: NodeId, quantity: NodeId },
    /// A free quantity its rows do not determine.
    Undetermined { quantity: NodeId },
    /// A quantity named where a free continuous one of the linearizer was needed.
    NotFree { quantity: NodeId },
    /// A retirement whose retired terms do not determine a retired quantity.
    RetiredUndetermined { quantity: NodeId },
    /// An integral, a draw or a chart linearization asked of the Fisher linearization, which
    /// approximates no integral.
    NotAnIntegral,
    /// A quantity of the retired terms that the replacement's query leaves neither free nor
    /// observed as it should: a kept quantity outside the blanket, or one observed, or a quantity
    /// given no value.
    Blanket { quantity: NodeId },
    /// A factor whose residual or Jacobian is not finite at the point linearized, at a batch
    /// row: a value outside its term's chart, such as a rotation a half-turn from its mean.
    NotFinite { term: NodeId, row: usize },
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Error::Core(error) => write!(f, "{error}"),
            Error::Elimination(error) => write!(f, "{error}"),
            Error::NotADensity => f.write_str("a linearization is of a density query"),
            Error::NoResidual { term } => write!(f, "term {term:?} binds no residual"),
            Error::ScaleNotFixed { term, quantity } => write!(
                f,
                "the scale of term {term:?} is reached from the free quantity {quantity:?}"
            ),
            Error::NoInformationRoot { term, quantity } => write!(
                f,
                "the scale of term {term:?} is free through {quantity:?}, and the term binds no \
                 information root or no tangent"
            ),
            Error::Undetermined { quantity } => {
                write!(f, "{quantity:?} is not determined by its rows")
            }
            Error::NotFree { quantity } => {
                write!(f, "{quantity:?} is not a free continuous quantity here")
            }
            Error::RetiredUndetermined { quantity } => write!(
                f,
                "the retired terms do not determine the retired quantity {quantity:?}"
            ),
            Error::NotAnIntegral => {
                f.write_str("the Fisher linearization approximates no integral")
            }
            Error::NotFinite { term, row } => {
                write!(f, "term {term:?} is not finite at row {row}")
            }
            Error::Blanket { quantity } => write!(
                f,
                "{quantity:?} is neither kept nor given a value as the replacement needs"
            ),
        }
    }
}

impl std::error::Error for Error {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Error::Core(error) => Some(error),
            Error::Elimination(error) => Some(error),
            _ => None,
        }
    }
}

impl From<skoro_core::Error> for Error {
    fn from(error: skoro_core::Error) -> Error {
        Error::Core(error)
    }
}

impl From<skoro_elimination::Error> for Error {
    fn from(error: skoro_elimination::Error) -> Error {
        Error::Elimination(error)
    }
}
