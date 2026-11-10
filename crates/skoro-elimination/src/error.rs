use std::fmt;

/// What an elimination refuses.
#[derive(Clone, Debug, PartialEq)]
#[non_exhaustive]
pub enum Error {
    /// A structure that names a block twice in a factor, out of order, or a block that does not
    /// exist.
    Structure { factor: usize },
    /// Rows, a linear term, a damping diagonal or a list of blocks whose shape does not fit the
    /// structure: the factor's, when it is one factor's rows.
    Shape { factor: Option<usize> },
    /// A block its rows do not determine.
    Undetermined { block: usize },
    /// An answer that needs every block eliminated, asked of an elimination that kept some.
    Incomplete,
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Error::Structure { factor } => write!(
                f,
                "factor {factor} does not name distinct blocks of the structure in increasing order"
            ),
            Error::Shape {
                factor: Some(factor),
            } => write!(f, "the rows of factor {factor} do not fit the structure"),
            Error::Shape { factor: None } => f.write_str("a value does not fit the structure"),
            Error::Undetermined { block } => {
                write!(f, "block {block} is not determined by its rows")
            }
            Error::Incomplete => f.write_str("the answer needs every block eliminated"),
        }
    }
}

impl std::error::Error for Error {}
