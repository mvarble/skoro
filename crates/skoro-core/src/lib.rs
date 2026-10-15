//! The core of skoro: quantities and their spaces, operations and their derivatives, the
//! compilation and probabilistic graphs, the writer, queries, generation and retirement.

mod address;
mod error;
mod id;
mod key;
mod q;
mod space;

pub use address::{Address, Segment};
pub use error::{Error, Failure};
pub use id::NodeId;
pub use key::Key;
pub use q::{
    Boolean, Categorical, CategoricalDyn, Declares, Integer, Matrix, MatrixDyn, Positive,
    PositiveDyn, ProductDyn, Q, QDyn, Real, Typed, Vector, VectorDyn,
};
pub use space::{Block, Space};
