//! The core of skoro: quantities and their spaces, operations and their derivatives, the
//! compilation and probabilistic graphs, the writer, queries, generation and retirement.

mod address;
mod compile;
mod composite;
mod error;
mod id;
mod into_host;
mod key;
mod model;
mod operation;
mod ops;
pub mod primitive;
mod program;
mod q;
mod query;
mod registry;
mod retraction;
pub mod scope;
mod space;
pub mod testing;
mod writer;

pub use address::{Address, Segment};
pub use composite::Composite;
pub use error::{Error, Failure};
pub use id::NodeId;
pub use into_host::{IntoHostValue, Rows};
pub use key::Key;
pub use model::{Node, Snapshot};
pub use operation::{
    FirstOrder, Forward, Kernel, Lowered, Lowering, Operation, Order, Param, Pullback, SecondOrder,
    SecondPullback,
};
pub use ops::{QAdd, QDiv, QMatMul, QMul, QNeg, QScale, QShift, QSub};
pub use program::{Frames, Free, Pass, Program, Second, Values};
pub use q::{
    Boolean, Categorical, CategoricalDyn, Declares, Integer, Matrix, MatrixDyn, Positive,
    PositiveDyn, ProductDyn, Q, QDyn, Real, Typed, Vector, VectorDyn,
};
pub use query::{Observations, Query};
pub use registry::{MakeOperation, Registry};
pub use space::{Block, Space};
pub use writer::{Reader, Writer};
