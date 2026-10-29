//! The core of skoro: quantities and their spaces, operations and their derivatives, the
//! compilation and probabilistic graphs, the writer, queries, generation and retirement.

mod address;
mod compile;
mod composite;
mod embedded;
mod error;
mod generate;
mod id;
mod into_host;
mod key;
mod layout;
mod model;
mod operation;
mod ops;
pub mod primitive;
mod program;
mod q;
mod query;
mod registry;
mod retire;
mod retraction;
pub mod scope;
mod space;
mod term;
pub mod testing;
mod writer;

pub use address::{Address, Segment};
pub use composite::Composite;
pub use embedded::Embedded;
pub use error::{Error, Failure};
pub use generate::Generated;
pub use id::NodeId;
pub use into_host::{IntoHostValue, Rows};
pub use key::Key;
pub use layout::Layout;
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
pub use registry::{MakeEmbedded, MakeOperation, Registry};
pub use retire::{Addresses, Plan, Replacement, Retirement};
pub use space::{Block, Space};
pub use term::{
    AnyNumber, AtMostOne, ExactlyOne, Multiplicity, Proposal, Role, Sample, Sampler, Term,
    TermHandle,
};
pub use writer::{Reader, Writer};
