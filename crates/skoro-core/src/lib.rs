//! The core of skoro: quantities and their spaces, operations and their derivatives, the
//! compilation and probabilistic graphs, the writer, queries, generation and retirement.

mod address;
mod id;
mod key;

pub use address::{Address, Segment};
pub use id::NodeId;
pub use key::Key;
