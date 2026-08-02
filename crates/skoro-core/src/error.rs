//! The unified, recoverable error type for interfacing with a Skoro model.
//!
//! Skoro distinguishes between two classes of failure:
//!
//! - **Internal invariants** (e.g. supplying the wrong number of inputs to a
//!   [`DiffKernel`](../../skoro_diff/diff/trait.DiffKernel.html), shape
//!   mismatches inside an operation kernel, lock poisoning). These indicate a
//!   bug in the library or its caller's use of an internal API and remain
//!   `panic!`s.
//! - **User-recoverable errors** at the model-building / evaluation boundary
//!   (e.g. referencing a node that does not exist, a quantity-type mismatch,
//!   observing a non-stochastic node, introducing a cycle, supplying points of
//!   the wrong shape, or composing dimensionally-incompatible operations).
//!   These are surfaced through [`SkoroError`].

use std::fmt;

use crate::id::Id;

/// A recoverable error produced when interfacing with a Skoro model.
///
/// This type unifies every user-recoverable failure across the model-building,
/// graph-mutation, evaluation, and operation-composition surfaces.
#[non_exhaustive]
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SkoroError {
    /// A node with the given [`Id`] does not exist in the graph.
    NodeNotFound { id: Id },
    /// The quantity type supplied by the caller does not match the type stored
    /// for the node in the graph.
    QuantityTypeMismatch { id: Id },
    /// The referenced node is not a stochastic node (required by e.g.
    /// `observe` / `unobserve` / `update_stochastic`).
    NotStochastic { id: Id },
    /// The referenced node is not a mapped node (required to become a
    /// log-density / log-likelihood).
    NotMapped { id: Id },
    /// The referenced node is not real-valued; only real values may become
    /// log-density / log-likelihood terms.
    NotReal { id: Id },
    /// A continuous (manifold-structured) quantity was required, but the
    /// referenced node is discrete.
    NotContinuous { id: Id },
    /// A stochastic quantity already has a log-density; mark the value as a
    /// log-likelihood instead.
    DuplicateDensity { stochastic_id: Id },
    /// A node was added or updated referencing a parent that does not exist.
    DanglingParent { parent_id: Id },
    /// The requested mutation would introduce a cycle into the graph.
    Cycle { id: Id },
    /// The node cannot be removed because it is still referenced as a parent of
    /// another node.
    NodeStillReferenced { id: Id, child_id: Id },
    /// A supplied point has a batch size inconsistent with the rest of the
    /// evaluation.
    BatchSizeMismatch {
        id: Id,
        expected: usize,
        found: usize,
    },
    /// A point required for evaluation (or retraction) was not supplied.
    MissingPoint { id: Id },
    /// The number of tangent/cotangent frames is not constant across the
    /// supplied quantities.
    InconsistentFrameCount { expected: usize, found: usize },
    /// Two points associated through a retraction have mismatched manifold
    /// (quantity) types.
    ManifoldMismatch { id: Id },
    /// A second-order pass was supplied an inconsistent set of order-1 / order-2
    /// cotangents (it must receive both or neither).
    InconsistentCotangents,
    /// A matrix multiplication was requested between dimensionally-incompatible
    /// quantities.
    IncompatibleMatmul { lhs: String, rhs: String },
    /// A triangular-matrix / vector product was requested between
    /// dimensionally-incompatible quantities.
    UnsupportedVectorProduct { tril: String, vector: String },
    /// A rotation was requested on a frame whose leading dimension is not 3.
    InvalidRotationFrame { rows: usize },
    /// Data of the wrong shape/length was supplied to a constructor.
    ShapeMismatch { expected: usize, found: usize },
}

impl fmt::Display for SkoroError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            SkoroError::NodeNotFound { id } => {
                write!(f, "node with id {id:?} does not exist in the graph")
            }
            SkoroError::QuantityTypeMismatch { id } => {
                write!(
                    f,
                    "supplied quantity type does not match the stored type for node {id:?}"
                )
            }
            SkoroError::NotStochastic { id } => {
                write!(f, "node {id:?} is not a stochastic node")
            }
            SkoroError::NotMapped { id } => {
                write!(f, "node {id:?} is not a mapped node")
            }
            SkoroError::NotReal { id } => {
                write!(
                    f,
                    "node {id:?} is not real-valued; only real values can be log-density terms"
                )
            }
            SkoroError::NotContinuous { id } => {
                write!(
                    f,
                    "node {id:?} is discrete; a continuous (manifold-structured) quantity is required"
                )
            }
            SkoroError::DuplicateDensity { stochastic_id } => {
                write!(
                    f,
                    "stochastic quantity {stochastic_id:?} can only have one log-density; \
                     mark the real value as a log-likelihood instead"
                )
            }
            SkoroError::DanglingParent { parent_id } => {
                write!(f, "quantity added with dangling parent {parent_id:?}")
            }
            SkoroError::Cycle { id } => {
                write!(f, "mutation of node {id:?} would introduce a graph cycle")
            }
            SkoroError::NodeStillReferenced { id, child_id } => {
                write!(
                    f,
                    "cannot remove node {id:?}; it is still a parent of node {child_id:?}"
                )
            }
            SkoroError::BatchSizeMismatch {
                id,
                expected,
                found,
            } => {
                write!(
                    f,
                    "batch size for node {id:?} is incorrect; expected {expected}, found {found}"
                )
            }
            SkoroError::MissingPoint { id } => {
                write!(f, "point for node {id:?} is missing")
            }
            SkoroError::InconsistentFrameCount { expected, found } => {
                write!(
                    f,
                    "the number of tensor frames must be constant across quantities; \
                     expected {expected}, found {found}"
                )
            }
            SkoroError::ManifoldMismatch { id } => {
                write!(
                    f,
                    "point for node {id:?} does not share the manifold type of its associated point"
                )
            }
            SkoroError::InconsistentCotangents => {
                write!(
                    f,
                    "a second-order pass must be supplied both order-1 and order-2 cotangents, \
                     or neither"
                )
            }
            SkoroError::IncompatibleMatmul { lhs, rhs } => {
                write!(
                    f,
                    "matrix multiplication between {lhs} and {rhs} is prohibited"
                )
            }
            SkoroError::UnsupportedVectorProduct { tril, vector } => {
                write!(
                    f,
                    "vector product between {tril} and {vector} is unsupported"
                )
            }
            SkoroError::InvalidRotationFrame { rows } => {
                write!(
                    f,
                    "cannot rotate vector frame whose leading dimension is {rows}, not 3"
                )
            }
            SkoroError::ShapeMismatch { expected, found } => {
                write!(
                    f,
                    "data of the wrong shape was supplied; expected {expected} elements, found {found}"
                )
            }
        }
    }
}

impl std::error::Error for SkoroError {}

/// Convenience alias for results that may fail with a [`SkoroError`].
pub type SkoroResult<T> = Result<T, SkoroError>;
