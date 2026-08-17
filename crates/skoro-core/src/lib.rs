//! # skoro-core
//!
//! Foundational types and traits shared across the workspace:
//!
//! - unique-id management ([`id`]),
//!
//! - the recoverable error type ([`error`]),
//!
//! - the continuous/discrete kind tags ([`kind`]),
//!
//! - flat-buffer packing of data ([`pack`]),
//!
//! - stochastic sampling ([`sampler`]),
//!
//! - kind-erasure on the backend tensors ([`kind_erased_tensor`]).
//!
//! - the indexed sequence abstraction ([`sequence`]).
//!
//! - discrete quantity types ([`discrete`]).

pub mod discrete;
pub mod error;
pub mod id;
pub mod kind;
pub mod kind_erased_tensor;
pub mod pack;
pub mod sampler;
pub mod sequence;
