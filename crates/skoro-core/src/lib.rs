//! # skoro-core
//!
//! Foundational types and traits shared across the workspace:
//!
//! - unique-id management ([`id`]),
//!
//! - the recoverable error type ([`error`]),
//!
//! - the label on a binding's reference edge ([`role`]),
//!
//! - the continuous/discrete kind tags ([`kind`]),
//!
//! - flat-buffer packing of data ([`pack`]),
//!
//! - stochastic sampling ([`sampler`]),
//!
//! - kind-erasure on the backend tensors ([`any_tensor`]).
//!
//! - the indexed sequence abstraction ([`sequence`]).
//!
//! - discrete quantity types ([`discrete`]).

pub mod any_tensor;
pub mod discrete;
pub mod error;
pub mod id;
pub mod kind;
pub mod pack;
pub mod role;
pub mod sampler;
pub mod sequence;
