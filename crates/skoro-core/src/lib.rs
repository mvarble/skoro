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
//! - the [`ConstantValue`](constant_value::ConstantValue) wrapper shared by continuous and
//!   discrete constant quantities ([`constant_value`]),
//!
//! - the [`OperationInput`](operation_input::OperationInput) operand shared by continuous and
//!   discrete operation tracks ([`operation_input`]),
//!
//! - the indexed sequence abstraction ([`sequence`]).
//!
//! - discrete quantity types ([`discrete`]).

pub mod constant_value;
pub mod discrete;
pub mod error;
pub mod id;
pub mod kind;
pub mod operation_input;
pub mod pack;
pub mod sampler;
pub mod sequence;
