//! # skoro-model
//!
//! The high-level, user-facing model interface. A [`model::ModelServer`] wraps a
//! [`skoro_graph::graph::Graph`]; a [`model::ModelClient`] is a cheap, cloneable
//! handle to it. Quantities are returned as typed proxies [`q::Q`]. The
//! [`ModelClientOps`] trait factors the distribution-agnostic client surface;
//! [`distribution::Distribution`] registers conditional distributions
//! (continuous or discrete, via the shared kind parameter); [`map`]'s
//! `register_mapN` and the [`ops`] / [`discrete`] modules expose deterministic
//! operations (with operator sugar) on quantities of either kind. [`residual`]
//! is where a builder writes down what it knows about a least-squares factor,
//! for an estimator to read back at solve time.

mod cleanup;
pub mod client_ops;
pub mod discrete;
pub mod distribution;
pub mod hash_map_builders;
pub mod jacobian;
pub mod map;
pub mod model;
pub mod ops;
pub mod point_map_access;
pub mod q;
pub mod residual;

pub use client_ops::{ModelBuilderOps, ModelClientOps};
pub use residual::{FactorRecord, ResidualRegistry};

/// The flattened tangent-space layout the [`jacobian`] functions index by, re-exported
/// from `skoro-graph` so a caller assembling dense matrices needs only this crate.
pub use skoro_graph::tangent_layout::{TangentBlock, TangentLayout};

/// The recoverable error type for interfacing with a model, re-exported from
/// `skoro-core`.
pub use skoro_core::error::{SkoroError, SkoroResult};

#[cfg(feature = "testing")]
pub mod test_utils;
