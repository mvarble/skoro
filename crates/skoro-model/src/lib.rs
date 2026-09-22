//! # skoro-model
//!
//! The high-level, user-facing model interface. A [`model::ModelServer`] wraps a
//! [`skoro_graph::graph::Graph`]; a [`model::ModelClient`] is a cheap, cloneable
//! handle to it. Quantities are returned as typed proxies [`q::Q`]. The
//! [`ModelReaderOps`] trait factors the distribution-agnostic client surface;
//! [`distribution::Distribution`] registers conditional distributions
//! (continuous or discrete, via the shared kind parameter); [`map`]'s
//! `register_mapN` and the [`ops`] / [`discrete`] modules expose deterministic
//! operations (with operator sugar) on quantities of either kind. A
//! distribution's [`Factor`](distribution::Factor) binds its term to the nodes
//! playing each of the [`roles`], and [`least_squares`] and [`information`]
//! read those bindings back at solve time.

mod cleanup;
pub mod client_ops;
pub mod coverage;
pub mod discrete;
pub mod distribution;
pub mod hash_map_builders;
pub mod information;
pub mod jacobian;
pub mod least_squares;
pub mod map;
pub mod model;
pub mod ops;
pub mod point_map_access;
pub mod q;
pub mod roles;

pub use client_ops::{ModelBuilderOps, ModelReaderOps};

/// The flattened tangent-space layout the [`jacobian`] functions index by, re-exported
/// from `skoro-graph` so a caller assembling dense matrices needs only this crate.
pub use skoro_graph::tangent_layout::{TangentBlock, TangentLayout};

/// The recoverable error type for interfacing with a model, re-exported from
/// `skoro-core`.
pub use skoro_core::error::{SkoroError, SkoroResult};

#[cfg(feature = "testing")]
pub mod test_utils;
