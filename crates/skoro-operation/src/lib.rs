//! This crate provides the utilties for performing autodifferentiation in Skoro. The architecture
//! is simplified, because the Id management and garbage collection will be managed by the
//! probabilistic model.

mod base;

pub use base::*;

pub mod diff;
pub mod discrete;
pub mod ops;
pub mod step;
pub mod tape;
