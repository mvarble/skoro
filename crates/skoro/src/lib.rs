//! skoro: probabilistic modeling on manifolds, with derivatives of first and second order.
//!
//! This crate re-exports the core and the tensor layer.

pub use skoro_core::*;
pub use skoro_tensor as tensor;
pub use skoro_tensor::{Device, Precision, Target};
