//! # skoro-so3
//!
//! The SO(3) rotation group as a Skoro manifold ([`quantity::SO3`], stored as a
//! flattened 3×3 matrix). [`ops`] provides differentiable operations — `exp`
//! (Rodrigues' formula), `log`, retraction/inverse-retraction, compose, rotate /
//! unrotate, and invert — each implemented as a `DiffKernel` for both backends.

pub mod ops;
pub mod quantity;
