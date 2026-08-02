//! # skoro-riemann
//!
//! Differential-geometry framework providing the [`Manifold`] / [`DynManifold`]
//! traits (a differentiable retraction and inverse-retraction per manifold) and
//! the flat manifold representations used throughout Skoro: the real line
//! ([`real::Real`]), Euclidean spaces ([`euclidean`]), and Euclidean matrix
//! spaces ([`matrix`]). Curved manifolds (SO(3), SE(3)) live in their own crates.

mod base;
pub mod euclidean;
pub mod matrix;
pub mod real;

pub use base::*;
