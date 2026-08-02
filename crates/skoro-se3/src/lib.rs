//! # skoro-se3
//!
//! The SE(3) rigid-body transform group as a Skoro manifold ([`quantity::SE3`],
//! a rotation + translation). [`ops`] provides its differentiable retraction and
//! inverse-retraction, composing SO(3) rotation operations (from `skoro-so3`)
//! with translation.

pub mod ops;
pub mod quantity;
