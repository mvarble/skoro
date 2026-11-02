//! The normal distribution of skoro: the law of a point moved by normal noise through its chart,
//! on every space whose chart is global and whose volume is known.
//!
//! On the core's Euclidean spaces it is the familiar normal, on the positive orthant the
//! log-normal, and on a curved type with a declared volume the normal of its tangent chart pushed
//! onto it. Its term is the Gaussian log-density of `x ⊖ μ`, whitened by the scale in one of four
//! forms, less the chart's log-volume; its sampler draws `μ ⊕ L Φ⁻¹(u)` exactly; and it binds the
//! roles of [`roles`] for the readers that look for them.

#[allow(dead_code)] // the term builds from them
mod composites;
#[allow(dead_code)] // `score` refuses through it
mod refuse;
pub mod roles;
mod sample;
mod scale;

pub use sample::quantile;
pub use scale::{Form, Scale, triangular};
