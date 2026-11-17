//! The normal distribution of skoro: the law of a point moved by normal noise through its chart,
//! on every space whose chart is global and whose volume is known.
//!
//! On the core's Euclidean spaces it is the familiar normal, on the positive orthant the
//! log-normal, and on a curved type with a declared volume the normal of its tangent chart pushed
//! onto it. Its term is the Gaussian log-density of `x ⊖ μ`, whitened by the scale in one of four
//! forms, less the chart's log-volume; its sampler draws `μ ⊕ L Φ⁻¹(u)` exactly; and it binds the
//! roles of [`roles`] for the readers that look for them.
//!
//! On a curved type the scale is in the coordinates of the chart, which are radians to first
//! order on the rotations, and a large one does not say that little is known. The mean is a mode
//! of the density against the type's measure only while every eigenvalue of the covariance is
//! below one. Beyond that the mass goes towards the chart's cut, which on `Rotation` is the
//! half-turns from the mean: at a standard deviation of 100 a draw is within 0.1 rad of a
//! half-turn with probability 0.98. The law is exact at every scale, in its term and in its
//! draws, and no scale is refused or altered; a quantity of which nothing is known is given
//! `skoro_uniform::Uniform`, the uniform distribution, which no normal is near.
//!
//! Over the stacked tangent of a retirement's blanket, [`Stacked`] writes a Gaussian replacement of
//! full rank, and [`Improper`] the improper Gaussian factor of any number of rows, a linear part
//! and a constant, which keeps what retired terms say about quantities they do not determine.

mod composites;
mod improper;
mod normal;
mod refuse;
pub mod roles;
mod sample;
mod scale;
mod stacked;

pub use improper::Improper;
pub use normal::{Normal, Parts};
pub use sample::{NormalSample, quantile};
pub use scale::{Form, Scale, triangular};
pub use stacked::Stacked;
