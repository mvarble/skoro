//! The normal distribution of skoro: the law of a point moved by normal noise through its chart,
//! on every space whose chart is global and whose volume is known.
//!
//! On the core's Euclidean spaces it is the familiar normal, on the positive orthant the
//! log-normal, and on a curved type with a declared volume the normal of its tangent chart pushed
//! onto it. Its term is the Gaussian log-density of `x ⊖ μ`, whitened by the scale in one of four
//! forms, less the chart's log-volume; its sampler draws `μ ⊕ L Φ⁻¹(u)` exactly; and it binds the
//! roles of [`roles`] for the readers that look for them.
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
