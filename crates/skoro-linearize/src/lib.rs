//! Linearization of skoro's queries: a density query of a model, at a batch of points, as one
//! factored least-squares problem per row, by its terms' whitened residuals or, for free scales,
//! by their information roots; and a retirement's replacement, the linearized integral of its
//! retired terms, as an improper Gaussian.
//!
//! A [`Linearizer`] is built once for a snapshot, a density query, a target and an
//! [`Assembly`]. It refuses by structure what cannot be linearized, reads the factor structure off
//! the terms' variables, and compiles the programs and the seeds its passes need. At each batch of
//! points it gives a [`Linearized`], one [`skoro_elimination::Problem`] per row over one
//! structure, which a [`skoro_elimination::Schedule`] eliminates row by row into a [`Solved`]: the
//! step, the marginals, the evidence and draws. A [`Replacing`] computes the replacement of a
//! retirement the same way and builds it as the normal's `Improper` factor.

mod assemble;
mod error;
mod linearizer;
mod replacing;
mod solved;

pub use assemble::Linearized;
pub use error::Error;
pub use linearizer::{Assembly, Linearizer};
pub use replacing::{LinearizedReplacement, Replacing};
pub use solved::{QuantityMarginal, Solved};
