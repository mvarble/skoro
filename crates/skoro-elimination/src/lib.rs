//! Elimination in square-root form: a sum of squared affine residuals and a linear term,
//!
//! `c + Σ_f ½|A_f τ_{K_f} - b_f|² + ηᵀτ`,
//!
//! over blocks of variables, minimized, integrated or reduced to some of its blocks one block at a
//! time, by orthogonal factorizations of its rows and never by its normal equations. It is the
//! note on elimination in square-root form as code, and nothing in it depends on where the rows
//! came from.
//!
//! A [`Structure`] says which blocks each factor involves; a [`Schedule`], computed from the
//! structure alone, orders the blocks and groups them into the cliques of a junction tree; a
//! [`Problem`] holds one set of numbers over a structure, and [`Problem::eliminate`] reduces it
//! along a schedule to an [`Elimination`], whose conditionals and constants answer the minimizer,
//! the integral, draws, marginals and the factor left on the blocks it kept. The factorizations
//! are faer's, run on the host in double precision; no faer type is public.

mod answers;
mod eliminate;
mod error;
mod matrix;
mod problem;
mod structure;

pub use answers::Marginal;
pub use eliminate::{Conditional, Elimination};
pub use error::Error;
pub use matrix::Matrix;
pub use problem::{Damping, Problem, Rows};
pub use structure::{Schedule, Structure};
