//! # skoro-linalg
//!
//! Backend-agnostic, differentiable linear-algebra primitives exposed as
//! operations on quantities: matrix multiply and transpose ([`matrix`]), vector
//! norms and cross products ([`vector`]), and the lower-triangular matrix
//! manifold ([`tril`]) used for covariance factors. Each operation is a
//! `DiffKernel` with `forward`/`vjp`/`hvp` implemented for both backends.

pub mod matrix;
pub mod tril;
pub mod vector;
