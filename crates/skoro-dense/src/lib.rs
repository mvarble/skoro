//! # skoro-dense
//!
//! Host-side dense linear algebra in `f64`, one matrix at a time, for the deterministic
//! estimators. Factorizations ([`factor::cholesky`], [`factor::spd_solve`]), least squares
//! ([`factor::lstsq`]) and the whitening a marginal needs ([`factor::whitening_from_precision`])
//! over [`dense::Dense`], with conversions to and from backend tensors.
//!
//! A [`dense::Dense`] is always exactly one matrix. A caller carrying a batch of independent
//! problems reaches the same factorizations through [`dense::Dense::split_tensor`] /
//! [`dense::Dense::split_vector`] and [`dense::Dense::stack_vector`], which move the whole
//! batch across the host boundary in one transfer and then work row by row — so the batch
//! axis is handled *around* the factorizations rather than inside them, and none of the six
//! functions in [`factor`] has a batch dimension.
//!
//! This crate depends on `skoro-backend` and `skoro-core` and **nothing else in the
//! workspace** — in particular not `skoro-graph` or `skoro-model`, so nothing that merely
//! uses `Normal` or `Tril` pays for `faer`.
//!
//! The boundary is two tensors wide: `J` on the way in and `δ` on the way out. Everything
//! between them is host `f64` and has no representation in the graph
//! (`docs/skoro.md` §4.2).
//!
//! ## What this is not
//!
//! A host `f64` solve is **not** `f64` end to end. When `B::FloatElem = f32` the Jacobians
//! arrive already rounded to `f32` and `δ` is rounded back on the way out. What the host
//! path buys is `f64` *inside* the solve, which is where the `κ²` conditioning of the
//! normal equations lives. A filter that wants an `f64` state should run
//! `Flex` under the `f64` float policy, where this round trip is a `memcpy`.

pub mod dense;
pub mod factor;
