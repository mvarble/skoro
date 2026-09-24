//! # skoro-dense
//!
//! Host-side linear algebra in `f64` for the deterministic estimators: dense factorizations over
//! one matrix at a time, and block-sparse elimination of a linear-Gaussian factor graph.
//!
//! ## Dense
//!
//! [`dense::Dense`] is one matrix, stored as a `faer::Mat<f64>`. [`factor`] holds the
//! factorizations and solves over it ([`factor::cholesky`], [`factor::spd_solve`],
//! [`factor::lstsq`], the triangular solves). A caller carrying a batch of independent problems
//! reaches them through [`dense::Dense::split_tensor`] / [`dense::Dense::split_vector`] and
//! [`dense::Dense::stack_vector`], which move the whole batch across the host boundary in one
//! transfer and then work row by row — the batch axis is handled *around* the factorizations, never
//! inside them.
//!
//! ## Sparse, by blocks
//!
//! A model's linearization is block-sparse: each term touches a few keys. The elimination here is
//! GTSAM's, at the level of keys rather than scalars:
//!
//! 1. [`factor_graph::GaussianFactorGraph`] — factors `‖Σ_k A_k δ_k − b‖²` over declared keys.
//! 2. [`ordering::Ordering`] — COLAMD on the block pattern (one entry per factor and key), or with
//!    chosen keys constrained last.
//! 3. [`elimination::JunctionTree`] — the symbolic elimination: which keys are eliminated together,
//!    and what separator each clique leaves. Computed once per structure and reused.
//! 4. [`bayes::eliminate`] — dense unpivoted QR on each front, giving a [`bayes::BayesTree`] of
//!    conditionals `R x_f + S x_s = d`, and the factors left on any keys that were kept.
//! 5. [`marginal::marginal`] — partial elimination onto kept keys: the Schur complement in
//!    square-root form, as a lower-triangular whitening ready for a `Tril`.
//!
//! None of it asks what the rows mean. Whitened residual rows with `b = −r` give Gauss–Newton
//! ([`bayes::BayesTree::back_substitute`]); Fisher-root rows with `b = 0` and a gradient give
//! Fisher scoring ([`bayes::BayesTree::solve_normal`]); either gives a Laplace marginal.
//!
//! Keys are any [`values::Key`]. This crate depends on `skoro-backend` and `skoro-core` and
//! **nothing else in the workspace** — in particular not `skoro-graph` or `skoro-model`, so nothing
//! that merely uses `Normal` or `Tril` pays for `faer`. The bridge from a model, keyed by graph
//! `Id`, is `skoro-linearize`.
//!
//! ## What this is not
//!
//! A host `f64` solve is **not** `f64` end to end. When `B::FloatElem = f32` the Jacobians
//! arrive already rounded to `f32` and `δ` is rounded back on the way out. What the host
//! path buys is `f64` *inside* the solve, which is where the conditioning lives. A filter that
//! wants an `f64` state should run `Flex` under the `f64` float policy, where this round trip is a
//! `memcpy`.

pub mod bayes;
pub mod dense;
pub mod elimination;
pub mod factor;
pub mod factor_graph;
pub mod marginal;
pub mod ordering;
pub mod values;
