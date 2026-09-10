//! # skoro-normal
//!
//! A manifold-aware Gaussian distribution ([`Normal`](distribution::Normal), parameterized by a
//! `mean: Q<B, T>` and a lower-triangular covariance factor `std_dev: Q<B, Tril>`) plus its sampler
//! ([`NormalSampler`](distribution::NormalSampler)).
//!
//! [`facade::add_gaussian_quantity`] is how a Gaussian conditional enters a model: it adds the
//! quantity, builds the whitened residual `r = L⁻¹ (x ⊖ μ)`, and records the factor with `μ` and
//! `L` named separately. [`joint::add_joint_factor`] does the same for a joint over quantities the
//! model already holds — what a marginalized-out quantity leaves behind.
//!
//! [`structure::GaussianBlock`] is the read side: the view of a recorded factor that carries the
//! `μ`/`L` refinement, which is what a filter needs, since `H = ∂μ/∂x` and `R = L Lᵀ` are different
//! objects. It is derived from the model's registry per call, never kept as a second list.

pub mod distribution;
pub mod facade;
pub mod information;
pub mod joint;
pub mod structure;
