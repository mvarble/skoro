//! # skoro-normal
//!
//! A manifold-aware Gaussian distribution ([`Normal`](distribution::Normal), parameterized by a
//! `mean: Q<B, T>` and a lower-triangular covariance factor `std_dev: Q<B, Tril>`) plus its sampler
//! ([`NormalSampler`](distribution::NormalSampler)).
//!
//! `add_quantity(Normal::new(μ, L))` is how a Gaussian conditional enters a model: it adds the
//! quantity, and [`Normal`](distribution::Normal)'s `Distribution::factor` builds the whitened
//! residual `r = L⁻¹ (x ⊖ μ)` and binds it alongside `μ` and `L` under their own roles.
//! [`joint::add_joint_factor`] does the same for a joint over quantities the model already holds —
//! what a marginalized-out quantity leaves behind.
//!
//! [`structure::GaussianBlock`] is the read side: the view of a term whose bindings carry the
//! `μ`/`L` refinement, which is what a filter needs, since `H = ∂μ/∂x` and `R = L Lᵀ` are different
//! objects. It is derived from the model's bindings per call, never kept as a second list.

pub mod distribution;
pub mod facade;
pub mod information;
pub mod joint;
pub mod structure;
