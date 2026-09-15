//! # skoro-normal
//!
//! A manifold-aware Gaussian distribution ([`Normal`](distribution::Normal), parameterized by a
//! `mean: Q<B, T>` and a lower-triangular covariance factor `std_dev: Q<B, Tril>`) plus its sampler
//! ([`NormalSampler`](distribution::NormalSampler)).
//!
//! A Gaussian conditional enters a model through `client.add_quantity(Normal::new(mean, std_dev))`.
//! Its factor binds the whitened residual `r = L⁻¹ (x ⊖ μ)`, `μ` and `L` separately, and the Fisher
//! information root over `[x, μ, L]` built in [`information`]. [`joint::add_joint_factor`] attaches
//! a joint over quantities the model already holds — what a marginalized-out quantity leaves
//! behind — with its own residual and root.
//!
//! [`structure::GaussianBlock`] is the read side: the view of a term that binds its `μ` and `L`
//! apart, which is what a filter needs, since `H = ∂μ/∂x` and `R = L Lᵀ` are different objects. It
//! is derived from the model's bindings per call, never kept as a second list.

pub mod distribution;
pub mod facade;
pub mod information;
pub mod joint;
pub mod structure;
