//! # skoro-normal
//!
//! A manifold-aware Gaussian distribution ([`Normal`](distribution::Normal), parameterized by a
//! `mean: Q<B, T>` and a lower-triangular covariance factor `std_dev: Q<B, Tril>`) plus its sampler
//! ([`NormalSampler`](distribution::NormalSampler)). Also provides the Gaussian model facades
//! ([`facade`]): the [`GaussianModelClientOps`](facade::GaussianModelClientOps) capability trait
//! and the Gaussian-restricted [`GaussianModelClient`](facade::GaussianModelClient) /
//! [`GaussianModelServer`](facade::GaussianModelServer).
//!
//! The [`structure`] module carries the *structural* claims — [`GaussianFactor`](structure::GaussianFactor)
//! on a single distribution, [`GaussianStructure`](structure::GaussianStructure) on a whole
//! model — which are what an inference algorithm bounds on. The facade traits are
//! capabilities and claim nothing about what a model contains.

pub mod distribution;
pub mod facade;
pub mod structure;
