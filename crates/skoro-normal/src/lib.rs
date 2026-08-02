//! # skoro-normal
//!
//! A manifold-aware Gaussian distribution ([`Normal`](distribution::Normal), parameterized by a
//! `mean: Q<B, T>` and a lower-triangular covariance factor `std_dev: Q<B, Tril>`) plus its sampler
//! ([`NormalSampler`](distribution::NormalSampler)). Also provides the Gaussian model facades
//! ([`facade`]): the [`GaussianModelClientOps`](facade::GaussianModelClientOps) capability trait
//! and the Gaussian-restricted [`GaussianModelClient`](facade::GaussianModelClient) /
//! [`GaussianModelServer`](facade::GaussianModelServer).

pub mod distribution;
pub mod facade;
