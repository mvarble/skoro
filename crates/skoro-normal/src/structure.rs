//! The Gaussian view of a factor's bindings.
//!
//! A [`Normal`](crate::distribution::Normal) factor's negative log-density is `½‖r‖²` plus a
//! normalizer, with `r(x) = L⁻¹ (x ⊖ μ)`. What distinguishes it from a merely
//! least-squares-shaped factor is that the mean and the whitening are reachable
//! **separately** — which is what a filter needs, since `H = ∂μ/∂x` and `R = L Lᵀ` are
//! different objects. [`GaussianBlock`] is the view of a term whose bindings carry all three
//! of `RESIDUAL`, `LOCATION` and `WHITENING`.

use skoro_backend::Backend;
use skoro_core::id::Id;

use skoro_model::{
    ModelClientOps,
    client_ops::FactorInfo,
    roles::{LOCATION, RESIDUAL, WHITENING},
};

/// One factor of a hierarchical Gaussian model, read off its bindings.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GaussianBlock {
    pub density: Id,
    pub residual: Id,
    pub mean: Id,
    pub whitening: Id,
}

impl GaussianBlock {
    /// The Gaussian view of `factor`, or `None` unless it carries exactly one `RESIDUAL`, one
    /// `LOCATION` and one `WHITENING` binding.
    ///
    /// Says nothing about whether the model's log-density scores the factor — pass a
    /// proposal-only [`FactorInfo`] and you get its Gaussian structure just the same. That is
    /// deliberate: [`all`](Self::all) is the one that takes a side.
    pub fn of<B: Backend, C: ModelClientOps<B>>(client: &C, factor: &FactorInfo) -> Option<Self> {
        let residual = single(client.bound(factor.density, RESIDUAL))?;
        let mean = single(client.bound(factor.density, LOCATION))?;
        let whitening = single(client.bound(factor.density, WHITENING))?;
        Some(Self {
            density: factor.density,
            residual,
            mean,
            whitening,
        })
    }

    /// The Gaussian blocks among the model's *log-density* terms, in term order.
    ///
    /// Proposal-only terms are skipped, matching
    /// [`residual_layout`](skoro_model::least_squares::residual_layout) and
    /// [`information_blocks`](skoro_model::information::information_blocks): a term the
    /// posterior does not score is no part of an objective assembled from these blocks. To see
    /// every Gaussian term, proposals included, walk
    /// [`factors`](ModelClientOps::factors) yourself and call [`of`](Self::of).
    pub fn all<B: Backend, C: ModelClientOps<B>>(client: &C) -> Vec<Self> {
        client
            .factors()
            .iter()
            .filter(|factor| factor.in_density)
            .filter_map(|factor| Self::of(client, factor))
            .collect()
    }
}

fn single(ids: Vec<Id>) -> Option<Id> {
    match ids.as_slice() {
        [id] => Some(*id),
        _ => None,
    }
}
