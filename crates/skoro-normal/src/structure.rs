//! The Gaussian view of a term's bindings.
//!
//! A [`Normal`](crate::distribution::Normal) factor's negative log-density is `½‖r‖²` plus a
//! normalizer, with `r(x) = L⁻¹ (x ⊖ μ)`. What distinguishes it from a merely
//! least-squares-shaped factor is that the mean and the whitening are reachable
//! **separately** — which is what a filter needs, since `H = ∂μ/∂x` and `R = L Lᵀ` are
//! different objects.
//!
//! That separation is bound when the factor is built, under
//! [`LOCATION`] and [`WHITENING`], and [`GaussianBlock`] is the view of a term that binds both
//! alongside its [`RESIDUAL`]. The *nonlinearity* of the mean needs no expression here: the
//! location is a graph node, which may be the output of arbitrarily many operations, so the
//! refinement is two ids rather than a new algebra.

use skoro_backend::Backend;
use skoro_core::id::Id;
use skoro_model::{
    ModelReaderOps,
    client_ops::FactorInfo,
    roles::{LOCATION, RESIDUAL, WHITENING},
};

/// One factor of a hierarchical Gaussian model, read off its term's bindings.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GaussianBlock {
    /// The log-density node.
    pub density: Id,
    /// The whitened-residual node — a row block of `J`.
    pub residual: Id,
    /// `μ`. `H = ∂μ/∂x` is a Jacobian row block over this.
    pub mean: Id,
    /// `L`, with `R = L Lᵀ`.
    pub whitening: Id,
}

impl GaussianBlock {
    /// The Gaussian view of `factor`, or `None` unless its term binds exactly one [`RESIDUAL`],
    /// one [`LOCATION`] and one [`WHITENING`].
    ///
    /// Says nothing about whether the model's log-density scores the factor — pass a
    /// proposal-only [`FactorInfo`] and you get its Gaussian structure just the same. That is
    /// deliberate: [`all`](Self::all) is the one that takes a side.
    pub fn of<B: Backend, C: ModelReaderOps<B>>(client: &C, factor: &FactorInfo) -> Option<Self> {
        let single = |ids: Vec<Id>| match ids.as_slice() {
            [id] => Some(*id),
            _ => None,
        };
        Some(Self {
            density: factor.density,
            residual: single(client.bound(factor.density, RESIDUAL))?,
            mean: single(client.bound(factor.density, LOCATION))?,
            whitening: single(client.bound(factor.density, WHITENING))?,
        })
    }

    /// The Gaussian blocks among the model's *log-density* terms, in term order — the row order of
    /// `J`, restricted to the terms that bind a location and a whitening.
    ///
    /// Derived from the bindings on every call rather than kept as a parallel list, so a retired
    /// term's block simply stops appearing.
    ///
    /// Proposal-only terms are skipped, matching
    /// [`residual_layout`](skoro_model::least_squares::residual_layout) and
    /// [`information_blocks`](skoro_model::information::information_blocks): a term the
    /// posterior does not score is no part of an objective assembled from these blocks. To see
    /// every Gaussian term, proposals included, walk
    /// [`factors`](ModelReaderOps::factors) yourself and call [`of`](Self::of).
    ///
    /// A scored term that is not Gaussian-shaped is also skipped, and the returned list does not
    /// say which. That is the same silence as the other two readers': a caller reading these
    /// blocks positionally beside a row layout should confirm the model holds no term it cannot
    /// see, with [`terms_missing`](skoro_model::coverage::terms_missing) over
    /// [`WHITENING`] or
    /// [`require_role`](skoro_model::coverage::require_role).
    pub fn all<B: Backend, C: ModelReaderOps<B>>(client: &C) -> Vec<Self> {
        client
            .factors()
            .iter()
            .filter(|factor| factor.in_density)
            .filter_map(|factor| Self::of(client, factor))
            .collect()
    }
}
