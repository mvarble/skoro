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
    ModelClientOps,
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
    pub fn of<B: Backend, C: ModelClientOps<B>>(client: &C, factor: &FactorInfo) -> Option<Self> {
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

    /// The Gaussian blocks among the model's terms, in term order — the row order of `J`,
    /// restricted to the terms that bind a location and a whitening.
    ///
    /// Derived from the bindings on every call rather than kept as a parallel list, so a retired
    /// term's block simply stops appearing.
    pub fn all<B: Backend, C: ModelClientOps<B>>(client: &C) -> Vec<Self> {
        client
            .factors()
            .iter()
            .filter_map(|factor| Self::of(client, factor))
            .collect()
    }
}
