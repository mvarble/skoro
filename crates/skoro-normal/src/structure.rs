//! The Gaussian view of a recorded least-squares factor.
//!
//! A [`Normal`](crate::distribution::Normal) factor's negative log-density is `½‖r‖²` plus a
//! normalizer, with `r(x) = L⁻¹ (x ⊖ μ)`. What distinguishes it from a merely
//! least-squares-shaped factor is that the mean and the whitening are reachable
//! **separately** — which is what a filter needs, since `H = ∂μ/∂x` and `R = L Lᵀ` are
//! different objects.
//!
//! That separation is recorded when the factor is built, in
//! [`FactorRecord::location`] and [`FactorRecord::whitening`], and [`GaussianBlock`] is the
//! view of a record that has both. The *nonlinearity* of the mean needs no expression here:
//! `location` names a graph node, which may be the output of arbitrarily many operations, so
//! the refinement is two ids rather than a new algebra.

use skoro_core::id::Id;
use skoro_model::FactorRecord;

/// One factor of a hierarchical Gaussian model, as its builder recorded it.
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
    /// The Gaussian view of `record`, or `None` if it carries no `μ`/`L` refinement.
    ///
    /// Deriving rather than keeping a parallel list is the point: membership is the
    /// registry's decision, so the two cannot come to disagree about which factors exist. A
    /// record whose factor was retired simply stops being offered.
    pub fn of(record: &FactorRecord) -> Option<Self> {
        Some(Self {
            density: record.density,
            residual: record.residual,
            mean: record.location?,
            whitening: record.whitening?,
        })
    }

    /// The Gaussian blocks among `records`, in the order they were recorded — the row order
    /// of `J`, restricted to the factors that carry a refinement.
    pub fn all(records: &[FactorRecord]) -> Vec<Self> {
        records.iter().filter_map(Self::of).collect()
    }
}
