use skoro_backend::Backend;
use skoro_core::error::SkoroResult;
use skoro_core::id::Id;
use skoro_core::kind::Kind;
use skoro_graph::quantity::QuantityType;
use skoro_riemann::real::Real;

pub use skoro_core::sampler::Sampler;

use crate::q::Q;

/// A conditional distribution over a quantity of kind `K`.
///
/// The kind parameter unifies continuous distributions (whose `log_density` is
/// a log-pdf) and discrete distributions (whose `log_density` is a log-pmf):
/// both produce a real-valued log-probability term, and both are registered
/// through the single [`ModelClient::add_quantity`](crate::model::ModelClient::add_quantity)
/// entry point.
pub trait Distribution<B: Backend, K: Kind<B>> {
    /// The quantity type this distribution is over.
    type Qty: QuantityType<B, K>;
    /// The sampler drawing values of [`Qty`](Distribution::Qty).
    type Sampler: Sampler<B, K>;

    fn quantity_type(&self) -> Self::Qty;
    fn sampler(&self) -> Self::Sampler;
    fn parent_ids(&self) -> &[Id];
    /// Build the log-density node scoring `x` under this distribution.
    ///
    /// Fallible because a distribution's parameters and its quantity can disagree
    /// dimensionally — a covariance factor sized for a different manifold, say — which
    /// is ordinary user error rather than an internal fault.
    fn log_density(&self, x: &Q<B, Self::Qty>) -> SkoroResult<Q<B, Real>>;
}
