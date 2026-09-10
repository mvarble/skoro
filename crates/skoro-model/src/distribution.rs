use skoro_backend::Backend;
use skoro_core::error::SkoroResult;
use skoro_core::id::Id;
use skoro_core::kind::Kind;
use skoro_graph::binding::Role;
use skoro_graph::quantity::QuantityType;
use skoro_riemann::real::Real;

pub use skoro_core::sampler::Sampler;

use crate::q::{Handle, Q};

/// A log-density term together with the bindings its builder declares for it.
///
/// The handles are load-bearing: between the moment a factor is built and the moment its
/// bindings are recorded, they are what keeps a bound node (a residual, a Fisher root) from being
/// collected.
#[derive(Debug)]
pub struct Factor<B: Backend> {
    pub log_density: Q<B, Real>,
    pub bindings: Vec<(Role, Handle<B>)>,
}

impl<B: Backend> Factor<B> {
    /// A factor with no bindings.
    pub fn new(log_density: Q<B, Real>) -> Self {
        Self {
            log_density,
            bindings: Vec::new(),
        }
    }

    /// Declare that `q` plays `role` for this factor's term.
    pub fn bind<T>(mut self, role: Role, q: &Q<B, T>) -> Self {
        self.bindings.push((role, Handle::from(q)));
        self
    }
}

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
