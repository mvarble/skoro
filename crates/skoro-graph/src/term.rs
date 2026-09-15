//! A term is a [`Node::LogLikelihood`](crate::node::Node::LogLikelihood) that is currently in
//! [`Graph::log_likelihoods`](crate::graph::Graph::log_likelihoods).
//!
//! A graph carries two densities over the same nodes: the posterior `p(x|z)` it scores, and the
//! generative model `q(x|z)` it samples from. Each term declares its membership in each, and the
//! log-weight is the difference:
//!
//! ```text
//! p = Σ { term : in_density }
//! q = Σ { term : in_proposal }
//! w = p − q
//! ```
//!
//! All four combinations are reachable:
//!
//! | `in_density` | `in_proposal` | |
//! |---|---|---|
//! | true | true | the density of a live stochastic quantity the posterior scores |
//! | true | false | an observed quantity's density, or a free-standing factor |
//! | false | true | a proposal-only term: it generates a quantity the posterior does not score here |
//! | false | false | a proposal-only quantity that has since been observed — a fixed value is not sampled, so it is a summand of neither |

use skoro_core::id::Id;

/// One term of the model, as read off [`Graph::terms`](crate::graph::Graph::terms).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct Term {
    /// The log-likelihood node.
    pub density: Id,
    /// The quantity this term is the log-density of, if any — `None` for a free-standing
    /// factor.
    pub scored: Option<Id>,
    /// A summand of the model's log-density `p(x|z)`. Read off
    /// [`LogLikelihoodNode::in_log_density`](crate::node::LogLikelihoodNode::in_log_density).
    pub in_density: bool,
    /// A summand of the generative model `q(x|z)`: this term is the density of a quantity that
    /// is still being generated.
    ///
    /// Derived from the node kind of [`scored`](Self::scored) rather than stored, so
    /// [`observe`](crate::graph::Graph::observe) moves a term out of the proposal — and
    /// `unobserve` moves it back — with no index to keep in step.
    pub in_proposal: bool,
}
