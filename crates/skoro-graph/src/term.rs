//! A term is a [`Node::LogLikelihood`](crate::node::Node::LogLikelihood) that is currently among
//! the [`log_likelihoods`](crate::graph::Graph::log_likelihoods): one summand of the model's
//! log-density.

use skoro_core::id::Id;

/// Whether a term scores a live stochastic quantity (a *proposal* term) or not (an observed
/// quantity's density, or a free-standing factor: a *weight* term).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum TermKind {
    Proposal,
    Weight,
}

/// One summand of the model's log-density, as read off [`Graph::terms`](crate::graph::Graph::terms).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct Term {
    /// The log-likelihood node.
    pub density: Id,
    /// The quantity this term is the log-density of; `None` for a free-standing factor.
    pub scored: Option<Id>,
    pub kind: TermKind,
}
