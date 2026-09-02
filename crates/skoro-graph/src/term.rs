//! A term is a [`Node::LogLikelihood`](crate::node::Node::LogLikelihood) that is currently in
//! [`Graph::log_likelihoods`](crate::graph::Graph::log_likelihoods) — one summand of the
//! model's objective.

use skoro_core::id::Id;

/// Whether a term scores a live stochastic quantity (its removal changes the *proposal*) or
/// not (an observed density, or a free-standing factor — its removal changes only the
/// *weight*).
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
    /// The quantity this term is the log-density of, if any — `None` for a free-standing
    /// factor.
    pub scored: Option<Id>,
    pub kind: TermKind,
}
