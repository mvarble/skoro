use std::fmt::Debug;
use std::sync::Arc;

use skoro_backend::Backend;
use skoro_core::{any_tensor::QuantityValue, id::Id, kind::QuantityKind};

use crate::{operation::DynGraphOperation, quantity::DynQuantityType, sampler::DynGraphSampler};

#[derive(Clone)]
pub enum Node<B: Backend> {
    /// A fixed, immutable value.
    Constant(ConstantNode<B>),
    /// A free variable with an initial value, intended to be optimized/inferred.
    Parameter(ParameterNode<B>),
    /// A stochastic quantity with a conditional sampler.
    Stochastic(StochasticNode<B>),
    /// A stochastic quantity bound to a concrete, fixed value.
    Observed(ObservedNode<B>),
    /// A quantity which is entirely determined by its parents.
    Mapped(MappedNode<B>),
    /// A dedicated transformation representing the real-valued log-likelihood factor within the
    /// model. Semantically identical to its sibling `Mapped`, but it is guaranteed to be a `Real`
    /// quantity type and has information on whether it is a density for a stochastic quantity.
    LogLikelihood(LogLikelihoodNode<B>),
}

impl<B: Backend> Node<B> {
    /// Obtain the parents of a node in our DAG.
    ///
    /// These are quantities whose realizations are required to realize this quantity.
    pub fn parent_ids(&self) -> &[Id] {
        match self {
            Node::Mapped(mapped) => &mapped.parent_ids,
            Node::LogLikelihood(dens) => &dens.parent_ids,
            Node::Stochastic(stoch) => &stoch.parent_ids,
            Node::Observed(obs) => &obs.parent_ids,
            Node::Parameter(_) | Node::Constant(_) => &[],
        }
    }

    /// The (kind-erased) quantity type of this node, if it carries one.
    pub fn quantity_type(&self) -> &dyn DynQuantityType<B> {
        match self {
            Node::Constant(node) => &*node.quantity_type,
            Node::Parameter(node) => &*node.quantity_type,
            Node::Stochastic(node) => &*node.quantity_type,
            Node::Observed(node) => &*node.quantity_type,
            Node::Mapped(node) => &*node.quantity_type,
            Node::LogLikelihood(node) => &*node.quantity_type,
        }
    }

    /// Whether this node is discrete (integer-valued, non-differentiable).
    pub fn is_discrete(&self) -> bool {
        self.quantity_type().kind() == QuantityKind::Discrete
    }

    /// Whether this node is continuous (float-valued, differentiable).
    pub fn is_continuous(&self) -> bool {
        self.quantity_type().kind() != QuantityKind::Discrete
    }
}

/// A fixed, immutable value.
#[derive(Clone)]
pub struct ConstantNode<B: Backend> {
    pub id: Id,
    pub value: QuantityValue<B>,
    pub quantity_type: Arc<dyn DynQuantityType<B>>,
}

/// A free variable with an initial value, intended to be optimized/inferred.
#[derive(Clone)]
pub struct ParameterNode<B: Backend> {
    pub id: Id,
    pub quantity_type: Arc<dyn DynQuantityType<B>>,
}

/// Represents a stochastic quantity with a known conditional distribution.
#[derive(Clone)]
pub struct StochasticNode<B: Backend> {
    pub id: Id,
    pub sampler: Arc<dyn DynGraphSampler<B>>,
    pub parent_ids: Vec<Id>,
    pub quantity_type: Arc<dyn DynQuantityType<B>>,
}

impl<B: Backend> Debug for StochasticNode<B> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("StochasticNode")
            .field("id", &self.id)
            .field("parent_ids", &self.parent_ids)
            .finish_non_exhaustive()
    }
}

/// A stochastic quantity bound to a concrete, fixed value.
#[derive(Clone)]
pub struct ObservedNode<B: Backend> {
    pub id: Id,
    pub value: QuantityValue<B>,
    pub sampler: Arc<dyn DynGraphSampler<B>>,
    pub parent_ids: Vec<Id>,
    pub quantity_type: Arc<dyn DynQuantityType<B>>,
}

impl<B: Backend> Debug for ObservedNode<B> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ObservedNode")
            .field("id", &self.id)
            .field("parent_ids", &self.parent_ids)
            .finish_non_exhaustive()
    }
}

/// A quantity which is entirely determined by its parents.
///
/// If no descendant of the mapped node is either stochastic or a free variable, then evaluating its
/// value would amount to walking the graph from constant descendants and evaluating determinsitic
/// functions. We short-circuit this computation by caching the constant value on creation.
#[derive(Clone)]
pub struct MappedNode<B: Backend> {
    pub id: Id,
    pub op: Arc<dyn DynGraphOperation<B>>,
    pub parent_ids: Vec<Id>,
    pub quantity_type: Arc<dyn DynQuantityType<B>>,
    pub value: Option<QuantityValue<B>>,
}

/// A mapped real value that corresponds to a log-density in the model.
///
/// Just as with a mapped node, the log-density is cached if none of its descendants are free or
/// stochastic.
#[derive(Clone)]
pub struct LogLikelihoodNode<B: Backend> {
    pub id: Id,
    pub op: Arc<dyn DynGraphOperation<B>>,
    pub stochastic_id: Option<Id>,
    pub parent_ids: Vec<Id>,
    pub quantity_type: Arc<dyn DynQuantityType<B>>,
    pub value: Option<QuantityValue<B>>,
}
