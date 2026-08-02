//! # skoro-graph
//!
//! The probabilistic DAG. [`graph::Graph`] is an arena of [`node::Node`] variants (constants,
//! parameters, stochastic quantities, and mapped transformations--each of which may be continuous
//! or discrete) that represents a model's generative structure.
//! [`graph::Graph::compile_backward_from`] lowers a subgraph into a fast index-based
//! [`compiler::CompiledSequence`] for forward evaluation and reverse-mode differentiation. Also
//! defines the kind-tagged quantity-type trait ([`QuantityType`](quantity::QuantityType), with
//! [`ContinuousQuantity`](quantity::ContinuousQuantity) and
//! [`DiscreteQuantity`](quantity::DiscreteQuantity) as the kind-specific capability traits).

pub mod compiler;
pub mod graph;
pub mod node;
pub mod operation;
pub mod point_map;
pub mod quantity;
pub mod sampler;
