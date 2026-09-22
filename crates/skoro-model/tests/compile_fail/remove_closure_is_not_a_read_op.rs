//! Retiring a cone of the model is a *builder* capability.
//!
//! `remove_closure` takes a quantity's referents with it — the term that scores it and every
//! binding into it — so it can dismantle any part of the model. An estimator bounded on
//! `ModelReaderOps` must not be able to reach it. If it ever moves to `ModelReaderOps`, every
//! estimator gains it silently, and this is the only thing that will notice.

use skoro_backend::Flex;
use skoro_core::{error::SkoroResult, id::Id};
use skoro_graph::node::Node;
use skoro_model::ModelReaderOps;

type B = Flex;

/// Bounded exactly as an estimator is: it evaluates and differentiates, and cannot change the
/// model.
fn an_estimator<M: ModelReaderOps<B>>(model: &M, roots: &[Id]) -> SkoroResult<Vec<Node<B>>> {
    model.remove_closure(roots)
}

fn main() {
    let _ = an_estimator::<skoro_model::model::ModelClient<B>>;
}
