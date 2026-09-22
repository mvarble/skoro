//! Removing even a leaf is a *builder* capability.
//!
//! `remove` never breaks a reference, which once put it on the read tier. That is a guarantee
//! about the graph, not about the model: removing a parameter shrinks the free set, and removing an
//! unbound term changes the log-density. So an estimator bounded on `ModelReaderOps` must not
//! reach it.

use skoro_backend::Flex;
use skoro_core::{error::SkoroResult, id::Id};
use skoro_graph::node::Node;
use skoro_model::ModelReaderOps;

type B = Flex;

fn an_estimator<M: ModelReaderOps<B>>(model: &M, ids: &[Id]) -> SkoroResult<Vec<Node<B>>> {
    // Fully qualified, so the error names this trait alone rather than every trait with a `remove`.
    <M as ModelReaderOps<B>>::remove(model, ids)
}

fn main() {
    let _ = an_estimator::<skoro_model::model::ModelClient<B>>;
}
