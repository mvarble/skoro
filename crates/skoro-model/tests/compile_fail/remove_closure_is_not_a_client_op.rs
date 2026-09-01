//! Retiring a cone of the model is a *builder* capability.
//!
//! `ModelClientOps::remove` is safe at the client tier because it refuses to break a reference:
//! an estimator holding it cannot delete a factor out from under the records it was handed —
//! `remove` returns `SkoroError::NodeStillBound` for a node a record names. `remove_closure`
//! deliberately does the opposite: it takes the referents too. So it belongs with the calls that
//! shape the model, not with the ones that only read it.
//!
//! This is the half a passing test cannot assert. If `remove_closure` ever moves to
//! `ModelClientOps`, every estimator gains it silently, and this is the only thing that will
//! notice.

use skoro_backend::NdArrayBackend;
use skoro_core::{error::SkoroResult, id::Id};
use skoro_graph::node::Node;
use skoro_model::ModelClientOps;

type B = NdArrayBackend<f64, i32>;

/// Bounded exactly as an estimator is: it evaluates and differentiates, and cannot grow or
/// dismantle the model.
fn an_estimator<M: ModelClientOps<B>>(model: &M, roots: &[Id]) -> SkoroResult<Vec<Node<B>>> {
    // Allowed: refuses to break a reference.
    let _ = model.remove(roots)?;
    // Not allowed: takes the cone.
    model.remove_closure(roots)
}

fn main() {
    let _ = an_estimator::<skoro_model::model::ModelClient<B>>;
}
