//! Retiring a cone of the model is a *builder* capability.
//!
//! `ModelClientOps::remove` is safe at the client tier because it refuses to break a reference:
//! an algorithm holding it cannot delete a factor out from under the certificate it was handed.
//! `remove_closure` deliberately does the opposite — it takes the referents too — so it belongs
//! with the calls that shape the model, not with the ones that only read it.
//!
//! This is the half a passing test cannot assert. If `remove_closure` ever moves to
//! `ModelClientOps`, every `NlsStructure` consumer gains it silently, and this is the only thing
//! that will notice.

use skoro_backend::NdArrayBackend;
use skoro_core::{error::SkoroResult, id::Id};
use skoro_graph::node::Node;
use skoro_model::NlsStructure;

type B = NdArrayBackend<f64, i32>;

/// Bounded exactly as an estimator is.
fn an_estimator<M: NlsStructure<B>>(model: &M, roots: &[Id]) -> SkoroResult<Vec<Node<B>>> {
    // Allowed: refuses to break a reference.
    let _ = model.remove(roots)?;
    // Not allowed: takes the cone.
    model.remove_closure(roots)
}

fn main() {
    let _ = an_estimator::<skoro_model::NlsModelClient<B>>;
}
