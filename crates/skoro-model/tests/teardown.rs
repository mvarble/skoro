//! Retiring a cone of the model: `remove_closure` from the client side.
//!
//! `ModelClientOps::remove` is leaf-only and all-or-nothing — it refuses a batch any survivor
//! still points into — so retiring a state means computing the downward closure by hand at every
//! call site. `ModelBuilderOps::remove_closure` is that walk, done once.
//!
//! Two things are checked here that are not checked at the graph layer, because neither is
//! visible from a `Graph`:
//!
//! - the closure is reachable through the **builder** trait and not the client one, which is what
//!   keeps an `NlsStructure`-bounded estimator from deleting a cone of its own evidence. The
//!   negative half is a `trybuild` case, since a test that compiles cannot assert that another
//!   does not;
//! - `cleanup()` survives it. A closure removal deliberately takes nodes the caller *does* hold a
//!   `Q` for, so dropping that handle afterwards queues an id naming nothing — and cleanup used to
//!   `expect` on exactly that.

use skoro_backend::NdArrayBackend;
use skoro_model::{
    ModelBuilderOps, ModelClientOps, map::register_map_slice, model::ModelClient, q::Q,
};
use skoro_operation::ops::TangentConcat;
use skoro_riemann::euclidean::EuclideanD;

type B = NdArrayBackend<f64, i32>;

/// A mapped node over `blocks`, of the summed width. Any operation would do; `TangentConcat` is
/// the one with no arity limit.
fn concat(blocks: &[&Q<B, EuclideanD>]) -> Q<B, EuclideanD> {
    let widths = vec![1; blocks.len()];
    register_map_slice(EuclideanD(blocks.len()), TangentConcat(widths), blocks).unwrap()
}

fn node_count(model: &ModelClient<B>) -> usize {
    model.server_call(|server| server.graph().node_count())
}

/// The closure takes everything downstream, where `remove` would have refused.
#[test]
fn remove_closure_takes_the_cone_a_leaf_removal_refuses() {
    let model = ModelClient::<B>::default();
    let a = model.add_parameter(EuclideanD(1));
    let b = model.add_parameter(EuclideanD(1));
    let ab = concat(&[&a, &b]);
    let top = concat(&[&ab]);

    // `remove` is the primitive that will not do this: `a` still has a child.
    assert!(ModelClientOps::remove(&model, &[a.id()]).is_err());

    let removed = ModelBuilderOps::remove_closure(&model, &[a.id()]).unwrap();
    assert_eq!(
        removed.len(),
        3,
        "a, the concatenation over it, and the top"
    );
    assert_eq!(node_count(&model), 1, "b alone");
    assert!(model.server_call(|server| server.graph().contains(b.id())));

    // The handles outlive the nodes; dropping them must stay uneventful.
    drop((ab, top));
    model.cleanup();
    assert_eq!(node_count(&model), 1);
}

/// Dropping a `Q` for a node the closure already took must not panic in `cleanup()`.
///
/// This is the failure a caller meets first, and it is specific to the closure form: `remove` is
/// leaf-only, so reaching it there means removing precisely the node whose handle you hold, while
/// `remove_closure` takes nodes you do not name. `cleanup` is a garbage collector, so finding a
/// node already collected is ordinary rather than an internal error.
#[test]
fn cleanup_tolerates_a_node_the_closure_already_took() {
    let model = ModelClient::<B>::default();
    let a = model.add_parameter(EuclideanD(1));
    let b = model.add_parameter(EuclideanD(1));
    let ab = concat(&[&a, &b]);

    // Takes `ab` too, which is a mapped leaf the caller still holds.
    ModelBuilderOps::remove_closure(&model, &[a.id()]).unwrap();
    assert_eq!(node_count(&model), 1);

    // Queues `ab`'s id, which now names nothing.
    drop(ab);
    model.cleanup();

    assert_eq!(node_count(&model), 1);
    assert!(model.server_call(|server| server.graph().contains(b.id())));
}

/// Removing an already-removed subtree is a no-op rather than an error.
///
/// A sliding window retires by policy, not by bookkeeping — two policies may name overlapping
/// cones on the same tick — so this has to be idempotent to be usable.
#[test]
fn a_second_removal_of_the_same_cone_is_a_no_op() {
    let model = ModelClient::<B>::default();
    let a = model.add_parameter(EuclideanD(1));
    let b = model.add_parameter(EuclideanD(1));
    let ab = concat(&[&a, &b]);

    assert_eq!(
        ModelBuilderOps::remove_closure(&model, &[a.id()])
            .unwrap()
            .len(),
        2
    );
    assert!(
        ModelBuilderOps::remove_closure(&model, &[a.id()])
            .unwrap()
            .is_empty()
    );
    assert_eq!(node_count(&model), 1);
    drop(ab);
    model.cleanup();
    assert_eq!(node_count(&model), 1);
}
