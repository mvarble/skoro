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

use skoro_backend::{NdArrayBackend, Point};
use skoro_core::{error::SkoroResult, id::Id, kind::Continuous};
use skoro_model::{
    ModelBuilderOps, ModelClientOps,
    distribution::{Distribution, Sampler},
    map::{register_map_slice, register_map1},
    model::ModelClient,
    q::Q,
};
use skoro_operation::{
    Input,
    ops::{LiteralScalarMultiplication, TangentConcat},
};
use skoro_riemann::{euclidean::EuclideanD, real::Real};

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

/// A distribution whose `log_density` does **not** rebuild over its parents — the one shape
/// the old drop-guard protocol could not survive.
///
/// Nothing in the `Distribution` contract says the density must be expressed over the
/// quantity's parents, so an implementation is free to score `x` alone. Under the
/// `is_mapped_leaf` flags that was fatal: the mean's handle dropped with its flag still
/// set — nothing ever cleared it, since clearing happened as a side effect of building a
/// child — and `cleanup()` called `remove` on a node with a live child, panicking on what
/// it called an internal error. With eligibility re-derived at collection time, the same
/// sequence is ordinary: the mean has a child (the stochastic node), so it is skipped.
struct LazyDistr {
    /// Never read, and that is the point: holding it is what keeps the mean's only handle
    /// alive until this struct is consumed, which is the moment the test is about.
    #[allow(dead_code)]
    mean: Q<B, Real>,
    ids: [Id; 1],
}

#[derive(Debug, Clone)]
struct LazySampler;

impl Sampler<B, Continuous> for LazySampler {
    fn sample(&self, _batch_size: usize, _parameters: Vec<Input<B>>) -> Point<B> {
        unimplemented!("the test never generates")
    }
}

impl Distribution<B, Continuous> for LazyDistr {
    type Qty = Real;
    type Sampler = LazySampler;

    fn quantity_type(&self) -> Real {
        Real
    }

    fn sampler(&self) -> LazySampler {
        LazySampler
    }

    fn parent_ids(&self) -> &[Id] {
        &self.ids
    }

    fn log_density(&self, x: &Q<B, Real>) -> SkoroResult<Q<B, Real>> {
        register_map1(Real, LiteralScalarMultiplication::new(2.0), x)
    }
}

/// Building with a lazy distribution must not panic in `cleanup()`, and the skipped
/// parent must survive.
#[test]
fn a_distribution_that_skips_its_parents_cannot_panic_cleanup() {
    let model = ModelClient::<B>::default();
    let c = model.add_constant(Real, &2.0f64);
    let mean = register_map1(Real, LiteralScalarMultiplication::new(2.0), &c).unwrap();
    let mean_id = mean.id();

    // The distribution holds the only handle to its mean — a mapped node — and that
    // handle drops at the end of this call, with the stochastic node still pointing at
    // the mean.
    ModelBuilderOps::add_quantity(
        &model,
        LazyDistr {
            mean,
            ids: [mean_id],
        },
    )
    .unwrap();

    model.cleanup();
    assert!(
        model.server_call(|server| server.graph().contains(mean_id)),
        "a node with a live child is skipped, not collected and not a panic"
    );
}

/// Demotion must reclaim a severed parent the orphaned density never named.
///
/// The reason `demote_to_parameter` reports what it severed instead of leaning on the
/// density's cascade. For an ordinary `Normal` the two coincide — its `log_density` is built
/// over the same mean and covariance the stochastic node takes as parents, so collecting the
/// orphaned density walks down to them anyway, and dropping the severed list changes nothing
/// observable. `LazyDistr` breaks that coincidence, which is the only way to tell the two
/// apart: its `mean` is a parent of the quantity and of nothing else, so no cascade from the
/// density will ever reach it.
///
/// The bare `cleanup()` before the demote is load-bearing. `mean`'s handle died inside
/// `add_quantity`, which queued it — so without a collection in between, the demote would
/// find that stale candidate still waiting and reclaim `mean` whether or not it reported
/// anything. Consuming it first is what makes this a test of the report rather than of the
/// queue's leftovers.
#[test]
fn demotion_reclaims_a_parent_the_density_never_named() {
    let model = ModelClient::<B>::default();
    let c = model.add_constant(Real, &2.0f64);
    let mean = register_map1(Real, LiteralScalarMultiplication::new(2.0), &c).unwrap();
    let mean_id = mean.id();

    let x = ModelBuilderOps::add_quantity(
        &model,
        LazyDistr {
            mean,
            ids: [mean_id],
        },
    )
    .unwrap();

    model.cleanup();
    assert!(model.server_call(|server| server.graph().contains(mean_id)));

    model.demote_to_parameter(&x).unwrap();

    assert!(
        !model.server_call(|server| server.graph().contains(mean_id)),
        "a severed parent outside the density's subgraph must still be reclaimed"
    );
    assert!(
        model.server_call(|server| server.graph().contains(c.id())),
        "its own parent is still held by a handle"
    );
}
