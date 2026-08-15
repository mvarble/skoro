use skoro_backend::{CotangentFrame, NdArrayBackend, Point};
use skoro_core::kind::Continuous;
use skoro_core::sampler::Sampler;
use skoro_graph::{
    graph::Graph,
    node::{LogDensityNode, MappedNode, Node, ObservedNode},
};
use skoro_operation::{
    Input,
    diff::{DiffModeState, DiffOperation, DiffOperationForward},
};
use skoro_riemann::real::Real;

/// A dummy backend to use for testing
type B = NdArrayBackend<f32, i32>;

#[derive(Debug, Clone)]
struct DummySampler;

impl Sampler<B, Continuous> for DummySampler {
    fn sample(&self, _batch_size: usize, _parameters: Vec<Input<B>>) -> Point<B> {
        unimplemented!()
    }
}

#[derive(Debug, Clone)]
struct DummyOperation;

impl DiffOperation<B> for DummyOperation {
    type VjpState = ();
    type HvpState = ();
    fn forward(&self, _: Vec<Input<B>>, _: DiffModeState<B>) -> DiffOperationForward<B, (), ()> {
        unimplemented!()
    }
    fn vjp(&self, _: &(), _: CotangentFrame<B>) -> Vec<Option<CotangentFrame<B>>> {
        unimplemented!()
    }
    fn hvp(&self, _: &(), _: &(), _: CotangentFrame<B>) -> Vec<Option<CotangentFrame<B>>> {
        unimplemented!()
    }
}

#[test]
fn graph_topology() {
    let mut graph = Graph::<B>::new();

    // Add some base parameters/constants
    let c1 = graph.add_parameter(Real);
    let c2 = graph.add_constant(Real, &2.0);

    // Add a stochastic node with c1 and c2 as parents
    let stoch_id = graph
        .add_stochastic(Real, DummySampler, vec![c1, c2])
        .unwrap();

    // Verify it added both nodes
    assert!(graph.contains(stoch_id));

    // Verify the log density has the stochastic variable as a parent
    if let Node::Stochastic(stoch) = graph.node(stoch_id).unwrap() {
        assert!(stoch.parent_ids.contains(&c1));
        assert!(stoch.parent_ids.contains(&c2));
    } else {
        panic!();
    }

    // Add a mapped node depending on the stochastic output
    let m1 = graph
        .add_mapped(Real, DummyOperation, vec![stoch_id])
        .unwrap();
    assert!(graph.contains(m1));
}

#[test]
fn add_stochastic_dangling_parent_errors() {
    use skoro_core::error::SkoroError;
    use skoro_core::id::IdGenerator;

    let mut graph = Graph::<B>::new();
    // an Id that was never issued to this graph
    let bogus = IdGenerator::new().issue_id();
    assert!(matches!(
        graph.add_stochastic(Real, DummySampler, vec![bogus]),
        Err(SkoroError::DanglingParent { .. })
    ));
}

#[test]
fn remove_with_children_errors() {
    use skoro_core::error::SkoroError;

    let mut graph = Graph::<B>::new();
    let p1 = graph.add_parameter(Real);
    let _m1 = graph.add_mapped(Real, DummyOperation, vec![p1]).unwrap();

    // This should error because m1 depends on p1
    assert!(matches!(
        graph.remove(&[p1]),
        Err(SkoroError::NodeStillReferenced { .. })
    ));
}

#[test]
fn observe() {
    let mut graph = Graph::<B>::new();
    let stoch_id = graph.add_stochastic(Real, DummySampler, vec![]).unwrap();

    assert!(matches!(graph.node(stoch_id).unwrap(), Node::Stochastic(_)));

    graph.observe::<_, Real, _>(stoch_id, &5.0).unwrap();

    match graph.node(stoch_id).unwrap() {
        Node::Observed(ObservedNode { value, .. }) => {
            let d = value
                .clone()
                .unwrap_float()
                .into_data()
                .to_vec::<f32>()
                .unwrap();
            assert_eq!(d.len(), 1);
            assert_eq!(d[0], 5.0);
        }
        _ => panic!("observation did not work"),
    }
}

/// Unregistering a density must clear the link recorded on the node, not only the maps.
///
/// `remove` reads `LogDensityNode::stochastic_id` to decide which map entries to drop, so a
/// stale field there is a live hazard rather than a cosmetic leak — see the sibling test.
#[test]
fn unregister_density_clears_the_node_link() {
    let mut graph = Graph::<B>::new();
    let stoch = graph.add_stochastic(Real, DummySampler, vec![]).unwrap();
    let density = graph.add_mapped(Real, DummyOperation, vec![stoch]).unwrap();
    graph.mark_density(density, stoch).unwrap();

    assert_eq!(graph.density_of(stoch), Some(density));

    // Replace the distribution wholesale. This unregisters the old density and strands the
    // node in the arena, which the caller is expected to clean up afterwards.
    graph
        .update_stochastic::<Continuous, _>(stoch, DummySampler, vec![])
        .unwrap();

    assert_eq!(graph.density_of(stoch), None);
    match graph.node(density).unwrap() {
        Node::LogDensity(LogDensityNode { stochastic_id, .. }) => {
            assert_eq!(
                *stochastic_id, None,
                "the orphaned density still claims the stochastic quantity it no longer scores"
            );
        }
        _ => panic!("the density node should still be in the arena"),
    }
}

/// Removing an orphaned density must not disturb a density registered against the same
/// stochastic quantity in the meantime.
///
/// This is the corruption the stale `stochastic_id` field allowed: `remove` read the orphan's
/// claim on `stoch` and deleted the *replacement* density's reverse-map entry, leaving
/// `density_of` reporting `None` for a quantity still scored by a term in `log_likelihoods`.
#[test]
fn removing_an_orphaned_density_preserves_the_live_one() {
    let mut graph = Graph::<B>::new();
    let stoch = graph.add_stochastic(Real, DummySampler, vec![]).unwrap();

    let first = graph.add_mapped(Real, DummyOperation, vec![stoch]).unwrap();
    graph.mark_density(first, stoch).unwrap();

    // Replace the distribution: `first` is now an orphan in the arena.
    graph
        .update_stochastic::<Continuous, _>(stoch, DummySampler, vec![])
        .unwrap();

    // Register a replacement density against the same quantity.
    let second = graph.add_mapped(Real, DummyOperation, vec![stoch]).unwrap();
    graph.mark_density(second, stoch).unwrap();
    assert_eq!(graph.density_of(stoch), Some(second));

    // Now clean up the orphan, as `unregister_density`'s contract tells callers to.
    graph.remove(&[first]).unwrap();

    assert_eq!(
        graph.density_of(stoch),
        Some(second),
        "removing the orphan deleted the live density's reverse-map entry"
    );
    assert_eq!(graph.stochastic_of(second), Some(stoch));
    assert!(graph.log_likelihoods().contains(&second));
}

/// An orphaned density is discoverable, and reclaimable through `remove_closure`.
#[test]
fn an_orphaned_density_can_be_found_and_reclaimed() {
    let mut graph = Graph::<B>::new();
    let stoch = graph.add_stochastic(Real, DummySampler, vec![]).unwrap();
    let density = graph.add_mapped(Real, DummyOperation, vec![stoch]).unwrap();
    graph.mark_density(density, stoch).unwrap();
    assert!(graph.orphaned_densities().is_empty());

    // Replacing the distribution strands the old density in the arena.
    graph
        .update_stochastic::<Continuous, _>(stoch, DummySampler, vec![])
        .unwrap();
    assert_eq!(graph.orphaned_densities(), vec![density]);

    // The orphan keeps its parent edge, so it blocks removal of the cone above it.
    assert!(matches!(
        graph.remove(&[stoch]),
        Err(skoro_core::error::SkoroError::NodeStillReferenced { .. })
    ));

    graph.remove_closure(&graph.orphaned_densities()).unwrap();
    assert!(graph.orphaned_densities().is_empty());
    assert!(!graph.contains(density));
    // With the orphan gone the quantity is free to retire.
    graph.remove(&[stoch]).unwrap();
    assert_eq!(graph.node_count(), 0);
}

/// `remove_closure` takes a whole subtree, including a diamond, in one call.
///
/// `remove` alone rejects any of these roots: each has a child outside the batch.
#[test]
fn remove_closure_takes_a_diamond_subtree() {
    let mut graph = Graph::<B>::new();
    let root = graph.add_parameter(Real);
    let keep = graph.add_parameter(Real);

    let left = graph.add_mapped(Real, DummyOperation, vec![root]).unwrap();
    let right = graph.add_mapped(Real, DummyOperation, vec![root]).unwrap();
    // Joins the two branches, and also depends on a node that must survive.
    let join = graph
        .add_mapped(Real, DummyOperation, vec![left, right, keep])
        .unwrap();

    assert!(matches!(
        graph.remove(&[root]),
        Err(skoro_core::error::SkoroError::NodeStillReferenced { .. })
    ));

    let removed = graph.remove_closure(&[root]).unwrap();
    assert_eq!(removed.len(), 4, "root, both branches and the join");
    for id in [root, left, right, join] {
        assert!(!graph.contains(id));
    }

    // The surviving node is untouched, and its edge to the removed join is gone.
    assert!(graph.contains(keep));
    assert_eq!(graph.children_of(keep), None);
    graph.validate_indices().unwrap();
}

/// A shared parent keeps its remaining children when only one of them is removed.
///
/// This is the case `remove`'s `// TODO: check if I need to remove children` left uncertain.
#[test]
fn removing_one_child_leaves_a_shared_parent_intact() {
    let mut graph = Graph::<B>::new();
    let shared = graph.add_parameter(Real);
    let first = graph.add_mapped(Real, DummyOperation, vec![shared]).unwrap();
    let second = graph.add_mapped(Real, DummyOperation, vec![shared]).unwrap();

    graph.remove(&[first]).unwrap();

    assert!(!graph.contains(first));
    assert!(graph.contains(second));
    let children = graph.children_of(shared).expect("the parent keeps an edge");
    assert_eq!(children.len(), 1);
    assert!(children.contains(&second));
    graph.validate_indices().unwrap();

    // Removing the last child drops the parent's entry entirely rather than leaving an
    // empty set behind — `validate_indices` rejects those.
    graph.remove(&[second]).unwrap();
    assert_eq!(graph.children_of(shared), None);
    graph.validate_indices().unwrap();
}

/// An operation that actually computes, so the constant-folding walk can be exercised.
#[derive(Debug, Clone)]
struct SumOperation;

impl DiffOperation<B> for SumOperation {
    type VjpState = ();
    type HvpState = ();
    fn forward(
        &self,
        inputs: Vec<Input<B>>,
        diff_state: DiffModeState<B>,
    ) -> DiffOperationForward<B, (), ()> {
        let sum = inputs
            .into_iter()
            .map(Input::into_float)
            .reduce(|a, b| a + b)
            .expect("at least one input");
        match diff_state {
            DiffModeState::ForwardOnly => DiffOperationForward::forward_only(sum),
            _ => unimplemented!("constant folding only evaluates forward-only"),
        }
    }
    fn vjp(&self, _: &(), _: CotangentFrame<B>) -> Vec<Option<CotangentFrame<B>>> {
        unimplemented!()
    }
    fn hvp(&self, _: &(), _: &(), _: CotangentFrame<B>) -> Vec<Option<CotangentFrame<B>>> {
        unimplemented!()
    }
}

fn folded(graph: &Graph<B>, id: skoro_core::id::Id) -> bool {
    matches!(
        graph.node(id),
        Some(Node::Mapped(MappedNode { value: Some(_), .. }))
    )
}

/// Observing must fold every descendant that became concrete, including one whose parents sit
/// at different depths.
///
/// This is the case a plain `visited` set gets wrong. The breadth-first walk reaches `join`
/// from `near` before `far` has folded, so `join` cannot fold on that first visit; a walk that
/// refuses to revisit it would leave it unfolded forever. Ordering by dependency instead
/// visits `join` once, after both parents have settled.
#[test]
fn observing_folds_a_descendant_whose_parents_differ_in_depth() {
    let mut graph = Graph::<B>::new();
    let stoch = graph.add_stochastic(Real, DummySampler, vec![]).unwrap();

    let near = graph.add_mapped(Real, SumOperation, vec![stoch]).unwrap();
    let middle = graph.add_mapped(Real, SumOperation, vec![stoch]).unwrap();
    let far = graph.add_mapped(Real, SumOperation, vec![middle]).unwrap();
    // Parents at depth 1 and depth 3.
    let join = graph.add_mapped(Real, SumOperation, vec![near, far]).unwrap();

    // Nothing is concrete while the root is still stochastic.
    for id in [near, middle, far, join] {
        assert!(!folded(&graph, id), "{id:?} folded before the root was fixed");
    }

    graph.observe::<_, Real, _>(stoch, &3.0).unwrap();

    for id in [near, middle, far, join] {
        assert!(folded(&graph, id), "{id:?} did not fold after observing the root");
    }

    // And unobserving must clear all of them again.
    graph.unobserve(stoch).unwrap();
    for id in [near, middle, far, join] {
        assert!(!folded(&graph, id), "{id:?} kept a stale fold after unobserve");
    }
}

/// Every mutating method leaves the derived collections consistent.
///
/// `debug_validate_indices` already asserts this after each call in debug builds; this walks a
/// sequence that touches every mutator in turn and checks explicitly, so the intent survives if
/// the internal assertions are ever relaxed.
#[test]
fn mutations_leave_the_indices_consistent() {
    let mut graph = Graph::<B>::new();
    let check = |graph: &Graph<B>, step: &str| {
        if let Err(disagreement) = graph.validate_indices() {
            panic!("after {step}: {disagreement}");
        }
    };

    let base = graph.add_parameter(Real);
    let konst = graph.add_constant(Real, &2.0);
    let stoch = graph
        .add_stochastic(Real, DummySampler, vec![base, konst])
        .unwrap();
    check(&graph, "building the leaves");

    let density = graph.add_mapped(Real, DummyOperation, vec![stoch]).unwrap();
    graph.mark_density(density, stoch).unwrap();
    check(&graph, "mark_density");

    let factor = graph.add_mapped(Real, DummyOperation, vec![stoch]).unwrap();
    graph.mark_likelihood(factor).unwrap();
    check(&graph, "mark_likelihood");

    // Observed on its own leaf: observing a node with mapped children sends `check_cache`
    // into `DummyOperation::forward`, which is `unimplemented!()`. The mutators under test
    // are the same either way.
    let lone = graph.add_stochastic(Real, DummySampler, vec![]).unwrap();
    graph.observe::<_, Real, _>(lone, &1.0).unwrap();
    check(&graph, "observe");

    graph.unobserve(lone).unwrap();
    check(&graph, "unobserve");

    graph
        .update_stochastic::<Continuous, _>(stoch, DummySampler, vec![base])
        .unwrap();
    check(&graph, "update_stochastic");

    // `update_stochastic` orphaned `density`; clean it up the way callers are told to.
    graph.remove(&[density]).unwrap();
    check(&graph, "remove of the orphaned density");

    graph.demote_to_parameter(stoch).unwrap();
    check(&graph, "demote_to_parameter");

    graph.remove(&[factor]).unwrap();
    graph.remove(&[stoch]).unwrap();
    graph.remove(&[lone]).unwrap();
    check(&graph, "removing the remaining nodes");

    // The parent edges must have been torn down with their children.
    assert_eq!(graph.children_of(base), None);
    assert_eq!(graph.children_of(konst), None);
}

/// `stochastic_of` is derived from the node, so it must answer for every id, not only for
/// those that once had a map entry.
///
/// A free-standing likelihood factor scores no particular quantity, and a plain mapped node is
/// not a density at all; both are `None`.
#[test]
fn stochastic_of_distinguishes_density_likelihood_and_plain_mapped() {
    let mut graph = Graph::<B>::new();
    let stoch = graph.add_stochastic(Real, DummySampler, vec![]).unwrap();

    let plain = graph.add_mapped(Real, DummyOperation, vec![stoch]).unwrap();
    assert_eq!(graph.stochastic_of(plain), None);

    let likelihood = graph.add_mapped(Real, DummyOperation, vec![stoch]).unwrap();
    graph.mark_likelihood(likelihood).unwrap();
    assert_eq!(graph.stochastic_of(likelihood), None);
    assert!(graph.log_likelihoods().contains(&likelihood));

    let density = graph.add_mapped(Real, DummyOperation, vec![stoch]).unwrap();
    graph.mark_density(density, stoch).unwrap();
    assert_eq!(graph.stochastic_of(density), Some(stoch));
    assert_eq!(graph.density_of(stoch), Some(density));
}

/// Demoting a density back to a bare likelihood must drop the reverse index too.
#[test]
fn re_marking_a_density_as_a_likelihood_clears_the_reverse_index() {
    let mut graph = Graph::<B>::new();
    let stoch = graph.add_stochastic(Real, DummySampler, vec![]).unwrap();
    let density = graph.add_mapped(Real, DummyOperation, vec![stoch]).unwrap();

    graph.mark_density(density, stoch).unwrap();
    assert_eq!(graph.density_of(stoch), Some(density));

    // The previous binding is now read off the node rather than out of a map.
    graph.mark_likelihood(density).unwrap();

    assert_eq!(graph.stochastic_of(density), None);
    assert_eq!(
        graph.density_of(stoch),
        None,
        "the reverse index still points at a density that no longer scores this quantity"
    );
    assert!(graph.log_likelihoods().contains(&density));
}

/// A stochastic quantity has at most one density — the reading `DuplicateDensity` names.
///
/// The guard used to key on the density id, so a *second* density claiming an already-scored
/// quantity silently overwrote the first's reverse-map entry instead of erroring.
#[test]
fn a_second_density_for_one_quantity_is_rejected() {
    use skoro_core::error::SkoroError;

    let mut graph = Graph::<B>::new();
    let stoch = graph.add_stochastic(Real, DummySampler, vec![]).unwrap();

    let first = graph.add_mapped(Real, DummyOperation, vec![stoch]).unwrap();
    graph.mark_density(first, stoch).unwrap();

    let second = graph.add_mapped(Real, DummyOperation, vec![stoch]).unwrap();
    assert!(matches!(
        graph.mark_density(second, stoch),
        Err(SkoroError::DuplicateDensity { .. })
    ));

    // The rejected call must leave the graph exactly as it was.
    assert_eq!(graph.density_of(stoch), Some(first));
    assert_eq!(graph.stochastic_of(first), Some(stoch));
    assert_eq!(graph.stochastic_of(second), None);
    assert!(!graph.log_likelihoods().contains(&second));
    assert!(matches!(graph.node(second).unwrap(), Node::Mapped(_)));
}

/// The other direction: one quantity per density. Rebinding an existing density to a
/// different quantity would strand the original `stochastic_to_log_density` entry.
#[test]
fn rebinding_a_density_to_another_quantity_is_rejected() {
    use skoro_core::error::SkoroError;

    let mut graph = Graph::<B>::new();
    let first_stoch = graph.add_stochastic(Real, DummySampler, vec![]).unwrap();
    let second_stoch = graph.add_stochastic(Real, DummySampler, vec![]).unwrap();

    let density = graph
        .add_mapped(Real, DummyOperation, vec![first_stoch])
        .unwrap();
    graph.mark_density(density, first_stoch).unwrap();

    assert!(matches!(
        graph.mark_density(density, second_stoch),
        Err(SkoroError::DuplicateDensity { .. })
    ));

    assert_eq!(graph.density_of(first_stoch), Some(density));
    assert_eq!(graph.density_of(second_stoch), None);
    assert_eq!(graph.stochastic_of(density), Some(first_stoch));
}

/// Re-marking a density that `unregister_density` demoted must restore it as a likelihood
/// term, not just as a map entry.
#[test]
fn re_marking_an_unregistered_density_restores_it_as_a_term() {
    let mut graph = Graph::<B>::new();
    let stoch = graph.add_stochastic(Real, DummySampler, vec![]).unwrap();
    let density = graph.add_mapped(Real, DummyOperation, vec![stoch]).unwrap();
    graph.mark_density(density, stoch).unwrap();
    assert!(graph.log_likelihoods().contains(&density));

    // Drops it from the likelihood terms and leaves it a `LogDensity` node in the arena.
    graph
        .update_stochastic::<Continuous, _>(stoch, DummySampler, vec![])
        .unwrap();
    assert!(!graph.log_likelihoods().contains(&density));

    // Re-marking now takes the `LogDensity` arm, which previously wrote the maps without
    // putting the node back among the terms.
    graph.mark_density(density, stoch).unwrap();

    assert_eq!(graph.density_of(stoch), Some(density));
    assert!(
        graph.log_likelihoods().contains(&density),
        "a re-marked density is registered in the maps but does not score the model"
    );
}
