use skoro_backend::{CotangentFrame, NdArrayBackend, Point, Tensor};
use skoro_core::any_tensor::QuantityValue;
use skoro_core::id::Id;
use skoro_core::kind::Continuous;
use skoro_core::sampler::Sampler;
use skoro_graph::{
    graph::Graph,
    node::{MappedNode, Node, ObservedNode},
    operation::DiffGraphOperation,
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
        .add_mapped(Real, DiffGraphOperation(DummyOperation), vec![stoch_id])
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
    let _m1 = graph
        .add_mapped(Real, DiffGraphOperation(DummyOperation), vec![p1])
        .unwrap();

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
                .as_unbatched()
                .expect("observe stores an unbatched value")
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

/// `remove_closure` takes a whole subtree, including a diamond, in one call.
///
/// `remove` alone rejects any of these roots: each has a child outside the batch.
#[test]
fn remove_closure_takes_a_diamond_subtree() {
    let mut graph = Graph::<B>::new();
    let root = graph.add_parameter(Real);
    let keep = graph.add_parameter(Real);

    let left = graph
        .add_mapped(Real, DiffGraphOperation(DummyOperation), vec![root])
        .unwrap();
    let right = graph
        .add_mapped(Real, DiffGraphOperation(DummyOperation), vec![root])
        .unwrap();
    // Joins the two branches, and also depends on a node that must survive.
    let join = graph
        .add_mapped(
            Real,
            DiffGraphOperation(DummyOperation),
            vec![left, right, keep],
        )
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
#[test]
fn removing_one_child_leaves_a_shared_parent_intact() {
    let mut graph = Graph::<B>::new();
    let shared = graph.add_parameter(Real);
    let first = graph
        .add_mapped(Real, DiffGraphOperation(DummyOperation), vec![shared])
        .unwrap();
    let second = graph
        .add_mapped(Real, DiffGraphOperation(DummyOperation), vec![shared])
        .unwrap();

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
            .map(Input::unwrap_float)
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
#[test]
fn observing_folds_a_descendant_whose_parents_differ_in_depth() {
    let mut graph = Graph::<B>::new();
    let stoch = graph.add_stochastic(Real, DummySampler, vec![]).unwrap();

    let near = graph
        .add_mapped(Real, DiffGraphOperation(SumOperation), vec![stoch])
        .unwrap();
    let middle = graph
        .add_mapped(Real, DiffGraphOperation(SumOperation), vec![stoch])
        .unwrap();
    let far = graph
        .add_mapped(Real, DiffGraphOperation(SumOperation), vec![middle])
        .unwrap();
    // Parents at depth 1 and depth 3.
    let join = graph
        .add_mapped(Real, DiffGraphOperation(SumOperation), vec![near, far])
        .unwrap();

    // Nothing is concrete while the root is still stochastic.
    for id in [near, middle, far, join] {
        assert!(
            !folded(&graph, id),
            "{id:?} folded before the root was fixed"
        );
    }

    graph.observe::<_, Real, _>(stoch, &3.0).unwrap();
    assert_eq!(graph.pinned_batch(), None);

    for id in [near, middle, far, join] {
        assert!(
            folded(&graph, id),
            "{id:?} did not fold after observing the root"
        );
    }

    // And unobserving must clear all of them again.
    graph.unobserve(stoch).unwrap();
    for id in [near, middle, far, join] {
        assert!(
            !folded(&graph, id),
            "{id:?} kept a stale fold after unobserve"
        );
    }
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

    let plain = graph
        .add_mapped(Real, DiffGraphOperation(DummyOperation), vec![stoch])
        .unwrap();
    assert_eq!(graph.stochastic_of(plain), None);

    let likelihood = graph
        .add_mapped(Real, DiffGraphOperation(DummyOperation), vec![stoch])
        .unwrap();
    graph.mark_log_likelihood(likelihood).unwrap();
    assert_eq!(graph.stochastic_of(likelihood), None);
    assert!(graph.log_likelihoods().contains(&likelihood));

    let density = graph
        .add_mapped(Real, DiffGraphOperation(DummyOperation), vec![stoch])
        .unwrap();
    graph.mark_log_density(density, stoch).unwrap();
    assert_eq!(graph.stochastic_of(density), Some(stoch));
    assert_eq!(graph.log_density_of(stoch), Some(density));
}

/// Demoting a density back to a bare likelihood must drop the reverse index too.
#[test]
fn re_marking_a_density_as_a_likelihood_clears_the_reverse_index() {
    let mut graph = Graph::<B>::new();
    let stoch = graph.add_stochastic(Real, DummySampler, vec![]).unwrap();
    let density = graph
        .add_mapped(Real, DiffGraphOperation(DummyOperation), vec![stoch])
        .unwrap();

    graph.mark_log_density(density, stoch).unwrap();
    assert_eq!(graph.log_density_of(stoch), Some(density));
    assert_eq!(graph.stochastic_of(density), Some(stoch));
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

    let first = graph
        .add_mapped(Real, DiffGraphOperation(DummyOperation), vec![stoch])
        .unwrap();
    graph.mark_log_density(first, stoch).unwrap();

    let second = graph
        .add_mapped(Real, DiffGraphOperation(DummyOperation), vec![stoch])
        .unwrap();
    assert!(matches!(
        graph.mark_log_density(second, stoch),
        Err(SkoroError::DuplicateDensity { .. })
    ));

    // The rejected call must leave the graph exactly as it was.
    assert_eq!(graph.log_density_of(stoch), Some(first));
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
        .add_mapped(Real, DiffGraphOperation(DummyOperation), vec![first_stoch])
        .unwrap();
    graph.mark_log_density(density, first_stoch).unwrap();

    assert!(matches!(
        graph.mark_log_density(density, second_stoch),
        Err(SkoroError::DuplicateDensity { .. })
    ));

    assert_eq!(graph.log_density_of(first_stoch), Some(density));
    assert_eq!(graph.log_density_of(second_stoch), None);
    assert_eq!(graph.stochastic_of(density), Some(first_stoch));
}

// ---------------------------------------------------------------------------------------------
// Batch-pinned constant values
// ---------------------------------------------------------------------------------------------

/// The elements of a folded mapped node's stored value, whatever its rank.
///
/// Also asserts which variant it is, because that is half of what these tests are about: a fold
/// over a batch-pinned parent has to *stay* rank-2. Collapsing it to rank-1 would keep row 0 and
/// silently discard the rest, and nothing downstream would announce that.
fn folded_rows(graph: &Graph<B>, id: skoro_core::id::Id, expect_pinned: Option<usize>) -> Vec<f32> {
    let Some(Node::Mapped(MappedNode { value: Some(v), .. })) = graph.node(id) else {
        panic!("expected a folded mapped node");
    };
    assert_eq!(v.pinned_batch(), expect_pinned, "wrong fold representation");
    match v {
        QuantityValue::Batched(input) => input
            .clone()
            .unwrap_float()
            .into_data()
            .to_vec::<f32>()
            .unwrap(),
        QuantityValue::Unbatched(c) => c
            .clone()
            .unwrap_float()
            .into_data()
            .to_vec::<f32>()
            .unwrap(),
    }
}

/// A fold over batch-pinned parents keeps every row.
///
/// The failure this guards against is the tempting one: folding at batch 1 and `unbatch`ing the
/// result, which is what the walk did before batch-pinned values existed. That path silently
/// returns row 0 three times.
#[test]
fn a_fold_over_batched_parents_keeps_its_rows() {
    let mut graph = Graph::<B>::new();
    let a = graph
        .add_batched_constant(Real, &[1.0f32, 2.0, 3.0])
        .unwrap();
    let b = graph
        .add_batched_constant(Real, &[10.0f32, 20.0, 30.0])
        .unwrap();
    let sum = graph
        .add_mapped(Real, DiffGraphOperation(SumOperation), vec![a, b])
        .unwrap();

    assert!(folded(&graph, sum));
    assert_eq!(folded_rows(&graph, sum, Some(3)), vec![11.0, 22.0, 33.0]);
}

/// A batch-agnostic parent broadcasts into a fold that a sibling has pinned.
///
/// This is the case that makes the fold batch a property of the *parent set* rather than of any
/// one parent: the rank-1 value has to expand to 3, not the pinned one collapse to 1.
#[test]
fn a_fold_mixing_pinned_and_agnostic_parents_runs_at_the_pinned_batch() {
    let mut graph = Graph::<B>::new();
    let batched = graph
        .add_batched_constant(Real, &[1.0f32, 2.0, 3.0])
        .unwrap();
    let shared = graph.add_constant(Real, &100.0f32);
    let sum = graph
        .add_mapped(
            Real,
            DiffGraphOperation(SumOperation),
            vec![batched, shared],
        )
        .unwrap();

    assert_eq!(folded_rows(&graph, sum, Some(3)), vec![101.0, 102.0, 103.0]);
}

/// Where nothing is pinned, folding is exactly what it was: rank-1, and reusable at any batch.
#[test]
fn a_fold_over_agnostic_parents_stays_agnostic() {
    let mut graph = Graph::<B>::new();
    let a = graph.add_constant(Real, &2.0f32);
    let b = graph.add_constant(Real, &5.0f32);
    let sum = graph
        .add_mapped(Real, DiffGraphOperation(SumOperation), vec![a, b])
        .unwrap();

    assert_eq!(folded_rows(&graph, sum, None), vec![7.0]);
}

/// A node which pins *different* batch-size cannot be added.
#[test]
fn pinning_different_batches_are_rejected() {
    let mut graph = Graph::<B>::new();
    let _ = graph
        .add_batched_constant(Real, &[1.0f32, 2.0, 3.0])
        .unwrap();

    assert!(matches!(
        graph.add_batched_constant(Real, &[1.0f32, 2.0, 3.0, 4.0, 5.0]),
        Err(skoro_core::error::SkoroError::BatchSizeMismatch { .. })
    ));

    graph.validate_indices().unwrap();
    assert_eq!(graph.pinned_batch(), Some(3));
    assert_eq!(graph.node_count(), 1);

    let other = graph.add_parameter(Real);
    assert!(matches!(
        graph.observe_batched::<_, Real, _>(other, &[1.0f32, 2.0, 3.0, 4.0, 5.0]),
        Err(skoro_core::error::SkoroError::BatchSizeMismatch { .. })
    ));
    graph.validate_indices().unwrap();
    assert_eq!(graph.pinned_batch(), Some(3));
    assert_eq!(graph.node_count(), 2);
}

/// Observing folds through a batch-pinned parent, and unobserving clears the fold again.
#[test]
fn observing_folds_through_a_pinned_parent_and_unobserving_clears_it() {
    let mut graph = Graph::<B>::new();
    let stoch = graph.add_stochastic(Real, DummySampler, vec![]).unwrap();
    let batched = graph
        .add_batched_constant(Real, &[1.0f32, 2.0, 3.0])
        .unwrap();
    let sum = graph
        .add_mapped(Real, DiffGraphOperation(SumOperation), vec![stoch, batched])
        .unwrap();

    assert!(!folded(&graph, sum));
    assert_eq!(graph.pinned_batch(), Some(3));

    graph.observe::<_, Real, _>(stoch, &10.0).unwrap();
    assert_eq!(graph.pinned_batch(), Some(3));

    println!("stoch: {:?}, batched: {:?}, sum: {:?}", stoch, batched, sum);
    if let Some(Node::Mapped(MappedNode { value, .. })) = graph.node(sum) {
        println!("{:#?}", value);
    } else {
        panic!("THIS SHOULD NOT HAPPEN!");
    }
    assert_eq!(folded_rows(&graph, sum, Some(3)), vec![11.0, 12.0, 13.0]);

    graph.unobserve(stoch).unwrap();
    assert!(!folded(&graph, sum));
    graph.validate_indices().unwrap();
}

/// Build the two-member stacked tangent every retirement test below starts from.
///
/// Returns the graph, the two scored parameters, the concatenation, and the node count from
/// before the call — five nodes later: an anchor and a per-block tangent per member, plus the
/// concatenation.
fn stacked_tangent_fixture() -> (Graph<B>, Id, Id, Id, usize) {
    use skoro_core::kind::Continuous;
    use skoro_graph::{point_map::PointMap, quantity::QuantityType};
    use skoro_operation::Input;

    let mut graph = Graph::<B>::new();
    let a = graph.add_parameter(Real);
    let b = graph.add_parameter(Real);
    let mut at = PointMap::<B>::default();
    for (id, v) in [(a, 1.0f32), (b, -2.0f32)] {
        at.insert(
            id,
            Input::Float(Tensor::<B, 2>::from_floats([[v]], &Default::default())),
            <Real as QuantityType<B, Continuous>>::into_dyn(Real),
        );
    }
    let before = graph.node_count();
    let joint = graph.stacked_tangent(&at, &[a, b]).unwrap();
    assert_eq!(graph.node_count() - before, 5);
    (graph, a, b, joint, before)
}

/// Retiring a joint residual reclaims the `2k` nodes `stacked_tangent` built for itself.
///
/// Those nodes are unreachable by every other route. They sit *upstream* of the concatenation, so
/// a downstream walk cannot find them, and no `Q` names them, so `cleanup()` never queues them.
/// Without the private-node sweep a sliding window leaks `2k` nodes per tick — which is the shape
/// `docs/skoro.md` GAP 3 had left open.
///
/// Rooted at a **scored quantity**, which is what a window actually retires. Note that the factor
/// body is already downstream of it, so what the sweep adds is the surviving member's half.
#[test]
fn retiring_a_stacked_tangent_reclaims_its_private_nodes() {
    let (mut graph, a, b, joint, before) = stacked_tangent_fixture();

    let removed = graph.remove_closure(&[a]).unwrap();
    // `a` itself, its per-block tangent, the concatenation — then both anchors and `b`'s
    // per-block tangent, none of which any walk from `a` reaches.
    assert_eq!(removed.len(), 6);
    assert_eq!(graph.node_count(), before - 1);
    assert!(graph.node(joint).is_none());
    assert!(graph.node(b).is_some(), "the scored survivor is untouched");
    graph.validate_indices().unwrap();
}

/// The sweep runs to a fixpoint, not once.
///
/// An anchor is its per-block tangent's parent, so it is not childless until that tangent has
/// gone — and the tangent is only reclaimed in the pass before. A single pass over the private set
/// would strand one anchor, always or never depending on `HashSet` iteration order. This is the
/// case that catches it: rooting at the concatenation makes *both* per-block tangents reclaimable
/// in pass one and both anchors only in pass two.
#[test]
fn reclaiming_private_nodes_runs_to_a_fixpoint() {
    let (mut graph, a, b, joint, before) = stacked_tangent_fixture();

    let removed = graph.remove_closure(&[joint]).unwrap();
    assert_eq!(removed.len(), 5, "the concatenation and all four internals");
    assert_eq!(graph.node_count(), before);
    assert!(graph.node(a).is_some());
    assert!(graph.node(b).is_some());
    graph.validate_indices().unwrap();
}

/// A private node is reclaimed when nothing points at it, and not before.
///
/// The concatenation is deliberately *not* private — it is what the caller is handed — so a
/// removal elsewhere in the model must leave the whole subgraph standing. Without this the sweep
/// would be free to eat a live factor the first time anything else was retired.
#[test]
fn an_unrelated_removal_reclaims_nothing() {
    let (mut graph, _a, _b, joint, before) = stacked_tangent_fixture();
    let spare = graph.add_parameter(Real);

    let removed = graph.remove_closure(&[spare]).unwrap();
    assert_eq!(removed.len(), 1);
    assert_eq!(graph.node_count(), before + 5);
    assert!(graph.node(joint).is_some());
    graph.validate_indices().unwrap();
}
