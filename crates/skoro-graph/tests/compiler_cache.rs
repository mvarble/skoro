//! The compiled-sequence cache: that it hits, that every mutation invalidates it, and that a
//! stale target id is an error rather than a panic.
//!
//! Identity is asserted with `Arc::ptr_eq`, not with a compile counter. The question these
//! tests ask is "did the caller get the *same* artifact back", which is what a cache is for;
//! a counter would answer it indirectly and would put test-only state in `Graph`.

mod common;

use std::sync::Arc;

use common::{Arithmetic, B, ConstantSampler, InertOperation, SumOperation, into_vec, points};
use skoro_core::error::SkoroError;
use skoro_core::id::Id;
use skoro_graph::compiler::{CompiledSequence, CompiledStep, ForwardInput, ForwardOptions};
use skoro_graph::graph::Graph;
use skoro_graph::operation::DiffGraphOperation;
use skoro_riemann::real::Real;

fn inert(graph: &mut Graph<B>, parents: Vec<Id>) -> Id {
    graph
        .add_mapped(Real, DiffGraphOperation(InertOperation), parents)
        .unwrap()
}

/// Compiling the same targets twice hands back the same artifact.
///
/// The property the whole phase exists for: a solver loop calling `evaluate_ids` every
/// iteration must not pay a compile every iteration.
#[test]
fn compiling_the_same_targets_twice_returns_the_same_sequence() {
    let m = Arithmetic::new();

    let first = m.graph.compile_backward_from(&[m.sq]).unwrap();
    let second = m.graph.compile_backward_from(&[m.sq]).unwrap();

    assert!(Arc::ptr_eq(&first, &second));
}

/// Different target lists are different programs and get different entries.
#[test]
fn different_targets_compile_separately() {
    let m = Arithmetic::new();

    let sq = m.graph.compile_backward_from(&[m.sq]).unwrap();
    let prod = m.graph.compile_backward_from(&[m.prod]).unwrap();

    assert!(!Arc::ptr_eq(&sq, &prod));
    // And the first entry survives the second: both stay live.
    let sq_again = m.graph.compile_backward_from(&[m.sq]).unwrap();
    assert!(Arc::ptr_eq(&sq, &sq_again));
}

/// Target *order* fixes step order, so it fixes every slot index -- the key cannot be
/// canonicalized.
///
/// `topological_sort` seeds its stack with the targets and pops LIFO, so `[prod, sq]` and
/// `[sq, prod]` are genuinely different programs. A sorted or deduped key would hand the
/// second caller the first caller's slot layout.
#[test]
fn target_order_is_part_of_the_key() {
    let m = Arithmetic::new();

    let forward = m.graph.compile_backward_from(&[m.prod, m.sq]).unwrap();
    let reversed = m.graph.compile_backward_from(&[m.sq, m.prod]).unwrap();

    assert!(!Arc::ptr_eq(&forward, &reversed));
    let forward_ids: Vec<_> = forward.steps.iter().map(|s| s.id()).collect();
    let reversed_ids: Vec<_> = reversed.steps.iter().map(|s| s.id()).collect();
    assert_ne!(
        forward_ids, reversed_ids,
        "the two orders must really compile to different step orders, \
         or this test is vacuous"
    );
}

/// A stale target id returns `NodeNotFound` rather than panicking.
///
/// `topological_sort` tolerates an unknown id and emits it verbatim; the lowering loop then
/// looked it up with `unwrap`. Reachable today from any caller holding an id across a
/// `remove`.
#[test]
fn a_removed_target_is_an_error_not_a_panic() {
    let mut m = Arithmetic::new();
    let removed = m.sq;
    m.graph.remove(&[removed]).unwrap();

    let err = m.graph.compile_backward_from(&[removed]).unwrap_err();

    assert!(
        matches!(err, SkoroError::NodeNotFound { id } if id == removed),
        "expected NodeNotFound for the removed target, got {err:?}"
    );
}

// Invalidation.
//
// `observe` is the case that matters for *correctness*, not just for identity: it turns a
// parameter into a constant at a fixed `Id`, which flips `constant_inputs` on every consumer.
// A cache that missed it would return values that are right and derivatives that are silently
// wrong.

#[test]
fn observing_a_parameter_recompiles_it_as_a_constant() {
    let m = Arithmetic::new();
    let before = m.graph.compile_backward_from(&[m.sum]).unwrap();
    assert!(matches!(
        step_for(&before, m.a),
        CompiledStep::Parameter { .. }
    ));
    assert!(!constant_inputs(&before, m.sum));

    let mut m = m;
    m.graph.observe::<_, Real, _>(m.a, &3.0f64).unwrap();
    let after = m.graph.compile_backward_from(&[m.sum]).unwrap();

    assert!(!Arc::ptr_eq(&before, &after));
    assert!(matches!(
        step_for(&after, m.a),
        CompiledStep::Constant { .. }
    ));
}

/// Observing *every* input of an operation makes it constant-input, so the differentiating
/// passes read its fold instead of re-entering it.
///
/// This is the flag a stale cache would get wrong while still returning the right value.
#[test]
fn observing_every_input_flips_constant_inputs() {
    let mut m = Arithmetic::new();
    assert!(!constant_inputs(
        &m.graph.compile_backward_from(&[m.sum]).unwrap(),
        m.sum
    ));

    m.graph.observe::<_, Real, _>(m.a, &3.0f64).unwrap();
    m.graph.observe::<_, Real, _>(m.b, &5.0f64).unwrap();

    assert!(constant_inputs(
        &m.graph.compile_backward_from(&[m.sum]).unwrap(),
        m.sum
    ));
}

/// `unobserve` is the other direction: a constant becomes a parameter again.
#[test]
fn unobserving_recompiles_it_as_a_parameter() {
    let mut m = Arithmetic::new();
    m.graph.observe::<_, Real, _>(m.a, &3.0f64).unwrap();
    let observed = m.graph.compile_backward_from(&[m.sum]).unwrap();

    m.graph.unobserve(m.a).unwrap();
    let after = m.graph.compile_backward_from(&[m.sum]).unwrap();

    assert!(!Arc::ptr_eq(&observed, &after));
    assert!(matches!(
        step_for(&after, m.a),
        CompiledStep::Parameter { .. }
    ));
}

/// Adding an unrelated node invalidates too.
///
/// A freshly issued `Id` cannot be an ancestor of any existing node, so this entry *would*
/// still be valid -- but the rule is that every mutator bumps, with no exceptions to defend.
/// This test is what keeps that rule from eroding.
#[test]
fn adding_an_unrelated_node_invalidates() {
    let mut m = Arithmetic::new();
    let before = m.graph.compile_backward_from(&[m.sum]).unwrap();

    m.graph.add_parameter(Real);

    let after = m.graph.compile_backward_from(&[m.sum]).unwrap();
    assert!(!Arc::ptr_eq(&before, &after));
}

/// Removing an unrelated node invalidates.
#[test]
fn removing_an_unrelated_node_invalidates() {
    let mut m = Arithmetic::new();
    let spectator = m.graph.add_parameter(Real);
    let before = m.graph.compile_backward_from(&[m.sum]).unwrap();

    m.graph.remove(&[spectator]).unwrap();

    let after = m.graph.compile_backward_from(&[m.sum]).unwrap();
    assert!(!Arc::ptr_eq(&before, &after));
}

/// `demote_to_parameter` turns a stochastic node into a free one at a fixed `Id`. The step kind
/// changes from `Sampler` to `Parameter` underneath an unchanged target list.
#[test]
fn demoting_a_stochastic_node_recompiles_it_as_a_parameter() {
    let mut graph = Graph::<B>::new();
    let z = graph
        .add_stochastic(Real, ConstantSampler(1.0), vec![])
        .unwrap();
    let y = inert(&mut graph, vec![z]);
    let before = graph.compile_backward_from(&[y]).unwrap();
    assert!(matches!(step_for(&before, z), CompiledStep::Sampler { .. }));

    graph.demote_to_parameter(z).unwrap();

    let after = graph.compile_backward_from(&[y]).unwrap();
    assert!(!Arc::ptr_eq(&before, &after));
    assert!(matches!(
        step_for(&after, z),
        CompiledStep::Parameter { .. }
    ));
}

/// A constant fold performed at `add_mapped` time changes what a *later* compile emits, so the
/// add that folded must invalidate as well.
#[test]
fn a_constant_fold_invalidates() {
    let mut graph = Graph::<B>::new();
    let k = graph.add_constant(Real, &2.0f64);
    let before = graph.compile_backward_from(&[k]).unwrap();

    // Every parent is constant, so this folds at add time.
    let folded = graph
        .add_mapped(Real, DiffGraphOperation(SumOperation), vec![k, k])
        .unwrap();

    let after = graph.compile_backward_from(&[k]).unwrap();
    assert!(!Arc::ptr_eq(&before, &after));

    // And the folded node really did fold, or the fixture is not exercising the fold path.
    let seq = graph.compile_backward_from(&[folded]).unwrap();
    assert!(constant_inputs(&seq, folded));
}

/// Cloning a graph does not share its cache.
///
/// The clone is a separate arena that will be mutated separately; sharing entries across them
/// would be the same staleness bug one indirection further out.
#[test]
fn a_clone_does_not_share_cached_sequences() {
    let m = Arithmetic::new();
    let original = m.graph.compile_backward_from(&[m.sum]).unwrap();

    let clone = m.graph.clone();
    let cloned = clone.compile_backward_from(&[m.sum]).unwrap();

    assert!(!Arc::ptr_eq(&original, &cloned));
}

/// Past the cap, the map is cleared rather than grown.
///
/// Target lists are caller-supplied, so key cardinality is caller-controlled; epoch
/// invalidation alone bounds staleness, not size. The first key is gone afterwards.
#[test]
fn the_cache_is_bounded() {
    let mut graph = Graph::<B>::new();
    let root = graph.add_parameter(Real);
    let mut ids = vec![root];
    for _ in 0..skoro_graph::graph::MAX_CACHED_SEQUENCES {
        let next = *ids.last().unwrap();
        ids.push(
            graph
                .add_mapped(Real, DiffGraphOperation(InertOperation), vec![next])
                .unwrap(),
        );
    }

    let first = graph.compile_backward_from(&[ids[0]]).unwrap();
    for id in &ids[1..] {
        graph.compile_backward_from(&[*id]).unwrap();
    }

    let first_again = graph.compile_backward_from(&[ids[0]]).unwrap();
    assert!(
        !Arc::ptr_eq(&first, &first_again),
        "the first entry should have been dropped when the cap was passed"
    );
}

/// Promoting a mapped node to a log-density changes what it lowers to: the `Mapped` arm keeps
/// the node's own quantity type, the `LogLikelihood` arm forces `Real`.
#[test]
fn marking_a_log_density_invalidates() {
    let mut graph = Graph::<B>::new();
    let z = graph
        .add_stochastic(Real, ConstantSampler(1.0), vec![])
        .unwrap();
    let density = inert(&mut graph, vec![z]);
    let before = graph.compile_backward_from(&[density]).unwrap();

    graph.mark_log_density(density, z).unwrap();

    let after = graph.compile_backward_from(&[density]).unwrap();
    assert!(!Arc::ptr_eq(&before, &after));
}

// Helpers

fn step_for(seq: &CompiledSequence<B>, id: Id) -> &CompiledStep<B> {
    seq.steps
        .iter()
        .find(|s| s.id() == id)
        .expect("id should be a step of this sequence")
}

fn constant_inputs(seq: &CompiledSequence<B>, id: Id) -> bool {
    match step_for(seq, id) {
        CompiledStep::Operation {
            constant_inputs, ..
        } => *constant_inputs,
        other => panic!("expected an operation step for {id:?}, got {other:?}"),
    }
}

// Reuse.
//
// Before the cache, every pass ran over a sequence nobody else had touched. Now the same
// artifact is handed to caller after caller, so "the passes allocate all their mutable state
// locally" stops being an implementation detail and becomes a contract.

/// A reused sequence evaluates each set of points on its own merits.
///
/// If any slot state survived a pass, the second evaluation would see the first one's values.
#[test]
fn a_reused_sequence_does_not_carry_state_between_evaluations() {
    let m = Arithmetic::new();

    let first = m
        .graph
        .evaluate(&points([(m.a, 3.0), (m.b, 5.0)]), m.sq)
        .unwrap();
    let second = m
        .graph
        .evaluate(&points([(m.a, 1.0), (m.b, 1.0)]), m.sq)
        .unwrap();
    let third = m
        .graph
        .evaluate(&points([(m.a, 3.0), (m.b, 5.0)]), m.sq)
        .unwrap();

    assert_eq!(into_vec(first), vec![64.0]);
    assert_eq!(into_vec(second), vec![4.0]);
    assert_eq!(into_vec(third), vec![64.0]);
}

/// Two threads may hold and run the same cached sequence at once.
///
/// This is what the `Arc` in the return type promises. The passes take `&self` and build every
/// slot vector locally, so the sharing is sound -- but the cache is the first thing in the tree
/// that actually hands one artifact to two callers.
#[test]
fn a_cached_sequence_can_be_evaluated_from_two_threads() {
    let m = Arithmetic::new();
    let (a, b, sq) = (m.a, m.b, m.sq);
    let sequence = m.graph.compile_backward_from(&[sq]).unwrap();

    let handles: Vec<_> = (0..4)
        .map(|i| {
            let sequence = sequence.clone();
            std::thread::spawn(move || {
                let v = i as f64;
                let pts = points([(a, v), (b, v)]);
                let out = sequence
                    .forward(ForwardInput {
                        points: &pts,
                        options: ForwardOptions {
                            batch_size: 1,
                            use_model_samplers: false,
                        },
                    })
                    .unwrap();
                let (value, _) = out.evaluations.get(&sq).unwrap();
                into_vec(value.clone().unwrap_float())
            })
        })
        .collect();

    for (i, handle) in handles.into_iter().enumerate() {
        // `sq = (a + b)^2` at `a = b = i`.
        let expected = (2.0 * i as f64).powi(2);
        assert_eq!(handle.join().unwrap(), vec![expected]);
    }
}

/// The load-bearing half of the test above, which four threads on a three-node graph are too
/// small to establish by racing: the type is shareable at all.
const _: fn() = || {
    fn assert_send_sync<T: Send + Sync>() {}
    assert_send_sync::<CompiledSequence<B>>();
};

// Partial mutation.
//
// `touch()` sits at the end of each mutator, so a method that mutates and *then* returns an
// error would leave the graph changed with the epoch unmoved -- a cached sequence compiled
// before the call would still be served afterwards. The fix is for such a method not to exist:
// a mutator either completes or changes nothing.

/// `remove` validates the whole batch before it deletes any of it.
///
/// Removing `[good, missing]` used to delete `good` and then fail on `missing`, leaving the
/// caller an error and a graph that had already lost a node. Beyond the atomicity, that path
/// returned without bumping the epoch, so a sequence compiled beforehand stayed live.
#[test]
fn a_removal_naming_a_missing_node_changes_nothing() {
    let mut graph = Graph::<B>::new();
    let good = graph.add_parameter(Real);
    // Stale by construction: issued, then removed.
    let missing = graph.add_parameter(Real);
    graph.remove(&[missing]).unwrap();
    let keeper = graph.add_parameter(Real);
    let before = graph.compile_backward_from(&[keeper]).unwrap();

    let Err(err) = graph.remove(&[good, missing]) else {
        panic!("removing a missing node should fail");
    };

    assert!(matches!(err, SkoroError::NodeNotFound { id } if id == missing));
    assert!(
        graph.contains(good),
        "the batch failed, so nothing in it should have been removed"
    );
    let after = graph.compile_backward_from(&[keeper]).unwrap();
    assert!(
        Arc::ptr_eq(&before, &after),
        "nothing changed, so the cached sequence should still stand"
    );
}

/// `add_mapped` naming a parent that does not exist registers no edges at all.
///
/// `register_children` checked each parent's existence *inside* its write loop, so a call
/// naming `[good, missing]` left `good` holding a child edge to an id that never became a
/// node -- a dangling index, and one the epoch never learned about because the error returned
/// before `touch`.
#[test]
fn a_mapped_node_with_a_missing_parent_registers_no_edges() {
    let mut graph = Graph::<B>::new();
    let good = graph.add_parameter(Real);
    let missing = graph.add_parameter(Real);
    graph.remove(&[missing]).unwrap();

    let before = graph.compile_backward_from(&[good]).unwrap();
    let err = graph
        .add_mapped(
            Real,
            DiffGraphOperation(InertOperation),
            vec![good, missing],
        )
        .unwrap_err();

    assert!(matches!(err, SkoroError::DanglingParent { parent_id } if parent_id == missing));
    assert!(
        graph.children_of(good).is_none_or(|kids| kids.is_empty()),
        "the call failed, so `good` should have gained no child"
    );
    graph.validate_indices().expect("indices stay coherent");
    let after = graph.compile_backward_from(&[good]).unwrap();
    assert!(
        Arc::ptr_eq(&before, &after),
        "nothing changed, so the cached sequence should still stand"
    );
}

/// `evaluate_ids` cannot be served a program the graph has since outgrown.
///
/// The cache is reached through `compile_backward_from`, and every evaluation entry point goes
/// through that one door -- but the door is what the other tests exercise directly. This is the
/// same question asked at the layer the payoff actually runs through: seed the cache, change
/// what the program means, and check the answer moved.
#[test]
fn evaluating_after_a_mutation_does_not_reuse_the_old_program() {
    let mut m = Arithmetic::new();
    let pts = points([(m.a, 3.0), (m.b, 5.0)]);

    // Seed the entry for these targets.
    let first = m.graph.evaluate(&pts, m.sum).unwrap();
    assert_eq!(into_vec(first), vec![8.0]);

    // `a` becomes a constant 10 at the same `Id`, so `sum` is now 15 -- and `a`'s supplied
    // point is ignored, because an observation lowers to a `Constant` step.
    m.graph.observe::<_, Real, _>(m.a, &10.0f64).unwrap();

    let second = m.graph.evaluate(&pts, m.sum).unwrap();
    assert_eq!(into_vec(second), vec![15.0]);
}
