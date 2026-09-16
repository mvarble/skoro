//! Slot liveness: dropping an operation's value once nothing below it will read it again.
//!
//! A compiled sequence holds one slot per step for the whole of a pass, so a long chain kept
//! every intermediate tensor alive until the pass returned even though each was read once.
//! `last_use` says, for each slot, the index of the last step that reads it; the passes release
//! the slot there.
//!
//! Only *operation* slots are released. The leaf slots -- constant, parameter, given, sampler --
//! are read again after `tape.backward` by the cotangent collection, which needs a supplied
//! point's dims and device to shape the zero cotangent of a leaf that reached no target.

mod common;

use common::{Arithmetic, B, ConstantSampler, InertOperation, SumOperation, into_vec, points};
use skoro_core::id::Id;
use skoro_graph::compiler::CompiledSequence;
use skoro_graph::graph::Graph;
use skoro_graph::operation::DiffGraphOperation;
use skoro_operation::ops::{Addition, ScalarMultiplication};
use skoro_riemann::real::Real;

/// A slot read by exactly one step dies at that step.
#[test]
fn a_chain_releases_each_link_at_its_only_consumer() {
    // `p -> u -> v -> w`, each the sole input of the next.
    let mut graph = Graph::<B>::new();
    let p = graph.add_parameter(Real);
    let u = chain(&mut graph, p);
    let v = chain(&mut graph, u);
    let w = chain(&mut graph, v);

    let seq = graph.compile_backward_from(&[w]).unwrap();
    let last = seq.last_use();

    // Slots are in step order: p, u, v, w.
    assert_eq!(ids(&seq), vec![p, u, v, w]);
    assert_eq!(last[1], 2, "u is read by v and never again");
    assert_eq!(last[2], 3, "v is read by w and never again");
}

/// A slot read by several steps dies at the last of them, not the first.
#[test]
fn a_shared_slot_lives_until_its_last_reader() {
    let m = Arithmetic::new();

    // `[sum, prod, sq]`: `sum` feeds `sq`, and `a` feeds `sum` and `prod` both.
    let seq = m.graph.compile_backward_from(&[m.sq, m.prod]).unwrap();
    let last = seq.last_use();
    let slot = |id: Id| ids(&seq).iter().position(|x| *x == id).unwrap();

    let a_readers: Vec<usize> = [m.sum, m.prod].iter().map(|id| slot(*id)).collect();
    assert_eq!(
        last[slot(m.a)],
        *a_readers.iter().max().unwrap(),
        "`a` feeds both `sum` and `prod`, so it lives until the later one"
    );
}

/// A slot nothing reads dies at the step that produced it.
///
/// A target is the usual case: its value is cloned into `evaluations` as the step runs, and
/// nothing in the sequence below it ever looks at the slot again.
#[test]
fn an_unread_slot_dies_where_it_is_born() {
    let m = Arithmetic::new();

    let seq = m.graph.compile_backward_from(&[m.sq]).unwrap();
    let last = seq.last_use();
    let target = ids(&seq).iter().position(|id| *id == m.sq).unwrap();

    assert_eq!(last[target], target);
}

/// A sampler's inputs count as reads.
///
/// `input_indices` is on `DynSequenceSampler` as well as `DynSequenceOperation`, and a sampler
/// step reads the point sequence when it draws. A liveness pass that only walked operations
/// would free a slot the sampler still needs.
#[test]
fn a_sampler_keeps_its_parents_alive() {
    let mut graph = Graph::<B>::new();
    let p = graph.add_parameter(Real);
    let loc = graph
        .add_mapped(Real, DiffGraphOperation(Addition), vec![p, p])
        .unwrap();
    let z = graph
        .add_stochastic(Real, ConstantSampler(1.0), vec![loc])
        .unwrap();

    let seq = graph.compile_backward_from(&[z]).unwrap();
    let last = seq.last_use();
    let slot = |id: Id| ids(&seq).iter().position(|x| *x == id).unwrap();

    assert_eq!(
        last[slot(loc)],
        slot(z),
        "the sampler reads `loc`, so `loc` lives until the sampler runs"
    );
}

/// Releasing slots does not change any answer.
///
/// The whole suite is the real guard against over-releasing -- a slot freed too early is an
/// `unwrap` on a `None` -- but a chain long enough to release most of itself is worth having
/// stated where the liveness lives.
#[test]
fn a_long_chain_still_evaluates_correctly() {
    let mut graph = Graph::<B>::new();
    let p = graph.add_parameter(Real);
    let mut tip = p;
    for _ in 0..12 {
        tip = chain(&mut graph, tip);
    }

    // Each link doubles: 2^12 = 4096 at `p = 1`.
    let out = graph.evaluate(&points([(p, 1.0)]), tip).unwrap();
    assert_eq!(into_vec(out), vec![4096.0]);
}

/// A leaf's slot survives the backward sweep.
///
/// `vjp_forward` reads a supplied point's dims and device *after* `tape.backward`, to shape the
/// zero cotangent of a leaf that reaches no target. Releasing leaf slots would panic there --
/// and only for a graph that has such a leaf, which is the case this pins.
#[test]
fn a_leaf_that_reaches_no_target_still_gets_its_zero_cotangent() {
    let m = Arithmetic::new();
    let mut graph = m.graph;
    let spectator = graph.add_parameter(Real);

    let out = graph
        .evaluate_ids_vjp(
            &points([(m.a, 3.0), (m.b, 5.0), (spectator, 1.0)]),
            &[(m.sum, common::tensor3(vec![1.0], [1, 1, 1]))]
                .into_iter()
                .collect(),
            &[m.sum, spectator],
        )
        .unwrap();

    assert_eq!(
        into_vec(out.cotangents.get(&spectator).unwrap().clone()),
        vec![0.0],
        "a leaf off every path has zero derivative, and that is an answer"
    );
}

// Helpers

fn chain(graph: &mut Graph<B>, parent: Id) -> Id {
    graph
        .add_mapped(Real, DiffGraphOperation(Addition), vec![parent, parent])
        .unwrap()
}

fn ids(seq: &CompiledSequence<B>) -> Vec<Id> {
    seq.steps.iter().map(|s| s.id()).collect()
}

const _: fn() = || {
    let _: Option<InertOperation> = None;
    let _: Option<SumOperation> = None;
    let _ = ScalarMultiplication;
};
