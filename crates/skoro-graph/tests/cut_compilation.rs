//! Cut compilation: a program that stops at a frontier of nodes whose values the caller
//! supplies, rather than walking back to every ancestor.
//!
//! The whole of this feature is one step kind. A cut boundary that lowered to
//! `CompiledStep::Constant` would make `is_differentiable_parent` report `false` for it, which
//! sets `constant_inputs` on every consumer, which keeps those consumers off the tape -- and
//! the pass would then return correct values with silently absent derivatives. `Given` is a
//! *differentiable* leaf: supplied like a constant, differentiated like a parameter.

mod common;

use std::sync::Arc;

use common::{Arithmetic, B, ConstantSampler, InertOperation, into_vec, points, scalar, tensor3};
use skoro_core::error::SkoroError;
use skoro_core::id::Id;
use skoro_graph::compiler::{
    CompiledSequence, CompiledStep, ForwardInput, ForwardOptions, HvpForwardInput, VjpForwardInput,
};
use skoro_graph::graph::Graph;
use skoro_graph::operation::DiffGraphOperation;
use skoro_operation::ops::{Addition, ScalarMultiplication};
use skoro_riemann::real::Real;

// Fixtures

/// `k1`, `k2` constant; `s = k1 + k2`; `y = s * s`. Every node folds at build time.
///
/// The configuration where a cut matters most: uncut, `y` has no differentiable input at all,
/// so lowering the cut as a `Constant` and lowering it as a `Given` disagree about whether `y`
/// reaches the tape.
struct AllFolded {
    graph: Graph<B>,
    s: Id,
    y: Id,
}

impl AllFolded {
    fn new() -> Self {
        let mut graph = Graph::<B>::new();
        let k1 = graph.add_constant(Real, &2.0f64);
        let k2 = graph.add_constant(Real, &3.0f64);
        let s = graph
            .add_mapped(Real, DiffGraphOperation(Addition), vec![k1, k2])
            .unwrap();
        let y = graph
            .add_mapped(Real, DiffGraphOperation(ScalarMultiplication), vec![s, s])
            .unwrap();
        Self { graph, s, y }
    }
}

// The trap

/// A cut node lowers to `Given`, not to `Constant`.
#[test]
fn a_cut_node_lowers_to_a_differentiable_leaf() {
    let m = AllFolded::new();

    let seq = m.graph.compile_between(&[m.s], &[m.y]).unwrap();

    assert!(
        matches!(step_for(&seq, m.s), CompiledStep::Given { .. }),
        "the cut boundary is supplied by the caller, but it is still a free variable"
    );
}

/// Cutting above a fold un-folds what is below it: the consumer regains a differentiable input.
///
/// This is the assertion the feature exists for. `y = s * s` has `constant_inputs: true` uncut,
/// because `s` carries a fold and folded mapped nodes are not differentiable inputs. Cut at
/// `s`, `y` must reach the tape -- and if it does not, the VJP below returns no cotangent while
/// the forward value stays perfectly correct.
#[test]
fn cutting_clears_constant_inputs_on_the_consumer() {
    let m = AllFolded::new();

    let uncut = m.graph.compile_backward_from(&[m.y]).unwrap();
    assert!(
        constant_inputs(&uncut, m.y),
        "fixture must really be all-folded, or this test is vacuous"
    );

    let cut = m.graph.compile_between(&[m.s], &[m.y]).unwrap();
    assert!(!constant_inputs(&cut, m.y));
}

/// And the derivative actually arrives. `y = s²`, so `∂y/∂s = 2s = 10` at `s = 5`.
#[test]
fn a_vjp_through_a_cut_reaches_the_cut_boundary() {
    let m = AllFolded::new();
    let seq = m.graph.compile_between(&[m.s], &[m.y]).unwrap();

    let pts = points([(m.s, 5.0)]);
    let out = seq
        .vjp_forward(VjpForwardInput {
            points: &pts,
            cotangents: &[(m.y, tensor3(vec![1.0], [1, 1, 1]))].into_iter().collect(),
            options: ForwardOptions {
                batch_size: 1,
                use_model_samplers: false,
            },
        })
        .unwrap();

    let cotangent = out
        .cotangents
        .get(&m.s)
        .expect("the cut boundary is a leaf, so it collects a cotangent");
    assert_eq!(into_vec(cotangent.clone()), vec![10.0]);
}

// Composition: the reason to cut at all

/// A cut splits one reverse pass into two that compose.
///
/// `sq = (a + b)²`. Differentiating it in one pass, and differentiating the upper half to the
/// cut then pushing that cotangent through the lower half, must agree. This is the property
/// that a silently dropped term breaks and that finite differences cannot catch -- a cut which
/// drops a term is self-consistently wrong.
#[test]
fn a_cut_pass_composes_with_the_pass_below_it() {
    let m = Arithmetic::new();
    let pts = points([(m.a, 3.0), (m.b, 5.0)]);
    let seed = |v: f64| [(m.sq, tensor3(vec![v], [1, 1, 1]))].into_iter().collect();

    // One pass, all the way down.
    let whole = m
        .graph
        .compile_backward_from(&[m.sq])
        .unwrap()
        .vjp_forward(VjpForwardInput {
            points: &pts,
            cotangents: &seed(1.0),
            options: opts(),
        })
        .unwrap();
    let direct = into_vec(whole.cotangents.get(&m.a).unwrap().clone());

    // Upper half: `sq` down to the cut at `sum`.
    let sum_value = into_vec(m.graph.evaluate(&pts, m.sum).unwrap())[0];
    let upper = m
        .graph
        .compile_between(&[m.sum], &[m.sq])
        .unwrap()
        .vjp_forward(VjpForwardInput {
            points: &points([(m.sum, sum_value)]),
            cotangents: &seed(1.0),
            options: opts(),
        })
        .unwrap();
    let at_cut = into_vec(upper.cotangents.get(&m.sum).unwrap().clone())[0];

    // Lower half: `sum` down to the leaves, seeded with what came out of the upper half.
    let lower = m
        .graph
        .compile_backward_from(&[m.sum])
        .unwrap()
        .vjp_forward(VjpForwardInput {
            points: &pts,
            cotangents: &[(m.sum, tensor3(vec![at_cut], [1, 1, 1]))]
                .into_iter()
                .collect(),
            options: opts(),
        })
        .unwrap();
    let composed = into_vec(lower.cotangents.get(&m.a).unwrap().clone());

    assert_eq!(direct, vec![16.0], "d(a+b)²/da = 2(a+b) = 16");
    assert_eq!(composed, direct);
}

// What a cut prunes, and what it leaves alone

/// A cut severs the traversal: the cut node's ancestors are not compiled.
#[test]
fn a_cut_prunes_everything_above_it() {
    let m = Arithmetic::new();

    let seq = m.graph.compile_between(&[m.sum], &[m.sq]).unwrap();

    let ids: Vec<Id> = seq.steps.iter().map(|s| s.id()).collect();
    assert_eq!(ids, vec![m.sum, m.sq]);
    assert!(!ids.contains(&m.a) && !ids.contains(&m.b));
}

/// Cutting at a node that is already a leaf changes nothing.
///
/// A parameter has no parents to sever and already reads its value from `points`, so the cut
/// has nothing to do. Keeping it a `Parameter` is what makes "cut at a leaf is a no-op" true
/// rather than merely usually true.
#[test]
fn cutting_at_a_parameter_is_a_no_op() {
    let m = Arithmetic::new();

    let uncut = m.graph.compile_backward_from(&[m.sum]).unwrap();
    let cut = m.graph.compile_between(&[m.a], &[m.sum]).unwrap();

    assert_eq!(kinds(&uncut), kinds(&cut));
    assert!(matches!(
        step_for(&cut, m.a),
        CompiledStep::Parameter { .. }
    ));
}

/// A cut at a node that no target reaches is ignored, not emitted.
#[test]
fn a_cut_off_every_path_is_ignored() {
    let m = Arithmetic::new();

    let seq = m.graph.compile_between(&[m.prod], &[m.sum]).unwrap();

    let ids: Vec<Id> = seq.steps.iter().map(|s| s.id()).collect();
    assert!(!ids.contains(&m.prod));
    assert_eq!(ids, vec![m.a, m.b, m.sum]);
}

/// Cutting a stochastic node replaces its sampler step.
///
/// Not cosmetic: a `Sampler` step holds its parents' slot indices, and the cut has just pruned
/// those slots. Leaving it a `Sampler` would index past the end of the sequence.
#[test]
fn cutting_a_stochastic_node_replaces_its_sampler() {
    let mut graph = Graph::<B>::new();
    let loc = graph.add_parameter(Real);
    let z = graph
        .add_stochastic(Real, ConstantSampler(1.0), vec![loc])
        .unwrap();
    let y = graph
        .add_mapped(Real, DiffGraphOperation(ScalarMultiplication), vec![z, z])
        .unwrap();

    let seq = graph.compile_between(&[z], &[y]).unwrap();

    assert!(matches!(step_for(&seq, z), CompiledStep::Given { .. }));
    let ids: Vec<Id> = seq.steps.iter().map(|s| s.id()).collect();
    assert_eq!(ids, vec![z, y], "the sampler's parent must be pruned too");
}

// The supplied value

/// A cut boundary is a free variable, so its value is reported as one.
///
/// `ForwardOutput::variables` is what `jacobian::fill_leaf_rows` reads to decide that a row is
/// also a column and deserves an identity block. For a cut node that is the right answer:
/// it *is* a free input of the compiled program, so `∂x/∂x = I`.
#[test]
fn a_cut_value_is_reported_as_a_variable() {
    let m = AllFolded::new();
    let seq = m.graph.compile_between(&[m.s], &[m.y]).unwrap();

    let pts = points([(m.s, 5.0)]);
    let out = seq
        .forward(ForwardInput {
            points: &pts,
            options: opts(),
        })
        .unwrap();

    assert!(out.variables.get(&m.s).is_some());
    assert!(out.constants.get(&m.s).is_none());
    let (value, _) = out.evaluations.get(&m.y).unwrap();
    assert_eq!(into_vec(value.clone().unwrap_float()), vec![25.0]);
}

/// A cut boundary with no supplied value is a caller error with a name.
#[test]
fn a_cut_without_a_value_is_missing_point() {
    let m = AllFolded::new();
    let seq = m.graph.compile_between(&[m.s], &[m.y]).unwrap();

    let err = seq
        .forward(ForwardInput {
            points: &points([]),
            options: opts(),
        })
        .unwrap_err();

    assert!(matches!(err, SkoroError::MissingPoint { id } if id == m.s));
}

// Caching

/// Cut programs are cached like any other, and cut *order* is not part of the program.
///
/// Targets seed the traversal, so their order fixes every slot index; cuts only terminate it,
/// so only their membership matters. The key canonicalizes one and not the other.
#[test]
fn cut_order_is_not_part_of_the_key() {
    let mut graph = Graph::<B>::new();
    let p = graph.add_parameter(Real);
    let u = graph
        .add_mapped(Real, DiffGraphOperation(Addition), vec![p, p])
        .unwrap();
    let v = graph
        .add_mapped(Real, DiffGraphOperation(Addition), vec![p, p])
        .unwrap();
    let y = graph
        .add_mapped(Real, DiffGraphOperation(Addition), vec![u, v])
        .unwrap();

    let a = graph.compile_between(&[u, v], &[y]).unwrap();
    let b = graph.compile_between(&[v, u], &[y]).unwrap();
    let c = graph.compile_between(&[u, u, v], &[y]).unwrap();

    assert!(Arc::ptr_eq(&a, &b), "cut order is not significant");
    assert!(Arc::ptr_eq(&a, &c), "a repeated cut is the same cut");
}

/// An uncut compile and a cut compile of the same targets are different programs.
#[test]
fn a_cut_compile_does_not_collide_with_an_uncut_one() {
    let m = Arithmetic::new();

    let uncut = m.graph.compile_backward_from(&[m.sq]).unwrap();
    let cut = m.graph.compile_between(&[m.sum], &[m.sq]).unwrap();

    assert!(!Arc::ptr_eq(&uncut, &cut));
}

/// `compile_backward_from` is the no-cut case, and shares its entry.
#[test]
fn compiling_with_no_cuts_is_compiling_backward() {
    let m = Arithmetic::new();

    let backward = m.graph.compile_backward_from(&[m.sq]).unwrap();
    let empty_cut = m.graph.compile_between(&[], &[m.sq]).unwrap();

    assert!(Arc::ptr_eq(&backward, &empty_cut));
}

/// A cut naming a node that does not exist is an error, like a target that does not.
#[test]
fn a_missing_cut_is_an_error() {
    let mut m = Arithmetic::new();
    let gone = m.graph.add_parameter(Real);
    m.graph.remove(&[gone]).unwrap();

    let err = m.graph.compile_between(&[gone], &[m.sq]).unwrap_err();

    assert!(matches!(err, SkoroError::NodeNotFound { id } if id == gone));
}

// support

/// `support` is every node the given one can reach, itself included.
#[test]
fn support_is_the_forward_closure() {
    let m = Arithmetic::new();

    assert_eq!(
        sorted(m.graph.support(m.a)),
        sorted(vec![m.a, m.sum, m.prod, m.sq])
    );
    assert_eq!(sorted(m.graph.support(m.sum)), sorted(vec![m.sum, m.sq]));
    assert_eq!(m.graph.support(m.sq), vec![m.sq]);
}

/// And it is deterministic.
///
/// `children` is a `HashSet`, so a forward walk gets none of the ordering that
/// `topological_sort` gets free from `parent_ids` being a `Vec`.
#[test]
fn support_is_deterministic() {
    let m = Arithmetic::new();

    let first = m.graph.support(m.a);
    for _ in 0..16 {
        assert_eq!(m.graph.support(m.a), first);
    }
}

/// An unknown id supports nothing.
#[test]
fn support_of_a_missing_node_is_empty() {
    let mut m = Arithmetic::new();
    let gone = m.graph.add_parameter(Real);
    m.graph.remove(&[gone]).unwrap();

    assert!(m.graph.support(gone).is_empty());
}

// Helpers

fn opts() -> ForwardOptions {
    ForwardOptions {
        batch_size: 1,
        use_model_samplers: false,
    }
}

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

/// The step kinds in order, as a comparable summary of a sequence's shape.
fn kinds(seq: &CompiledSequence<B>) -> Vec<&'static str> {
    seq.steps
        .iter()
        .map(|s| match s {
            CompiledStep::Constant { .. } => "constant",
            CompiledStep::Parameter { .. } => "parameter",
            CompiledStep::Sampler { .. } => "sampler",
            CompiledStep::Given { .. } => "given",
            CompiledStep::Operation { .. } => "operation",
        })
        .collect()
}

fn sorted(mut ids: Vec<Id>) -> Vec<Id> {
    ids.sort();
    ids
}

// Referenced only by fixtures above; keeps the import list honest.
const _: fn() = || {
    let _: Option<InertOperation> = None;
};

// Second order, and the numerical reference

/// Cutting at a node gives the same derivatives as replacing it with a free variable.
///
/// This is the statement that makes a cut *mean* something, and it is how the numerical
/// harness reaches this feature at all: `test_hvp_numerically` and friends drive
/// `Graph::evaluate_*`, which compile uncut, so they cannot be pointed at a cut program
/// directly. They can be pointed at the graph on the right-hand side of this equality, which
/// the test below does -- so the cut inherits a finite-difference check through it.
///
/// Both orders are compared. A cut that reached the tape for the first derivative but dropped
/// the second would pass every test above this one.
#[test]
fn a_cut_matches_the_same_graph_with_a_free_node_in_its_place() {
    // `c` is mapped, and cut.
    let mut cut_graph = Graph::<B>::new();
    let p = cut_graph.add_parameter(Real);
    let c_mapped = cut_graph
        .add_mapped(Real, DiffGraphOperation(Addition), vec![p, p])
        .unwrap();
    let y_cut = cut_graph
        .add_mapped(
            Real,
            DiffGraphOperation(ScalarMultiplication),
            vec![c_mapped, c_mapped],
        )
        .unwrap();

    // `c` is free, and nothing is cut.
    let mut free_graph = Graph::<B>::new();
    let c_free = free_graph.add_parameter(Real);
    let y_free = free_graph
        .add_mapped(
            Real,
            DiffGraphOperation(ScalarMultiplication),
            vec![c_free, c_free],
        )
        .unwrap();

    let at = 5.0;
    let ord1_seed = |id: Id| [(id, tensor3(vec![1.0], [1, 1, 1]))].into_iter().collect();
    let ord2_seed = |id: Id| [(id, tensor3(vec![0.0], [1, 1, 1]))].into_iter().collect();

    // First order.
    let cut_vjp = cut_graph
        .compile_between(&[c_mapped], &[y_cut])
        .unwrap()
        .vjp_forward(VjpForwardInput {
            points: &points([(c_mapped, at)]),
            cotangents: &ord1_seed(y_cut),
            options: opts(),
        })
        .unwrap();
    let free_vjp = free_graph
        .evaluate_vjp(&points([(c_free, at)]), &ord1_seed(y_free), y_free)
        .unwrap();
    assert_eq!(
        into_vec(cut_vjp.cotangents.get(&c_mapped).unwrap().clone()),
        into_vec(free_vjp.cotangents.get(&c_free).unwrap().clone()),
    );

    // Second order.
    let cut_hvp = cut_graph
        .compile_between(&[c_mapped], &[y_cut])
        .unwrap()
        .hvp_forward(HvpForwardInput {
            points: &points([(c_mapped, at)]),
            tangents: &[(c_mapped, scalar(1.0))].into_iter().collect(),
            ord1_cotangents: &ord1_seed(y_cut),
            ord2_cotangents: &ord2_seed(y_cut),
            options: opts(),
        })
        .unwrap();
    let free_hvp = free_graph
        .evaluate_hvp(
            &points([(c_free, at)]),
            &[(c_free, scalar(1.0))].into_iter().collect(),
            &ord1_seed(y_free),
            &ord2_seed(y_free),
            y_free,
        )
        .unwrap();
    assert_eq!(
        into_vec(cut_hvp.ord2_cotangents.get(&c_mapped).unwrap().clone()),
        into_vec(free_hvp.ord2_cotangents.get(&c_free).unwrap().clone()),
    );

    // And the right-hand side is itself checked against forward differences, so the equality
    // above is anchored to something other than another compile of the same tree.
    skoro_graph::test_utils::test_vjp_numerically(
        &free_graph,
        c_free,
        y_free,
        points([(c_free, at)]),
        common::DELTA,
        common::EPSILON,
    );
    skoro_graph::test_utils::test_hvp_numerically(
        &free_graph,
        c_free,
        y_free,
        points([(c_free, at)]),
        common::DELTA,
        common::EPSILON,
    );
}

/// A valid cut does not excuse a stale target.
///
/// The two validations are separate layers: `compile_between` checks the cuts up front, and the
/// lowering loop checks each visited node as it arrives. A cut that exists gets past the first
/// and must not mask the second.
#[test]
fn a_valid_cut_does_not_excuse_a_missing_target() {
    let mut m = Arithmetic::new();
    let gone = m.graph.add_parameter(Real);
    m.graph.remove(&[gone]).unwrap();

    let err = m.graph.compile_between(&[m.sum], &[gone]).unwrap_err();

    assert!(matches!(err, SkoroError::NodeNotFound { id } if id == gone));
}

/// Cut programs share the cap with uncut ones.
///
/// They share one map, so they share the bound. A cut program is no cheaper to hold than an
/// uncut one -- it holds the same boxed operations, just fewer of them.
#[test]
fn cut_programs_are_bounded_too() {
    let mut graph = Graph::<B>::new();
    let root = graph.add_parameter(Real);
    let mut ids = vec![root];
    for _ in 0..skoro_graph::graph::MAX_CACHED_SEQUENCES {
        let next = *ids.last().unwrap();
        ids.push(
            graph
                .add_mapped(Real, DiffGraphOperation(Addition), vec![next, next])
                .unwrap(),
        );
    }
    let tip = *ids.last().unwrap();

    // Each distinct cut is a distinct program over the same targets.
    let first = graph.compile_between(&[ids[0]], &[tip]).unwrap();
    for id in &ids[1..] {
        graph.compile_between(&[*id], &[tip]).unwrap();
    }

    let first_again = graph.compile_between(&[ids[0]], &[tip]).unwrap();
    assert!(
        !Arc::ptr_eq(&first, &first_again),
        "cut programs count against the same cap as uncut ones"
    );
}
