//! The `dyn` layer: kind-erased quantity types, node accessors, and the operation/sampler
//! adapters that let a graph store either kind uniformly.
//!
//! None of this computes anything. It is the plumbing that decides *which* computation runs, so
//! the assertions here are about discrimination -- does a discrete type refuse to hand out a
//! manifold, does `compare` reject a look-alike, does a cloned adapter still lower the same way.

mod common;

use common::*;
use skoro_backend::{Int, Tensor, TensorData};
use skoro_core::{
    discrete::{Boolean, Categorical, Integer},
    kind::{Continuous, Discrete, QuantityKind},
};
use skoro_graph::{
    compiler::{CompiledStep, ForwardOptions},
    graph::Graph,
    node::Node,
    operation::{DiffGraphOperation, DiscreteGraphOperation, DynGraphOperation},
    quantity::{DynQuantityType, QuantityType},
    sampler::{DynGraphSampler, GraphSampler},
};
use skoro_operation::Input;
use skoro_riemann::{euclidean::EuclideanD, real::Real};

fn erase_continuous<T: QuantityType<B, Continuous>>(
    t: T,
) -> std::sync::Arc<dyn DynQuantityType<B>> {
    t.into_dyn()
}

fn erase_discrete<T: QuantityType<B, Discrete>>(t: T) -> std::sync::Arc<dyn DynQuantityType<B>> {
    t.into_dyn()
}

/// `QuantityType` behind its erasure boundary: a continuous type reports a manifold and no
/// cardinality, a discrete one the reverse, and comparison discriminates by concrete type *and*
/// value rather than by the discriminant alone. Each kind unwraps its own `Input` variant and
/// refuses the other's.
mod quantity_types {
    use super::*;

    #[test]
    fn a_continuous_type_reports_its_manifold_and_no_cardinality() {
        let q = erase_continuous(EuclideanD(3));

        assert_eq!(q.kind(), QuantityKind::Continuous);
        assert_eq!(q.num_components(), 3);
        assert_eq!(q.as_manifold().unwrap().dimension(), 3);
        assert!(q.inverse_retraction_op().is_some());
        // A continuous quantity has no finite support, by definition rather than by accident.
        assert_eq!(q.cardinality(), None);
    }

    #[test]
    fn a_discrete_type_reports_its_cardinality_and_no_manifold() {
        for (q, cardinality, components) in [
            (erase_discrete(Boolean), Some(2), 1),
            (erase_discrete(Categorical(5)), Some(5), 1),
            (erase_discrete(Integer), None, 1),
        ] {
            assert_eq!(q.kind(), QuantityKind::Discrete);
            assert_eq!(q.num_components(), components);
            assert_eq!(q.cardinality(), cardinality);
            // No tangent space, so neither a manifold nor an inverse-retraction.
            assert!(q.as_manifold().is_none());
            assert!(q.inverse_retraction_op().is_none());
        }
    }

    /// `compare` is equality on the *concrete* type, so two erased types agree only when they
    /// would have agreed unerased.
    #[test]
    fn compare_discriminates_by_concrete_type_and_value() {
        let three = erase_continuous(EuclideanD(3));
        let also_three = erase_continuous(EuclideanD(3));
        let two = erase_continuous(EuclideanD(2));
        let real = erase_continuous(Real);

        assert!(three.compare(also_three.as_any()));
        assert!(
            !three.compare(two.as_any()),
            "same type, different dimension"
        );
        assert!(
            !real.compare(erase_continuous(EuclideanD(1)).as_any()),
            "same storage and dimension, different manifold"
        );

        // And across kinds.
        let boolean = erase_discrete(Boolean);
        assert!(boolean.compare(erase_discrete(Boolean).as_any()));
        assert!(!boolean.compare(erase_discrete(Categorical(2)).as_any()));
        assert!(!boolean.compare(real.as_any()));
    }

    /// `clone_arc` exists because `Arc::clone` needs a sized receiver and `&dyn` is not one.
    #[test]
    fn clone_arc_produces_an_equal_handle() {
        for q in [erase_continuous(Real), erase_discrete(Categorical(4))] {
            let copy = q.clone_arc();
            assert_eq!(copy.kind(), q.kind());
            assert_eq!(copy.num_components(), q.num_components());
            assert_eq!(copy.cardinality(), q.cardinality());
            assert!(copy.compare(q.as_any()));
        }
    }

    /// `get_inner_tensor` is the typed unwrap: each kind reads the `Input` variant that is its
    /// own, and the wrong variant is a caller bug rather than a recoverable error.
    ///
    /// This is the one place where the `Kind` type parameter stops being a phantom and picks a
    /// concrete tensor element type, so it is worth checking both directions rather than only
    /// the one a graph happens to take.
    #[test]
    fn each_kind_unwraps_its_own_input_variant() {
        let x = tensor2(vec![1.0, 2.0, 3.0], 1, 3);
        let unwrapped = QuantityType::<B, Continuous>::get_inner_tensor(
            &EuclideanD(3),
            Input::Float(x.clone()),
        );
        assert_eq!(into_vec(unwrapped), into_vec(x));

        let n = Tensor::<B, 2, Int>::from_data(
            TensorData::new(vec![2i32], [1, 1]),
            &Default::default(),
        );
        let unwrapped =
            QuantityType::<B, Discrete>::get_inner_tensor(&Categorical(4), Input::Int(n));
        assert_eq!(unwrapped.into_data().to_vec::<i32>().unwrap(), vec![2]);
    }

    #[test]
    #[should_panic]
    fn a_kind_will_not_unwrap_the_other_variant() {
        let n = Tensor::<B, 2, Int>::from_data(
            TensorData::new(vec![2i32], [1, 1]),
            &Default::default(),
        );
        let _ = QuantityType::<B, Continuous>::get_inner_tensor(&Real, Input::Int(n));
    }

    /// The erased types are `Debug`, which is what makes a graph dump readable.
    #[test]
    fn erased_types_are_debug() {
        assert!(format!("{:?}", erase_continuous(EuclideanD(3))).contains('3'));
        assert!(!format!("{:?}", erase_discrete(Boolean)).is_empty());
    }
}

/// The `Node` enum's common accessors, answered by every variant: its parents, its quantity type,
/// and which kind it holds. These are the questions the compiler asks without knowing which
/// variant it has, so a variant that answered one of them wrongly would mislower silently.
mod nodes {
    use super::*;

    /// Every variant answers `parent_ids`, `quantity_type` and the kind predicates -- including
    /// the two leaf kinds, whose parent list is empty rather than absent.
    #[test]
    fn every_variant_answers_the_common_accessors() {
        let mut graph = Graph::<B>::new();

        let constant = graph.add_constant(Real, &1.0f64);
        let parameter = graph.add_parameter(Real);
        let stochastic = graph
            .add_stochastic(Real, ConstantSampler(0.0), vec![parameter])
            .unwrap();
        let observed = graph
            .add_stochastic(Real, ConstantSampler(0.0), vec![parameter])
            .unwrap();
        graph.observe::<_, Real, _>(observed, &2.0f64).unwrap();
        let mapped = graph
            .add_mapped(Real, DiffGraphOperation(InertOperation), vec![stochastic])
            .unwrap();
        let factor = graph
            .add_mapped(Real, DiffGraphOperation(InertOperation), vec![stochastic])
            .unwrap();
        graph.mark_log_likelihood(factor).unwrap();

        // Leaves have no parents; everything else names the node it was built from.
        assert!(graph.node(constant).unwrap().parent_ids().is_empty());
        assert!(graph.node(parameter).unwrap().parent_ids().is_empty());
        assert_eq!(graph.node(stochastic).unwrap().parent_ids(), &[parameter]);
        assert_eq!(graph.node(observed).unwrap().parent_ids(), &[parameter]);
        assert_eq!(graph.node(mapped).unwrap().parent_ids(), &[stochastic]);
        assert_eq!(graph.node(factor).unwrap().parent_ids(), &[stochastic]);

        for id in [constant, parameter, stochastic, observed, mapped, factor] {
            let node = graph.node(id).unwrap();
            assert_eq!(node.quantity_type().num_components(), 1);
            assert!(node.is_continuous());
            assert!(!node.is_discrete());
        }
    }

    #[test]
    fn discrete_nodes_report_the_other_kind() {
        let mut graph = Graph::<B>::new();
        let c = graph.add_constant(Categorical(3), &1usize);
        let p = graph.add_parameter(Integer);
        let s = graph
            .add_stochastic(Boolean, ConstantIntSampler(1), vec![])
            .unwrap();

        for id in [c, p, s] {
            let node = graph.node(id).unwrap();
            assert!(node.is_discrete());
            assert!(!node.is_continuous());
            assert_eq!(node.quantity_type().kind(), QuantityKind::Discrete);
        }
    }

    /// The two sampler-bearing nodes print without their samplers, which have no useful `Debug`.
    #[test]
    fn sampler_bearing_nodes_are_debug() {
        let mut graph = Graph::<B>::new();
        let p = graph.add_parameter(Real);
        let s = graph
            .add_stochastic(Real, ConstantSampler(0.0), vec![p])
            .unwrap();

        let Some(Node::Stochastic(node)) = graph.node(s) else {
            panic!("expected a stochastic node");
        };
        let text = format!("{node:?}");
        assert!(text.contains("StochasticNode"));
        assert!(text.contains("parent_ids"));

        graph.observe::<_, Real, _>(s, &1.0f64).unwrap();
        let Some(Node::Observed(node)) = graph.node(s) else {
            panic!("expected an observed node");
        };
        assert!(format!("{node:?}").contains("ObservedNode"));
    }

    /// A node is cheap to clone, which is what lets `remove` hand the caller its contents back.
    #[test]
    fn nodes_clone() {
        let mut graph = Graph::<B>::new();
        let p = graph.add_parameter(Real);
        let node = graph.node(p).unwrap().clone();
        assert!(node.parent_ids().is_empty());
    }
}

/// The boxed adapters that carry an operation or a sampler into a compiled sequence: which
/// adapter differentiability chose, cloning through the box, the name they print, and where
/// lowering positions them. An unrealized parent is a sequencing bug, so it panics.
mod adapter {
    use super::*;

    /// The two operation adapters differ in exactly one externally visible way, and it is the
    /// one the compiler branches on.
    #[test]
    fn differentiability_is_the_adapter_that_was_chosen() {
        let diff = DiffGraphOperation(InertOperation);
        let discrete = DiscreteGraphOperation(RoundToInt);

        assert!(DynGraphOperation::<B>::is_differentiable(&diff));
        assert!(!DynGraphOperation::<B>::is_differentiable(&discrete));
    }

    #[test]
    fn adapters_clone_through_the_box() {
        let boxed: Box<dyn DynGraphOperation<B>> = Box::new(DiffGraphOperation(InertOperation));
        let copy = boxed.clone();
        assert!(copy.is_differentiable());
        assert!(copy.clone_box().is_differentiable());

        let boxed: Box<dyn DynGraphOperation<B>> = Box::new(DiscreteGraphOperation(RoundToInt));
        let copy = boxed.clone();
        assert!(!copy.is_differentiable());
        assert!(!copy.clone_box().is_differentiable());
    }

    #[test]
    fn adapters_name_the_operation_they_wrap_when_printed() {
        let diff = format!("{:?}", DiffGraphOperation(InertOperation));
        assert!(diff.contains("DiffGraphOperation"));
        assert!(diff.contains("InertOperation"));

        let discrete = format!("{:?}", DiscreteGraphOperation(RoundToInt));
        assert!(discrete.contains("DiscreteGraphOperation"));
        assert!(discrete.contains("RoundToInt"));
    }

    /// Lowering positions the operation in a sequence; both adapters accept the same call and
    /// produce a step that reports the indices back.
    #[test]
    fn lowering_positions_the_operation_in_a_sequence() {
        let diff: Box<dyn DynGraphOperation<B>> = Box::new(DiffGraphOperation(InertOperation));
        let lowered = diff.lower(vec![0, 1], vec![true, false], 2);
        assert_eq!(lowered.input_indices(), &[0, 1]);
        assert_eq!(lowered.output_index(), 2);
        assert_eq!(lowered.input_diffs(), &[true, false]);

        // A discrete operation ignores the diff flags: none of its inputs are differentiable.
        let discrete: Box<dyn DynGraphOperation<B>> = Box::new(DiscreteGraphOperation(RoundToInt));
        let lowered = discrete.lower(vec![5], vec![true], 6);
        assert_eq!(lowered.input_indices(), &[5]);
        assert_eq!(lowered.output_index(), 6);
        assert_eq!(lowered.input_diffs(), &[false]);
    }

    #[test]
    fn samplers_clone_and_lower_through_the_box() {
        let boxed: Box<dyn DynGraphSampler<B>> =
            Box::new(GraphSampler::<B, Continuous, _>::new(ConstantSampler(1.5)));
        assert!(format!("{boxed:?}").contains("sampler"));

        let copy = boxed.clone();
        let lowered = copy.lower(vec![0], 1);
        assert_eq!(lowered.input_indices(), &[0]);
        assert_eq!(lowered.output_index(), 1);

        // And it still samples what it was built with, once its parent slot is filled.
        let mut seq = vec![Some(skoro_operation::Input::Float(scalar(0.0))), None];
        lowered.sample(2, &mut seq);
        assert_eq!(
            into_vec(seq[1].take().unwrap().unwrap_float()),
            vec![1.5, 1.5]
        );
    }

    /// A sampler whose parent has not been realized yet is a sequencing bug, and says so.
    ///
    /// The compiler emits steps in topological order precisely so this cannot happen; reaching
    /// it means the ordering was wrong, which is an internal invariant rather than a user error.
    #[test]
    #[should_panic(expected = "requires parent at (relative) input index 0")]
    fn a_sampler_whose_parent_is_unrealized_panics() {
        let sampler = GraphSampler::<B, Continuous, _>::new(ConstantSampler(1.5));
        let lowered = DynGraphSampler::lower(&sampler, vec![0], 1);
        let mut seq = vec![None, None];
        lowered.sample(1, &mut seq);
    }

    #[test]
    fn discrete_samplers_lower_too() {
        let sampler = GraphSampler::<B, Discrete, _>::new(ConstantIntSampler(7));
        let lowered = sampler.clone_box().lower(vec![], 0);
        let mut seq = vec![None];
        lowered.sample(1, &mut seq);
        assert_eq!(
            seq[0]
                .take()
                .unwrap()
                .unwrap_int()
                .into_data()
                .to_vec::<i32>()
                .unwrap(),
            vec![7]
        );
    }
}

/// A `CompiledSequence` as a finished artifact: the steps it reports, and that every node kind
/// lowered to its own step. An observed quantity lowers to a `Constant` -- the point of observing
/// it -- and forward options default to a single unsampled batch.
mod compiled {
    use super::*;

    /// The compiled sequence prints its steps, and each step knows which node it evaluates.
    #[test]
    fn a_compiled_sequence_reports_its_steps() {
        let m = Arithmetic::new();
        let sequence = m.graph.compile_backward_from(&[m.sq]).unwrap();

        // `a`, `b`, `sum`, `sq` -- `prod` is outside `sq`'s cone.
        assert_eq!(sequence.steps.len(), 4);
        let ids: Vec<_> = sequence.steps.iter().map(CompiledStep::id).collect();
        assert_eq!(ids, vec![m.a, m.b, m.sum, m.sq]);

        assert!(matches!(sequence.steps[0], CompiledStep::Parameter { .. }));
        assert!(matches!(sequence.steps[2], CompiledStep::Operation { .. }));
        assert!(format!("{sequence:?}").contains("CompiledSequence"));
    }

    /// Constants, samplers and operations all reach their own step kind.
    #[test]
    fn every_node_kind_lowers_to_its_own_step() {
        let mut graph = Graph::<B>::new();
        let c = graph.add_constant(Real, &1.0f64);
        let p = graph.add_parameter(Real);
        let s = graph
            .add_stochastic(Real, ConstantSampler(0.0), vec![p])
            .unwrap();
        let m = graph
            .add_mapped(
                Real,
                DiffGraphOperation(skoro_operation::ops::Addition),
                vec![c, s],
            )
            .unwrap();

        let sequence = graph.compile_backward_from(&[m]).unwrap();
        let kinds: Vec<&str> = sequence
            .steps
            .iter()
            .map(|step| match step {
                CompiledStep::Constant { .. } => "constant",
                CompiledStep::Parameter { .. } => "parameter",
                CompiledStep::Sampler { .. } => "sampler",
                CompiledStep::Given { .. } => "given",
                CompiledStep::Operation { .. } => "operation",
            })
            .collect();
        assert_eq!(kinds, vec!["constant", "parameter", "sampler", "operation"]);
    }

    /// An observed quantity lowers to a *constant* step -- there is no separate observation
    /// kind, which is why forward output reports observations among the constants.
    #[test]
    fn an_observed_quantity_lowers_to_a_constant() {
        let mut graph = Graph::<B>::new();
        let s = graph
            .add_stochastic(Real, ConstantSampler(0.0), vec![])
            .unwrap();
        graph.observe::<_, Real, _>(s, &3.0f64).unwrap();

        let sequence = graph.compile_backward_from(&[s]).unwrap();
        assert!(matches!(sequence.steps[0], CompiledStep::Constant { .. }));

        let out = graph.generate(1).unwrap();
        assert!(out.constants.get(&s).is_some());
        assert!(out.variables.get(&s).is_none());
    }

    #[test]
    fn forward_options_default_to_a_single_unsampled_batch() {
        let options = ForwardOptions::default();
        assert_eq!(options.batch_size, 1);
        assert!(!options.use_model_samplers);
        assert!(format!("{options:?}").contains("batch_size"));
    }
}
