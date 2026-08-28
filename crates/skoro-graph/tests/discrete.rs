//! Discrete quantities in the graph: integer-valued, non-differentiable, and carried through the
//! same kind-erased machinery as everything else.
//!
//! The point of these tests is that "discrete" is a runtime fact the graph has to keep straight,
//! not a compile-time one it can forget. A discrete node must evaluate, must surface its value,
//! and must be routed around by every differentiation pass without anyone downstream having to
//! ask.

mod common;

use common::*;
use skoro_core::discrete::{Boolean, Categorical, Integer};
use skoro_graph::{
    graph::Graph,
    node::Node,
    operation::{DiffGraphOperation, DiscreteGraphOperation},
};
use skoro_riemann::real::Real;

/// Discrete constants, parameters and stochastic quantities in the arena: added, surfaced through
/// the forward pass, batched, sampled and observed. "Discrete" is a runtime fact the graph has to
/// keep straight, so every one of these asserts the integer value came back as an integer.
mod nodes {
    use super::*;

    /// Ensure we can add discrete constants and parameters, and that these are known to be as such
    /// during runtime.
    #[test]
    fn add_discrete_constant_and_parameter() {
        let mut graph = Graph::<B>::new();

        // A categorical with 5 categories, fixed at index 3.
        let c = graph.add_constant(Categorical(5), &3usize);
        // A free boolean parameter.
        let p = graph.add_parameter(Boolean);

        assert!(graph.contains(c));
        assert!(graph.contains(p));

        // Both nodes report as discrete.
        assert!(graph.node(c).unwrap().is_discrete());
        assert!(graph.node(p).unwrap().is_discrete());

        // The stored variants are the unified constant/parameter ones.
        assert!(matches!(graph.node(c).unwrap(), Node::Constant(_)));
        assert!(matches!(graph.node(p).unwrap(), Node::Parameter(_)));
    }

    /// Ensure discrete constants get stuck into the appropriate value-set, and that they are
    /// discretely held.
    #[test]
    fn discrete_constant_surfaces_in_forward_output() {
        let mut graph = Graph::<B>::new();
        let c = graph.add_constant(Categorical(4), &2usize);

        // Generating the model should surface the discrete value (and not panic in
        // the differentiation-free forward path).
        let out = graph.generate(1).unwrap();
        let (value, _) = out
            .constants
            .get(&c)
            .expect("discrete constant should be surfaced in forward output");
        let data = value
            .clone()
            .unwrap_int()
            .into_data()
            .to_vec::<i32>()
            .unwrap();
        assert_eq!(data, vec![2]);
    }

    /// A batched discrete constant carries one value per row, like its continuous counterpart.
    #[test]
    fn discrete_constants_can_be_batched() {
        let mut graph = Graph::<B>::new();
        let c = graph
            .add_batched_constant(Categorical(4), &[0usize, 3, 1])
            .unwrap();
        assert_eq!(graph.pinned_batch(), Some(3));

        let out = graph.generate(3).unwrap();
        let (value, _) = out.constants.get(&c).unwrap();
        assert_eq!(
            value
                .clone()
                .unwrap_int()
                .into_data()
                .to_vec::<i32>()
                .unwrap(),
            vec![0, 3, 1]
        );
    }

    /// A discrete stochastic quantity samples integers, and observing one binds it to an integer.
    #[test]
    fn discrete_stochastic_quantities_sample_and_observe() {
        let mut graph = Graph::<B>::new();
        let z = graph
            .add_stochastic(Boolean, ConstantIntSampler(1), vec![])
            .unwrap();

        let out = graph.generate(2).unwrap();
        let (value, _) = out.variables.get(&z).unwrap();
        assert_eq!(
            value
                .clone()
                .unwrap_int()
                .into_data()
                .to_vec::<i32>()
                .unwrap(),
            vec![1, 1]
        );

        graph.observe::<_, Boolean, _>(z, &true).unwrap();
        assert!(matches!(graph.node(z), Some(Node::Observed(_))));
        assert!(graph.node(z).unwrap().is_discrete());
        graph.validate_indices().unwrap();
    }
}

/// Observing a discrete quantity with a continuous value, or the reverse, is refused. The two kinds
/// have different tensor element types, so this is not a shape question -- storing a float where an
/// integer belongs changes what the value *means*.
mod discrimination {
    use super::*;

    /// Observing a discrete quantity with a continuous value, or the reverse, is refused.
    ///
    /// The kinds have different tensor element types, so this is not a shape question -- storing a
    /// float where an integer belongs would change what the value *means*.
    #[test]
    fn a_value_of_the_wrong_kind_is_rejected() {
        let mut graph = Graph::<B>::new();
        let n = graph.add_parameter(Integer);
        let r = graph.add_parameter(Real);

        assert!(matches!(
            graph.observe::<_, Real, _>(n, &1.0f64),
            Err(skoro_core::error::SkoroError::QuantityTypeMismatch { id }) if id == n
        ));
        assert!(matches!(
            graph.observe::<_, Integer, _>(r, &1i32),
            Err(skoro_core::error::SkoroError::QuantityTypeMismatch { id }) if id == r
        ));
        graph.validate_indices().unwrap();
    }
}

/// A discrete quantity has no tangent space, and every pass has to route around it without anyone
/// downstream asking. A discrete *output* makes the whole node non-differentiable even when its
/// input is continuous, a discrete parent contributes no cotangent, and neither can join a tangent
/// layout. Getting this wrong does not produce a wrong number -- it produces a cotangent for a
/// quantity that has nowhere to put one.
mod never_differentiates {
    use super::*;

    /// A discretization bridge: continuous in, integer out.
    ///
    /// This is the one place the two kinds meet, and the compiler has to notice that the *output*
    /// being discrete makes the whole node non-differentiable even though its input is not.
    #[test]
    fn a_discrete_operation_reads_a_continuous_parent() {
        let mut graph = Graph::<B>::new();
        let x = graph.add_parameter(Real);
        let n = graph
            .add_mapped(Integer, DiscreteGraphOperation(RoundToInt), vec![x])
            .unwrap();

        let out = graph.evaluate_ids(&points([(x, 2.4)]), &[n]).unwrap();
        let (value, _) = out.evaluations.get(&n).unwrap();
        assert_eq!(
            value
                .clone()
                .unwrap_int()
                .into_data()
                .to_vec::<i32>()
                .unwrap(),
            vec![2]
        );

        // Asking for it as a continuous value is an error, not a silent cast.
        assert!(matches!(
            graph.evaluate(&points([(x, 2.4)]), n),
            Err(skoro_core::error::SkoroError::NotContinuous { id }) if id == n
        ));
    }

    /// A discrete parent contributes no derivative to a continuous child.
    ///
    /// `is_differentiable_parent` is what decides this, and getting it wrong would not produce a
    /// wrong number -- it would produce a cotangent for a quantity that has no tangent space.
    #[test]
    fn a_discrete_parent_receives_no_cotangent() {
        let mut graph = Graph::<B>::new();
        let x = graph.add_parameter(Real);
        let n = graph.add_parameter(Integer);
        // `ShiftByInt` reads both, but only `x` is differentiable.
        let y = graph
            .add_mapped(Real, DiffGraphOperation(ShiftByInt), vec![x, n])
            .unwrap();

        let mut point = points([(x, 1.5)]);
        point.insert(n, skoro_operation::Input::Int(int2(vec![2], 1, 1)));

        let cotangents = frames([(y, tensor3(vec![1.0], [1, 1, 1]))]);
        let out = graph.evaluate_ids_vjp(&point, &cotangents, &[y]).unwrap();

        assert!(out.cotangents.contains_key(&x));
        assert!(
            !out.cotangents.contains_key(&n),
            "a discrete leaf has no tangent space to receive a cotangent in"
        );

        // The value is still the sum of both, so the discrete parent is genuinely read.
        let (value, _) = out.primal.evaluations.get(&y).unwrap();
        assert_eq!(into_vec(value.clone().unwrap_float()), vec![3.5]);
    }

    /// Discrete quantities are not manifolds, so they cannot join a tangent layout.
    #[test]
    fn discrete_quantities_have_no_tangent_block() {
        use skoro_graph::tangent_layout::TangentLayout;

        let mut graph = Graph::<B>::new();
        let n = graph.add_parameter(Categorical(3));
        assert!(matches!(
            TangentLayout::from_ids(&graph, &[n]),
            Err(skoro_core::error::SkoroError::NotContinuous { id }) if id == n
        ));
    }
}
