//! The evaluation pipeline: `CompiledSequence`'s four passes, reached through `Graph`.
//!
//! Every derivative claim here is checked against a forward difference via
//! `skoro_graph::test_utils`, never against a recorded constant. A recorded constant only says
//! the answer did not change; the numerical check says it is right.

mod common;

use common::*;
use skoro_core::error::SkoroError;
use skoro_graph::test_utils::{
    test_hvp_cross_numerically, test_hvp_numerically, test_jvp_numerically, test_vjp_numerically,
};

// =============================================================================================
mod forward {
    use super::*;

    #[test]
    fn evaluates_a_chain_of_mapped_nodes() {
        let m = Arithmetic::new();
        let p = m.point();

        assert_eq!(into_vec(m.graph.evaluate(&p, m.sum).unwrap()), vec![8.0]);
        assert_eq!(into_vec(m.graph.evaluate(&p, m.prod).unwrap()), vec![15.0]);
        assert_eq!(into_vec(m.graph.evaluate(&p, m.sq).unwrap()), vec![64.0]);
    }

    /// `evaluate_ids` computes a shared subexpression once and reports both consumers.
    #[test]
    fn evaluate_ids_reports_every_target() {
        let m = Arithmetic::new();
        let out = m.graph.evaluate_ids(&m.point(), &[m.prod, m.sq]).unwrap();

        for (id, expected) in [(m.prod, 15.0), (m.sq, 64.0)] {
            let (value, _) = out.evaluations.get(&id).expect("target evaluated");
            assert_eq!(into_vec(value.clone().unwrap_float()), vec![expected]);
        }
        // The free variables come back separately from the derived values.
        assert!(out.variables.get(&m.a).is_some());
        assert!(out.variables.get(&m.b).is_some());
    }

    /// A parameter with no supplied point is a recoverable error, not a panic.
    #[test]
    fn a_missing_parameter_is_reported() {
        let m = Arithmetic::new();
        let partial = points([(m.a, 3.0)]);
        assert!(matches!(
            m.graph.evaluate(&partial, m.sum),
            Err(SkoroError::MissingPoint { id }) if id == m.b
        ));
    }

    /// The batch a pass runs at comes from the supplied points, and everything must match it.
    #[test]
    fn a_disagreeing_batch_is_rejected() {
        let m = Arithmetic::new();
        let mut p = m.point();
        p.insert(
            m.b,
            skoro_operation::Input::Float(tensor2(vec![1.0, 2.0], 2, 1)),
        );
        assert!(matches!(
            m.graph.evaluate(&p, m.sum),
            Err(SkoroError::BatchSizeMismatch { .. })
        ));
    }

    /// A batch-agnostic constant broadcasts to whatever batch the pass runs at.
    #[test]
    fn an_unbatched_constant_broadcasts() {
        use skoro_graph::operation::DiffGraphOperation;
        use skoro_operation::ops::Addition;
        use skoro_riemann::real::Real;

        let mut graph = Arithmetic::new();
        let k = graph.graph.add_constant(Real, &10.0f64);
        let shifted = graph
            .graph
            .add_mapped(Real, DiffGraphOperation(Addition), vec![graph.a, k])
            .unwrap();

        let mut p = skoro_operation::ValueMap::<B>::new();
        p.insert(
            graph.a,
            skoro_operation::Input::Float(tensor2(vec![1.0, 2.0, 3.0], 3, 1)),
        );
        assert_eq!(
            into_vec(graph.graph.evaluate(&p, shifted).unwrap()),
            vec![11.0, 12.0, 13.0]
        );
    }
}

// =============================================================================================
mod jvp {
    use super::*;

    #[test]
    fn pushforward_matches_a_forward_difference() {
        let m = Arithmetic::new();
        for out in [m.sum, m.prod, m.sq] {
            test_jvp_numerically(&m.graph, m.a, out, m.point(), DELTA, EPSILON);
            test_jvp_numerically(&m.graph, m.b, out, m.point(), DELTA, EPSILON);
        }
    }

    /// Rectangular Jacobian: 1-dimensional input, 3-dimensional output and back.
    #[test]
    fn pushforward_of_a_rectangular_jacobian() {
        let m = Scaling::new();
        test_jvp_numerically(&m.graph, m.s, m.y, m.point(), DELTA, EPSILON);
        test_jvp_numerically(&m.graph, m.v, m.y, m.point(), DELTA, EPSILON);
    }

    /// Frame counts must agree across the quantities of one pass.
    #[test]
    fn disagreeing_frame_counts_are_rejected() {
        let m = Scaling::new();
        let tangents = frames([
            (m.s, tensor3(vec![1.0, 0.0], [1, 1, 2])),
            (m.v, tensor3(vec![0.0; 9], [1, 3, 3])),
        ]);
        assert!(matches!(
            m.graph.evaluate_jvp(&m.point(), &tangents, m.y),
            Err(SkoroError::InconsistentFrameCount { .. })
        ));
    }
}

// =============================================================================================
mod vjp {
    use super::*;

    #[test]
    fn pullback_matches_a_forward_difference() {
        let m = Arithmetic::new();
        for out in [m.sum, m.prod, m.sq] {
            test_vjp_numerically(&m.graph, m.a, out, m.point(), DELTA, EPSILON);
            test_vjp_numerically(&m.graph, m.b, out, m.point(), DELTA, EPSILON);
        }
    }

    #[test]
    fn pullback_of_a_rectangular_jacobian() {
        let m = Scaling::new();
        test_vjp_numerically(&m.graph, m.s, m.y, m.point(), DELTA, EPSILON);
        test_vjp_numerically(&m.graph, m.v, m.y, m.point(), DELTA, EPSILON);
    }

    /// A leaf that does not reach the target has zero derivative with respect to it -- and that
    /// is an answer, so it comes back as an explicit zero rather than being absent.
    ///
    /// Reachable from any Jacobian whose column set is wider than the subgraph feeding its rows,
    /// where a missing entry would read as "not computed" instead of "computed, and zero".
    #[test]
    fn an_unreached_leaf_gets_a_zero_cotangent() {
        use skoro_riemann::real::Real;

        let mut m = Arithmetic::new();
        // A free variable nothing downstream of `sum` consumes.
        let spectator = m.graph.add_parameter(Real);
        let mut p = m.point();
        p.insert(spectator, skoro_operation::Input::Float(scalar(7.0)));

        // Compile from both, so the spectator is in the sequence but not in `sum`'s cone.
        let cotangents = frames([(m.sum, tensor3(vec![1.0], [1, 1, 1]))]);
        let out = m
            .graph
            .evaluate_ids_vjp(&p, &cotangents, &[m.sum, spectator])
            .unwrap();

        let zero = out
            .cotangents
            .get(&spectator)
            .expect("an unreached leaf still reports a cotangent");
        assert_eq!(zero.dims(), [1, 1, 1]);
        assert_eq!(into_vec(zero.clone()), vec![0.0]);
    }

    /// `evaluate_ord1` is the seeded-with-ones convenience over `evaluate_vjp`, with the frame
    /// axis squeezed away.
    #[test]
    fn ord1_squeezes_the_frame_axis() {
        let m = Arithmetic::new();
        let eval = m.graph.evaluate_ord1(&m.point(), m.prod).unwrap();
        assert_eq!(into_vec(eval.value), vec![15.0]);
        // d(ab)/da = b = 5, d(ab)/db = a = 3.
        assert_eq!(eval.cotangents[&m.a].dims(), [1, 1]);
        assert_eq!(into_vec(eval.cotangents[&m.a].clone()), vec![5.0]);
        assert_eq!(into_vec(eval.cotangents[&m.b].clone()), vec![3.0]);
    }
}

// =============================================================================================
mod hvp {
    use super::*;

    /// The diagonal blocks. `sq = (a + b)²` has `∂²/∂a² = 2`; `prod = ab` has `∂²/∂a² = 0`.
    #[test]
    fn diagonal_blocks_match_a_forward_difference() {
        let m = Arithmetic::new();
        for out in [m.sum, m.prod, m.sq] {
            test_hvp_numerically(&m.graph, m.a, out, m.point(), DELTA, EPSILON);
            test_hvp_numerically(&m.graph, m.b, out, m.point(), DELTA, EPSILON);
        }
    }

    /// The mixed blocks, which the diagonal check cannot reach.
    ///
    /// `∂²(ab)/∂a∂b = 1` is exactly the term a `hvp` that forgot its Leibniz rule drops while
    /// still passing every first-order test.
    #[test]
    fn mixed_blocks_match_a_forward_difference() {
        let m = Arithmetic::new();
        for out in [m.prod, m.sq] {
            test_hvp_cross_numerically(&m.graph, m.a, m.b, out, m.point(), DELTA, EPSILON);
            test_hvp_cross_numerically(&m.graph, m.b, m.a, out, m.point(), DELTA, EPSILON);
        }
    }

    #[test]
    fn mixed_blocks_across_differing_dimensions() {
        let m = Scaling::new();
        test_hvp_cross_numerically(&m.graph, m.s, m.v, m.y, m.point(), DELTA, EPSILON);
        test_hvp_cross_numerically(&m.graph, m.v, m.s, m.y, m.point(), DELTA, EPSILON);
    }

    /// A second-order pass needs both cotangent orders or neither.
    #[test]
    fn one_order_alone_is_rejected() {
        let m = Arithmetic::new();
        let ord1 = frames([(m.prod, tensor3(vec![1.0], [1, 1, 1]))]);
        let ord2 = frames([]);
        let tangents = vectors([(m.a, scalar(1.0))]);
        assert!(matches!(
            m.graph
                .evaluate_hvp(&m.point(), &tangents, &ord1, &ord2, m.prod),
            Err(SkoroError::InconsistentCotangents)
        ));
    }

    /// `evaluate_ord2` seeds ones/zeros and squeezes, like `evaluate_ord1`.
    #[test]
    fn ord2_squeezes_the_frame_axis() {
        let m = Arithmetic::new();
        let tangents = vectors([(m.a, scalar(1.0))]);
        let eval = m
            .graph
            .evaluate_ord2(&m.point(), &tangents, m.prod)
            .unwrap();
        assert_eq!(into_vec(eval.value), vec![15.0]);
        assert_eq!(eval.ord1_cotangents[&m.a].dims(), [1, 1]);
        // Seeding a tangent on `a` and reading the order-2 cotangent at `b` gives ∂²(ab)/∂a∂b.
        assert_eq!(into_vec(eval.ord2_cotangents[&m.b].clone()), vec![1.0]);
        assert_eq!(into_vec(eval.ord2_cotangents[&m.a].clone()), vec![0.0]);
    }
}

// =============================================================================================
mod log_density {
    use super::*;

    /// A model scored by two factors: `log p = (a + b) + ab`.
    ///
    /// Two factors rather than one, because a single factor cannot tell "summed the terms" from
    /// "returned the last term".
    fn scored() -> Arithmetic {
        let mut m = Arithmetic::new();
        m.graph.mark_log_likelihood(m.sum).unwrap();
        m.graph.mark_log_likelihood(m.prod).unwrap();
        m
    }

    #[test]
    fn factors_are_summed() {
        let m = scored();
        // (3 + 5) + (3 · 5) = 23
        assert_eq!(
            into_vec(m.graph.log_density(&m.point()).unwrap()),
            vec![23.0]
        );
    }

    #[test]
    fn ord1_reports_the_gradient_of_the_sum() {
        let m = scored();
        let eval = m.graph.log_density_ord1(&m.point()).unwrap();
        assert_eq!(into_vec(eval.value), vec![23.0]);
        // ∂/∂a [(a+b) + ab] = 1 + b = 6;  ∂/∂b = 1 + a = 4.
        assert_eq!(into_vec(eval.cotangents[&m.a].clone()), vec![6.0]);
        assert_eq!(into_vec(eval.cotangents[&m.b].clone()), vec![4.0]);
    }

    #[test]
    fn ord2_reports_the_hessian_product() {
        let m = scored();
        let tangents = vectors([(m.a, scalar(1.0))]);
        let eval = m.graph.log_density_ord2(&m.point(), &tangents).unwrap();
        assert_eq!(into_vec(eval.value), vec![23.0]);
        assert_eq!(into_vec(eval.ord1_cotangents[&m.a].clone()), vec![6.0]);
        // H·e_a = [∂²/∂a², ∂²/∂a∂b] = [0, 1].
        assert_eq!(into_vec(eval.ord2_cotangents[&m.a].clone()), vec![0.0]);
        assert_eq!(into_vec(eval.ord2_cotangents[&m.b].clone()), vec![1.0]);
    }

    /// An unscored model has no factors, so its log-density is the empty sum.
    #[test]
    fn an_unscored_model_scores_zero() {
        let m = Arithmetic::new();
        assert_eq!(
            into_vec(m.graph.log_density(&m.point()).unwrap()),
            vec![0.0]
        );
    }
}

// =============================================================================================
mod sampling {
    use super::*;
    use skoro_riemann::real::Real;

    /// `generate` runs the model's own samplers; nothing needs to be supplied.
    #[test]
    fn generate_runs_the_model_samplers() {
        let mut graph = skoro_graph::graph::Graph::<B>::new();
        let z = graph
            .add_stochastic(Real, ConstantSampler(4.0), vec![])
            .unwrap();

        let out = graph.generate(3).unwrap();
        let (value, _) = out
            .variables
            .get(&z)
            .expect("sampled quantity is a variable");
        assert_eq!(into_vec(value.clone().unwrap_float()), vec![4.0, 4.0, 4.0]);
    }

    /// A sampler reads its parents' realized values, which is the whole point of the
    /// parent-index plumbing in `SequenceSampler`.
    #[test]
    fn a_sampler_sees_its_parents() {
        let mut graph = skoro_graph::graph::Graph::<B>::new();
        let root = graph
            .add_stochastic(Real, ConstantSampler(10.0), vec![])
            .unwrap();
        let child = graph
            .add_stochastic(Real, ShiftSampler(0.5), vec![root])
            .unwrap();

        let out = graph.generate(1).unwrap();
        let (value, _) = out.variables.get(&child).unwrap();
        assert_eq!(into_vec(value.clone().unwrap_float()), vec![10.5]);
    }

    /// `generate_from` pins the quantities it is given and samples only the rest.
    #[test]
    fn generate_from_respects_supplied_points() {
        let mut graph = skoro_graph::graph::Graph::<B>::new();
        let root = graph
            .add_stochastic(Real, ConstantSampler(10.0), vec![])
            .unwrap();
        let child = graph
            .add_stochastic(Real, ShiftSampler(0.5), vec![root])
            .unwrap();

        let out = graph.generate_from(&points([(root, -1.0)]), 1).unwrap();
        let (value, _) = out.variables.get(&child).unwrap();
        assert_eq!(into_vec(value.clone().unwrap_float()), vec![-0.5]);
    }

    /// Without `use_model_samplers`, an unsupplied stochastic quantity is an error rather than a
    /// silent draw -- `evaluate` must never invent data.
    #[test]
    fn evaluation_will_not_sample_on_its_own() {
        let mut graph = skoro_graph::graph::Graph::<B>::new();
        let z = graph
            .add_stochastic(Real, ConstantSampler(4.0), vec![])
            .unwrap();
        let y = graph
            .add_mapped(
                Real,
                skoro_graph::operation::DiffGraphOperation(skoro_operation::ops::Addition),
                vec![z, z],
            )
            .unwrap();

        assert!(matches!(
            graph.evaluate(&skoro_operation::ValueMap::<B>::new(), y),
            Err(SkoroError::MissingPoint { id }) if id == z
        ));
    }

    /// A stochastic quantity is a differentiable leaf: it collects cotangents like a parameter.
    #[test]
    fn a_continuous_sampler_collects_cotangents() {
        let mut graph = skoro_graph::graph::Graph::<B>::new();
        let z = graph
            .add_stochastic(Real, ConstantSampler(4.0), vec![])
            .unwrap();
        let p = graph.add_parameter(Real);
        let y = graph
            .add_mapped(
                Real,
                skoro_graph::operation::DiffGraphOperation(
                    skoro_operation::ops::ScalarMultiplication,
                ),
                vec![z, p],
            )
            .unwrap();

        let point = points([(z, 4.0), (p, 6.0)]);
        test_vjp_numerically(&graph, z, y, point.clone(), DELTA, EPSILON);
        test_vjp_numerically(&graph, p, y, point, DELTA, EPSILON);
    }
}

// =============================================================================================
mod lowering {
    use super::*;
    use skoro_core::discrete::Integer;
    use skoro_graph::operation::{DiffGraphOperation, DiscreteGraphOperation};
    use skoro_operation::ops::Addition;
    use skoro_riemann::real::Real;

    /// A discrete operation compiles `forward_only`, so a differentiation pass routes around it
    /// instead of calling into it.
    ///
    /// `SequencedDiscreteOperation`'s `jvp_forward` is an `unreachable!()`. If the compiler ever
    /// stopped setting `forward_only`, this test would panic there rather than fail an
    /// assertion -- which is precisely the guarantee worth pinning down.
    #[test]
    fn a_discrete_node_is_skipped_by_a_differentiation_pass() {
        let mut graph = skoro_graph::graph::Graph::<B>::new();
        let x = graph.add_parameter(Real);
        let rounded = graph
            .add_mapped(Integer, DiscreteGraphOperation(RoundToInt), vec![x])
            .unwrap();
        let doubled = graph
            .add_mapped(Real, DiffGraphOperation(Addition), vec![x, x])
            .unwrap();

        let point = points([(x, 2.7)]);
        let tangents = frames([(x, tensor3(vec![1.0], [1, 1, 1]))]);
        let out = graph
            .evaluate_ids_jvp(&point, &tangents, &[doubled, rounded])
            .unwrap();

        // The discrete node still evaluates...
        let (value, _) = out.primal.evaluations.get(&rounded).unwrap();
        assert_eq!(
            value
                .clone()
                .unwrap_int()
                .into_data()
                .to_vec::<i32>()
                .unwrap(),
            vec![3]
        );
        // ...and carries no tangent, while its continuous sibling does.
        assert!(!out.tangents.contains_key(&rounded));
        assert_eq!(into_vec(out.tangents[&doubled].clone()), vec![2.0]);
    }

    /// A mapped node over only constants folds at construction, and every later pass reads the
    /// cache instead of re-running the operation.
    #[test]
    fn an_all_constant_node_is_folded_and_reused() {
        use skoro_graph::node::{MappedNode, Node};

        let mut graph = skoro_graph::graph::Graph::<B>::new();
        let c = graph.add_constant(Real, &2.0f64);
        let d = graph.add_constant(Real, &5.0f64);
        let folded = graph
            .add_mapped(Real, DiffGraphOperation(SumOperation), vec![c, d])
            .unwrap();

        assert!(matches!(
            graph.node(folded),
            Some(Node::Mapped(MappedNode { value: Some(_), .. }))
        ));

        // `SumOperation` only implements `ForwardOnly`, so a pass that re-entered it in VJP mode
        // would panic. Reaching an answer at all is the assertion.
        let cotangents = frames([(folded, tensor3(vec![1.0], [1, 1, 1]))]);
        let out = graph
            .evaluate_ids_vjp(
                &skoro_operation::ValueMap::<B>::new(),
                &cotangents,
                &[folded],
            )
            .unwrap();
        let (value, _) = out.primal.evaluations.get(&folded).unwrap();
        assert_eq!(into_vec(value.clone().unwrap_float()), vec![7.0]);
    }

    /// The JVP pass reuses that fold too, on the same `constant_inputs` gate as VJP and HVP.
    ///
    /// The three passes each carry their own copy of the step loop, and JVP used to gate on
    /// `forward_only` instead -- which is `!op.is_differentiable()`, a strictly narrower
    /// condition. A *differentiable* operation over only constants therefore got re-entered in
    /// JVP mode, so a graph that `evaluate_ids_vjp` handled would fail under
    /// `evaluate_ids_jvp`. `SumOperation` implements `ForwardOnly` alone, so reaching an answer
    /// at all is the assertion.
    #[test]
    fn an_all_constant_node_is_reused_by_the_jvp_pass_too() {
        let mut graph = skoro_graph::graph::Graph::<B>::new();
        let c = graph.add_constant(Real, &2.0f64);
        let d = graph.add_constant(Real, &5.0f64);
        let folded = graph
            .add_mapped(Real, DiffGraphOperation(SumOperation), vec![c, d])
            .unwrap();

        let out = graph
            .evaluate_ids_jvp(
                &skoro_operation::ValueMap::<B>::new(),
                &frames([]),
                &[folded],
            )
            .unwrap();
        let (value, _) = out.primal.evaluations.get(&folded).unwrap();
        assert_eq!(into_vec(value.clone().unwrap_float()), vec![7.0]);
        // Nothing free above it, so it has no tangent -- not a zero one.
        assert!(!out.tangents.contains_key(&folded));
    }

    /// A batch-pinned constant forces every pass to run at its batch.
    #[test]
    fn a_pinned_constant_fixes_the_batch() {
        let mut graph = skoro_graph::graph::Graph::<B>::new();
        let pinned = graph
            .add_batched_constant(Real, &[1.0f64, 2.0, 3.0])
            .unwrap();
        let x = graph.add_parameter(Real);
        let y = graph
            .add_mapped(Real, DiffGraphOperation(Addition), vec![x, pinned])
            .unwrap();

        // Batch 3 agrees.
        let mut ok = skoro_operation::ValueMap::<B>::new();
        ok.insert(
            x,
            skoro_operation::Input::Float(tensor2(vec![10.0, 20.0, 30.0], 3, 1)),
        );
        assert_eq!(
            into_vec(graph.evaluate(&ok, y).unwrap()),
            vec![11.0, 22.0, 33.0]
        );

        // Batch 1 does not, and is an error rather than a broadcast.
        assert!(matches!(
            graph.evaluate(&points([(x, 10.0)]), y),
            Err(SkoroError::BatchSizeMismatch { .. })
        ));
    }

    /// The same disagreement, reported by each of the four passes rather than only the forward
    /// one.
    ///
    /// Every pass carries its own copy of the step loop, so "a pinned value is materialized at
    /// the pass's batch, and refuses if it cannot be" is four separate claims. A pass that let
    /// the failure through would broadcast row 0 across the batch and hand back a gradient that
    /// looks entirely reasonable.
    #[test]
    fn a_pinned_constant_fixes_the_batch_in_every_pass() {
        let mut graph = skoro_graph::graph::Graph::<B>::new();
        let pinned = graph
            .add_batched_constant(Real, &[1.0f64, 2.0, 3.0])
            .unwrap();
        let x = graph.add_parameter(Real);
        let y = graph
            .add_mapped(Real, DiffGraphOperation(Addition), vec![x, pinned])
            .unwrap();

        let wrong = points([(x, 10.0)]);
        let seed = tensor3(vec![1.0], [1, 1, 1]);

        assert!(matches!(
            graph.evaluate_jvp(&wrong, &frames([(x, seed.clone())]), y),
            Err(SkoroError::BatchSizeMismatch { .. })
        ));
        assert!(matches!(
            graph.evaluate_vjp(&wrong, &frames([(y, seed.clone())]), y),
            Err(SkoroError::BatchSizeMismatch { .. })
        ));
        assert!(matches!(
            graph.evaluate_hvp(
                &wrong,
                &vectors([(x, scalar(1.0))]),
                &frames([(y, seed.clone())]),
                &frames([(y, seed.zeros_like())]),
                y,
            ),
            Err(SkoroError::BatchSizeMismatch { .. })
        ));
    }

    /// A free variable with no point supplied is a named error in every pass, never a panic.
    ///
    /// This is the leaf-step failure the passes propagate rather than raise: `non_op_forward`
    /// reports it once and each pass has to hand it back up. The complaint names the quantity,
    /// which is the whole point -- an unwrapped `None` deep in a sequence names nothing.
    #[test]
    fn a_missing_point_is_reported_by_every_pass() {
        let m = Arithmetic::new();
        let nothing = skoro_operation::ValueMap::<B>::new();
        let seed = tensor3(vec![1.0], [1, 1, 1]);

        for out in [
            m.graph.evaluate(&nothing, m.sum).err(),
            m.graph
                .evaluate_jvp(&nothing, &frames([(m.a, seed.clone())]), m.sum)
                .err(),
            m.graph
                .evaluate_vjp(&nothing, &frames([(m.sum, seed.clone())]), m.sum)
                .err(),
            m.graph
                .evaluate_hvp(
                    &nothing,
                    &vectors([(m.a, scalar(1.0))]),
                    &frames([(m.sum, seed.clone())]),
                    &frames([(m.sum, seed.zeros_like())]),
                    m.sum,
                )
                .err(),
        ] {
            assert!(
                matches!(out, Some(SkoroError::MissingPoint { .. })),
                "{out:?}"
            );
        }
    }
}

// =============================================================================================
/// Seeding paths that only a non-operation step reaches.
///
/// `non_op_forward` threads tangents onto parameters *and* samplers, and `hvp_forward` seeds a
/// tangent onto a folded operation whose inputs are all constant. Both are branches no test
/// above passes through, because they need a seed on something that is not the plain free
/// variable at the bottom of a chain.
mod seeding {
    use super::*;
    use skoro_graph::operation::DiffGraphOperation;
    use skoro_operation::ops::{Addition, ScalarMultiplication};
    use skoro_riemann::real::Real;

    /// A stochastic quantity carries a seeded tangent just as a parameter does.
    #[test]
    fn a_sampler_node_carries_a_seeded_tangent() {
        let mut graph = skoro_graph::graph::Graph::<B>::new();
        let z = graph
            .add_stochastic(Real, ConstantSampler(4.0), vec![])
            .unwrap();
        let p = graph.add_parameter(Real);
        let y = graph
            .add_mapped(Real, DiffGraphOperation(ScalarMultiplication), vec![z, p])
            .unwrap();

        let point = points([(z, 4.0), (p, 6.0)]);
        let tangents = frames([(z, tensor3(vec![1.0], [1, 1, 1]))]);
        let out = graph.evaluate_ids_jvp(&point, &tangents, &[y]).unwrap();
        // d(z·p)/dz = p = 6.
        assert_eq!(into_vec(out.tangents[&y].clone()), vec![6.0]);
    }

    /// The same seed, in a second-order pass, where tangents are rank-2 vectors instead.
    #[test]
    fn a_sampler_node_carries_a_seeded_tangent_vector() {
        let mut graph = skoro_graph::graph::Graph::<B>::new();
        let z = graph
            .add_stochastic(Real, ConstantSampler(4.0), vec![])
            .unwrap();
        let p = graph.add_parameter(Real);
        let y = graph
            .add_mapped(Real, DiffGraphOperation(ScalarMultiplication), vec![z, p])
            .unwrap();

        let point = points([(z, 4.0), (p, 6.0)]);
        test_hvp_cross_numerically(&graph, z, p, y, point, DELTA, EPSILON);
    }

    /// A folded, all-constant node still accepts a tangent in a second-order pass.
    ///
    /// It contributes nothing — a constant has no derivative — but the seed has to be *placed*
    /// so the downstream operation reads a full tangent slot rather than a hole.
    #[test]
    fn a_folded_node_accepts_a_tangent_seed() {
        let mut graph = skoro_graph::graph::Graph::<B>::new();
        let c = graph.add_constant(Real, &2.0f64);
        let d = graph.add_constant(Real, &5.0f64);
        // All parents constant, so this folds and compiles with `constant_inputs`.
        let folded = graph
            .add_mapped(Real, DiffGraphOperation(Addition), vec![c, d])
            .unwrap();
        let x = graph.add_parameter(Real);
        let y = graph
            .add_mapped(
                Real,
                DiffGraphOperation(ScalarMultiplication),
                vec![folded, x],
            )
            .unwrap();

        let point = points([(x, 3.0)]);
        let tangents = vectors([(folded, scalar(1.0)), (x, scalar(1.0))]);
        let ord1 = frames([(y, tensor3(vec![1.0], [1, 1, 1]))]);
        let ord2 = frames([(y, tensor3(vec![0.0], [1, 1, 1]))]);

        let out = graph
            .evaluate_hvp(&point, &tangents, &ord1, &ord2, y)
            .unwrap();
        // y = (c + d)·x = 7·3 = 21, and dy/dx = 7.
        assert_eq!(into_vec(out.value), vec![21.0]);
        assert_eq!(into_vec(out.ord1_cotangents[&x].clone()), vec![7.0]);
        // The seed on the folded node is reported back, having been placed in the sequence.
        assert_eq!(into_vec(out.tangents[&folded].clone()), vec![1.0]);
    }

    /// The same, one order down: a folded node accepts a tangent *frame* in a JVP pass.
    ///
    /// It contributes nothing downstream — a folded node is a constant as far as the graph is
    /// concerned, so `y`'s pushforward is `folded · dx` and the seed on `folded` itself never
    /// enters it. Placing the seed is still the pass's job: the slot has to be reported back,
    /// the same as the second-order case above, or a caller that seeded a whole layout gets an
    /// answer with a hole in it and no way to tell which quantity it belongs to.
    #[test]
    fn a_folded_node_accepts_a_tangent_frame() {
        let mut graph = skoro_graph::graph::Graph::<B>::new();
        let c = graph.add_constant(Real, &2.0f64);
        let d = graph.add_constant(Real, &5.0f64);
        let folded = graph
            .add_mapped(Real, DiffGraphOperation(Addition), vec![c, d])
            .unwrap();
        let x = graph.add_parameter(Real);
        let y = graph
            .add_mapped(
                Real,
                DiffGraphOperation(ScalarMultiplication),
                vec![folded, x],
            )
            .unwrap();

        let point = points([(x, 3.0)]);
        let seed = tensor3(vec![1.0], [1, 1, 1]);
        let tangents = frames([(folded, seed.clone()), (x, seed)]);
        let out = graph.evaluate_jvp(&point, &tangents, y).unwrap();

        assert_eq!(into_vec(out.value), vec![21.0]);
        // dy = folded·dx = 7, with no term from the seed on `folded`.
        assert_eq!(into_vec(out.tangents[&y].clone()), vec![7.0]);
        assert_eq!(into_vec(out.tangents[&folded].clone()), vec![1.0]);
    }
}

// =============================================================================================
mod defaults {
    use super::*;
    use skoro_graph::graph::Graph;
    use skoro_riemann::real::Real;

    #[test]
    fn an_empty_graph_is_the_default() {
        let graph = Graph::<B>::default();
        assert_eq!(graph.node_count(), 0);
        assert!(graph.log_likelihoods().is_empty());
        assert!(graph.pinned_batch().is_none());
        assert!(format!("{graph:?}").contains("node_count"));
        graph.validate_indices().unwrap();
    }

    /// With no points supplied there is nothing to read a batch from, so a pass runs at batch 1.
    ///
    /// Reachable whenever every quantity in the cone is constant — a model scored entirely by
    /// frozen factors, which is what a fully-observed graph becomes.
    #[test]
    fn an_evaluation_with_no_points_runs_at_batch_one() {
        let mut graph = Graph::<B>::new();
        let c = graph.add_constant(Real, &2.0f64);
        let d = graph.add_constant(Real, &5.0f64);
        let sum = graph
            .add_mapped(
                Real,
                skoro_graph::operation::DiffGraphOperation(SumOperation),
                vec![c, d],
            )
            .unwrap();
        graph.mark_log_likelihood(sum).unwrap();

        let empty = skoro_operation::ValueMap::<B>::new();
        assert_eq!(into_vec(graph.evaluate(&empty, sum).unwrap()), vec![7.0]);
        assert_eq!(into_vec(graph.log_density(&empty).unwrap()), vec![7.0]);
        assert_eq!(
            into_vec(graph.log_density_ord1(&empty).unwrap().value),
            vec![7.0]
        );
        let no_tangents = vectors([]);
        assert_eq!(
            into_vec(graph.log_density_ord2(&empty, &no_tangents).unwrap().value),
            vec![7.0]
        );

        // The multi-target entry points derive the batch the same way, each in its own copy of
        // that `if let`, and each is the one a caller reaches when asking for several nodes at
        // once. A constant-only cone is also the only way a `Constant` step is ever seeded in a
        // JVP pass, so this is what proves the seed check reaches one.
        let no_frames = frames([]);
        let ones = frames([(sum, tensor3(vec![1.0], [1, 1, 1]))]);
        let zeros = frames([(sum, tensor3(vec![0.0], [1, 1, 1]))]);
        let ids = [sum];
        assert_eq!(
            into_vec(
                graph
                    .evaluate_ids_jvp(&empty, &no_frames, &ids)
                    .unwrap()
                    .primal
                    .evaluations
                    .get(&sum)
                    .unwrap()
                    .0
                    .clone()
                    .unwrap_float()
            ),
            vec![7.0]
        );
        assert_eq!(
            into_vec(
                graph
                    .evaluate_ids_ord1(&empty, &ids)
                    .unwrap()
                    .primal
                    .evaluations
                    .get(&sum)
                    .unwrap()
                    .0
                    .clone()
                    .unwrap_float()
            ),
            vec![7.0]
        );
        assert_eq!(
            into_vec(
                graph
                    .evaluate_ids_ord2(&empty, &no_tangents, &ids)
                    .unwrap()
                    .primal
                    .evaluations
                    .get(&sum)
                    .unwrap()
                    .0
                    .clone()
                    .unwrap_float()
            ),
            vec![7.0]
        );
        assert_eq!(
            into_vec(
                graph
                    .evaluate_ids_hvp(&empty, &no_tangents, &ones, &zeros, &ids)
                    .unwrap()
                    .primal
                    .evaluations
                    .get(&sum)
                    .unwrap()
                    .0
                    .clone()
                    .unwrap_float()
            ),
            vec![7.0]
        );
    }

    /// A likelihood factor may itself be a parent, and its differentiability follows whether it
    /// folded — the same rule as any other mapped node.
    #[test]
    fn a_likelihood_factor_can_be_a_parent() {
        let mut graph = Graph::<B>::new();
        let x = graph.add_parameter(Real);
        let factor = graph
            .add_mapped(
                Real,
                skoro_graph::operation::DiffGraphOperation(skoro_operation::ops::Addition),
                vec![x, x],
            )
            .unwrap();
        graph.mark_log_likelihood(factor).unwrap();

        let doubled = graph
            .add_mapped(
                Real,
                skoro_graph::operation::DiffGraphOperation(skoro_operation::ops::Addition),
                vec![factor, factor],
            )
            .unwrap();

        let point = points([(x, 1.5)]);
        assert_eq!(
            into_vec(graph.evaluate(&point, doubled).unwrap()),
            vec![6.0]
        );
        test_vjp_numerically(&graph, x, doubled, point, DELTA, EPSILON);
    }
}
