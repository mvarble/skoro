//! Mutating the arena: adding, marking, observing, demoting and removing nodes.
//!
//! Every test here ends by asserting `validate_indices()`, because the derived collections
//! (`children`, `log_likelihoods`, `stochastic_to_log_density`, `pinnable`, `private_nodes`) are
//! maintained by hand across a dozen methods and that is exactly the arrangement that drifts.
//! The mutators call it themselves in debug builds; asserting it again is what keeps the *release*
//! behaviour honest.

mod common;

use common::*;
use skoro_core::any_tensor::QuantityValue;
use skoro_core::error::SkoroError;
use skoro_core::id::{Id, IdGenerator};
use skoro_graph::{
    graph::{Graph, TermKind},
    node::{MappedNode, Node, ObservedNode},
    operation::DiffGraphOperation,
};
use skoro_riemann::{euclidean::EuclideanD, real::Real};

/// An id no graph in this file ever issued.
///
/// Ids are issued from zero per graph, so a *fresh* generator's first id is `Id(0)` -- which is
/// the first node of every graph here. Skipping well past the largest fixture is what makes it
/// actually unknown.
fn bogus_id() -> Id {
    let generator = IdGenerator::new();
    let mut id = generator.issue_id();
    for _ in 0..1000 {
        id = generator.issue_id();
    }
    id
}

fn inert(graph: &mut Graph<B>, parents: Vec<Id>) -> Id {
    graph
        .add_mapped(Real, DiffGraphOperation(InertOperation), parents)
        .unwrap()
}

fn summed(graph: &mut Graph<B>, parents: Vec<Id>) -> Id {
    graph
        .add_mapped(Real, DiffGraphOperation(SumOperation), parents)
        .unwrap()
}

fn folded(graph: &Graph<B>, id: Id) -> bool {
    matches!(
        graph.node(id),
        Some(Node::Mapped(MappedNode { value: Some(_), .. }))
    )
}

fn stochastic(graph: &mut Graph<B>, parents: Vec<Id>) -> Id {
    graph
        .add_stochastic(Real, ConstantSampler(0.0), parents)
        .unwrap()
}

/// The shape of the arena: parents recorded on a node and inverted into its children, a
/// topological order that puts ancestors first, and the quantity type each node kind reports.
/// A parent that does not exist is rejected at the call that named it.
mod topology {
    use super::*;

    #[test]
    fn parents_are_recorded_on_the_node_and_inverted_into_children() {
        let mut graph = Graph::<B>::new();
        let c1 = graph.add_parameter(Real);
        let c2 = graph.add_constant(Real, &2.0f64);

        let stoch = stochastic(&mut graph, vec![c1, c2]);
        assert!(graph.contains(stoch));

        let Node::Stochastic(node) = graph.node(stoch).unwrap() else {
            panic!("expected a stochastic node");
        };
        assert!(node.parent_ids.contains(&c1));
        assert!(node.parent_ids.contains(&c2));

        // The reverse edge is derived, not stored twice.
        assert!(graph.children_of(c1).unwrap().contains(&stoch));
        assert!(graph.children_of(c2).unwrap().contains(&stoch));

        let m1 = inert(&mut graph, vec![stoch]);
        assert!(graph.contains(m1));
        assert_eq!(graph.node_count(), 4);
        assert_eq!(graph.nodes().count(), 4);
        graph.validate_indices().unwrap();
    }

    #[test]
    fn a_parent_that_does_not_exist_is_rejected() {
        let mut graph = Graph::<B>::new();
        assert!(matches!(
            graph.add_stochastic(Real, ConstantSampler(0.0), vec![bogus_id()]),
            Err(SkoroError::DanglingParent { .. })
        ));
        assert!(matches!(
            graph.add_mapped(Real, DiffGraphOperation(InertOperation), vec![bogus_id()]),
            Err(SkoroError::DanglingParent { .. })
        ));
        graph.validate_indices().unwrap();
    }

    /// `topological_sort` orders ancestors before descendants, from the requested leaves back.
    #[test]
    fn topological_sort_puts_ancestors_first() {
        let mut graph = Graph::<B>::new();
        let root = graph.add_parameter(Real);
        let mid = inert(&mut graph, vec![root]);
        let leaf = inert(&mut graph, vec![mid]);

        assert_eq!(graph.topological_sort(&[leaf]), vec![root, mid, leaf]);
        // Sorting from an interior node walks only that node's cone.
        assert_eq!(graph.topological_sort(&[mid]), vec![root, mid]);
    }

    #[test]
    fn quantity_types_round_trip_and_report_mismatches() {
        let mut graph = Graph::<B>::new();
        let r = graph.add_parameter(Real);
        let v = graph.add_parameter(EuclideanD(3));

        assert_eq!(graph.get_quantity_type::<_, Real>(r).unwrap(), Real);
        assert_eq!(
            graph.get_quantity_type::<_, EuclideanD>(v).unwrap(),
            EuclideanD(3)
        );
        assert!(matches!(
            graph.get_quantity_type::<_, EuclideanD>(r),
            Err(SkoroError::QuantityTypeMismatch { id }) if id == r
        ));
        assert!(matches!(
            graph.get_quantity_type::<_, Real>(bogus_id()),
            Err(SkoroError::NodeNotFound { .. })
        ));

        assert_eq!(graph.get_manifold(v).unwrap().dimension(), 3);
    }

    /// Every node kind answers `get_quantity_type`, not just the two free ones.
    ///
    /// The lookup is what `observe` and `mark_log_density` validate through, and both are
    /// reached on nodes that have already changed kind once -- an observed quantity, a mapped
    /// node promoted to a density. A match arm that forgot one of those would only fail on the
    /// second mutation of a node, which is the hardest kind of gap to notice.
    #[test]
    fn every_node_kind_reports_its_quantity_type() {
        let mut graph = Graph::<B>::new();
        let c = graph.add_constant(Real, &1.0f64);
        let p = graph.add_parameter(Real);
        let s = stochastic(&mut graph, vec![]);
        let m = inert(&mut graph, vec![p]);
        let d = inert(&mut graph, vec![p]);
        graph.mark_log_density(d, s).unwrap();

        let obs = stochastic(&mut graph, vec![]);
        graph.observe::<_, Real, _>(obs, &2.0f64).unwrap();

        for id in [c, p, s, m, d, obs] {
            assert_eq!(
                graph.get_quantity_type::<_, Real>(id).unwrap(),
                Real,
                "{id:?}"
            );
        }
        graph.validate_indices().unwrap();
    }

    /// A discrete quantity has no manifold, so neither a dimension nor an inverse-retraction.
    #[test]
    fn a_discrete_quantity_has_no_tangent_space() {
        use skoro_core::discrete::Integer;
        let mut graph = Graph::<B>::new();
        let n = graph.add_parameter(Integer);

        assert!(matches!(
            graph.get_manifold(n),
            Err(SkoroError::NotContinuous { id }) if id == n
        ));
        assert!(matches!(
            graph.inverse_retraction_op(n),
            Err(SkoroError::NotContinuous { id }) if id == n
        ));
    }

    /// The continuous counterpart: dimension is the *manifold* dimension, and it comes back
    /// paired with the operation that produces `x ⊖ p`.
    #[test]
    fn a_continuous_quantity_yields_its_inverse_retraction() {
        let mut graph = Graph::<B>::new();
        let v = graph.add_parameter(EuclideanD(3));
        let (dimension, op) = graph.inverse_retraction_op(v).unwrap();
        assert_eq!(dimension, 3);
        assert!(op.is_differentiable());
    }
}

/// `remove` is leaf-only and refuses a node any survivor still points into; `remove_closure` is the
/// downward walk that takes the whole cone instead. What both must leave consistent is the density
/// bookkeeping: removing a scored quantity retires its orphaned density, and removing a density
/// clears both indices that named it.
mod removal {
    use super::*;

    #[test]
    fn a_node_with_a_surviving_child_cannot_be_removed() {
        let mut graph = Graph::<B>::new();
        let p1 = graph.add_parameter(Real);
        let _m1 = inert(&mut graph, vec![p1]);

        assert!(matches!(
            graph.remove(&[p1]),
            Err(SkoroError::NodeStillReferenced { .. })
        ));
        graph.validate_indices().unwrap();
    }

    #[test]
    fn removing_an_absent_node_is_reported() {
        let mut graph = Graph::<B>::new();
        assert!(matches!(
            graph.remove(&[bogus_id()]),
            Err(SkoroError::NodeNotFound { .. })
        ));
    }

    /// `remove_closure` takes a whole subtree, including a diamond, in one call.
    ///
    /// `remove` alone rejects any of these roots: each has a child outside the batch.
    #[test]
    fn remove_closure_takes_a_diamond_subtree() {
        let mut graph = Graph::<B>::new();
        let root = graph.add_parameter(Real);
        let keep = graph.add_parameter(Real);

        let left = inert(&mut graph, vec![root]);
        let right = inert(&mut graph, vec![root]);
        // Joins the two branches, and also depends on a node that must survive.
        let join = inert(&mut graph, vec![left, right, keep]);

        assert!(matches!(
            graph.remove(&[root]),
            Err(SkoroError::NodeStillReferenced { .. })
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

    /// Removing an already-removed subtree is a no-op rather than an error.
    #[test]
    fn remove_closure_skips_ids_that_are_already_gone() {
        let mut graph = Graph::<B>::new();
        let root = graph.add_parameter(Real);
        assert_eq!(graph.remove_closure(&[root]).unwrap().len(), 1);
        assert_eq!(graph.remove_closure(&[root]).unwrap().len(), 0);
        graph.validate_indices().unwrap();
    }

    /// A shared parent keeps its remaining children when only one of them is removed.
    #[test]
    fn removing_one_child_leaves_a_shared_parent_intact() {
        let mut graph = Graph::<B>::new();
        let shared = graph.add_parameter(Real);
        let first = inert(&mut graph, vec![shared]);
        let second = inert(&mut graph, vec![shared]);

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

    /// Removing a stochastic quantity retires the density that scored it.
    ///
    /// The density need not be a *child* of the quantity it scores — `mark_log_density` never
    /// required that — so removing the quantity can leave behind a density whose `stochastic_id`
    /// names a node that is gone. Left alone that is exactly the disagreement
    /// `validate_indices` reports: the density still claims a quantity the reverse index no
    /// longer maps.
    #[test]
    fn removing_a_scored_quantity_retires_its_orphaned_density() {
        let mut graph = Graph::<B>::new();
        let stoch = stochastic(&mut graph, vec![]);
        let other = graph.add_parameter(Real);
        // The density hangs off `other`, not off the quantity it scores.
        let density = inert(&mut graph, vec![other]);
        graph.mark_log_density(density, stoch).unwrap();

        graph.remove(&[stoch]).unwrap();

        // The density survives as an ordinary mapped node, having lost its scoring role.
        assert!(matches!(graph.node(density), Some(Node::Mapped(_))));
        assert_eq!(graph.stochastic_of(density), None);
        assert!(!graph.log_likelihoods().contains(&density));
        graph.validate_indices().unwrap();
    }

    /// An unscored quantity has nothing to retire, and removal must not invent an entry to clear.
    ///
    /// Most stochastic quantities in a model *are* scored, so the `None` arm is the one a fix to
    /// the scored case can quietly break -- a `remove` that reached for
    /// `stochastic_to_log_density[&id]` unconditionally would pass every test above and panic
    /// here.
    #[test]
    fn removing_an_unscored_quantity_touches_no_density_index() {
        let mut graph = Graph::<B>::new();
        let scored = stochastic(&mut graph, vec![]);
        let unscored = stochastic(&mut graph, vec![]);
        let free = graph.add_parameter(Real);
        let density = inert(&mut graph, vec![free]);
        graph.mark_log_density(density, scored).unwrap();

        graph.remove(&[unscored]).unwrap();

        // The other quantity's mapping is untouched.
        assert_eq!(graph.log_density_of(scored), Some(density));
        graph.validate_indices().unwrap();
    }

    /// Removing a likelihood factor takes it out of the arena of scoring terms.
    ///
    /// `log_likelihoods` is what every objective sums over, so a removed factor left named there
    /// is not a stale index that someone might later notice -- it is a `compile_backward_from`
    /// target that no longer exists.
    #[test]
    fn removing_a_factor_takes_it_out_of_the_scoring_arena() {
        let mut graph = Graph::<B>::new();
        let p = graph.add_parameter(Real);
        let factor = inert(&mut graph, vec![p]);
        graph.mark_log_likelihood(factor).unwrap();
        assert_eq!(graph.log_likelihoods(), &[factor]);

        graph.remove(&[factor]).unwrap();
        assert!(graph.log_likelihoods().is_empty());
        graph.validate_indices().unwrap();
    }

    /// The same, for a density: both the arena and the reverse index let go.
    #[test]
    fn removing_a_density_clears_both_indices() {
        let mut graph = Graph::<B>::new();
        let stoch = stochastic(&mut graph, vec![]);
        let density = inert(&mut graph, vec![stoch]);
        graph.mark_log_density(density, stoch).unwrap();

        graph.remove(&[density]).unwrap();
        assert!(graph.log_likelihoods().is_empty());
        assert_eq!(graph.log_density_of(stoch), None);
        graph.validate_indices().unwrap();
    }

    /// Retiring a quantity and its density together is one closed batch, and the caller gets the
    /// density back as the factor it still was.
    #[test]
    fn removing_a_quantity_and_its_density_together() {
        let mut graph = Graph::<B>::new();
        let stoch = stochastic(&mut graph, vec![]);
        let density = inert(&mut graph, vec![stoch]);
        graph.mark_log_density(density, stoch).unwrap();

        // Quantity first, so the density is still a `LogLikelihood` when its own turn comes.
        // Retiring it early would change the node kind handed back to the caller, which is why
        // a density inside the batch is left alone rather than demoted.
        let removed = graph.remove(&[stoch, density]).unwrap();
        assert_eq!(removed.len(), 2);
        assert!(matches!(removed[1], Node::LogLikelihood(_)));
        assert_eq!(graph.node_count(), 0);
        graph.validate_indices().unwrap();
    }
}

/// Marking a mapped node as a likelihood term, and as the density *of* a quantity. The two marks
/// are distinct states, the transitions between them are idempotent, and the ones that would leave
/// a quantity with two densities -- or a density bound to two quantities -- are rejected.
mod density_marks {
    use super::*;

    /// `stochastic_of` is derived from the node, so it must answer for every id, not only for
    /// those that once had a map entry.
    ///
    /// A free-standing likelihood factor scores no particular quantity, and a plain mapped node
    /// is not a density at all; both are `None`.
    #[test]
    fn distinguishes_density_likelihood_and_plain_mapped() {
        let mut graph = Graph::<B>::new();
        let stoch = stochastic(&mut graph, vec![]);

        let plain = inert(&mut graph, vec![stoch]);
        assert_eq!(graph.stochastic_of(plain), None);
        assert!(!graph.is_scoring(plain));

        let likelihood = inert(&mut graph, vec![stoch]);
        graph.mark_log_likelihood(likelihood).unwrap();
        assert_eq!(graph.stochastic_of(likelihood), None);
        assert!(graph.is_scoring(likelihood));

        let density = inert(&mut graph, vec![stoch]);
        graph.mark_log_density(density, stoch).unwrap();
        assert_eq!(graph.stochastic_of(density), Some(stoch));
        assert_eq!(graph.log_density_of(stoch), Some(density));
        graph.validate_indices().unwrap();
    }

    /// Promoting an existing likelihood factor to a density must not name it twice.
    ///
    /// The node is already in `log_likelihoods`; pushing it again would leave the term counted
    /// twice in every objective that sums them.
    #[test]
    fn promoting_a_likelihood_to_a_density_does_not_duplicate_the_term() {
        let mut graph = Graph::<B>::new();
        let stoch = stochastic(&mut graph, vec![]);
        let factor = inert(&mut graph, vec![stoch]);

        graph.mark_log_likelihood(factor).unwrap();
        graph.mark_log_density(factor, stoch).unwrap();

        assert_eq!(graph.log_likelihoods(), &[factor]);
        assert_eq!(graph.stochastic_of(factor), Some(stoch));
        graph.validate_indices().unwrap();
    }

    /// Re-marking the same pair is a no-op, not a duplicate.
    #[test]
    fn marking_the_same_density_twice_is_idempotent() {
        let mut graph = Graph::<B>::new();
        let stoch = stochastic(&mut graph, vec![]);
        let density = inert(&mut graph, vec![stoch]);

        graph.mark_log_density(density, stoch).unwrap();
        graph.mark_log_density(density, stoch).unwrap();
        assert_eq!(graph.log_likelihoods(), &[density]);
        graph.validate_indices().unwrap();
    }

    /// A stochastic quantity has at most one density — the reading `DuplicateDensity` names.
    ///
    /// The guard used to key on the density id, so a *second* density claiming an already-scored
    /// quantity silently overwrote the first's reverse-map entry instead of erroring.
    #[test]
    fn a_second_density_for_one_quantity_is_rejected() {
        let mut graph = Graph::<B>::new();
        let stoch = stochastic(&mut graph, vec![]);

        let first = inert(&mut graph, vec![stoch]);
        graph.mark_log_density(first, stoch).unwrap();

        let second = inert(&mut graph, vec![stoch]);
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
        graph.validate_indices().unwrap();
    }

    /// The other direction: one quantity per density. Rebinding an existing density to a
    /// different quantity would strand the original `stochastic_to_log_density` entry.
    ///
    /// Checked unconditionally rather than behind `debug_assertions`: a release build that
    /// silently rebinds is precisely the case with no assertion left to catch it.
    #[test]
    fn rebinding_a_density_to_another_quantity_is_rejected() {
        let mut graph = Graph::<B>::new();
        let first_stoch = stochastic(&mut graph, vec![]);
        let second_stoch = stochastic(&mut graph, vec![]);

        let density = inert(&mut graph, vec![first_stoch]);
        graph.mark_log_density(density, first_stoch).unwrap();

        assert!(matches!(
            graph.mark_log_density(density, second_stoch),
            Err(SkoroError::DuplicateDensity { .. })
        ));

        assert_eq!(graph.log_density_of(first_stoch), Some(density));
        assert_eq!(graph.log_density_of(second_stoch), None);
        assert_eq!(graph.stochastic_of(density), Some(first_stoch));
        graph.validate_indices().unwrap();
    }

    /// Only real-valued mapped nodes can score a model.
    #[test]
    fn only_a_real_mapped_node_can_be_marked() {
        let mut graph = Graph::<B>::new();
        let stoch = stochastic(&mut graph, vec![]);
        let vector = graph
            .add_mapped(
                EuclideanD(3),
                DiffGraphOperation(InertOperation),
                vec![stoch],
            )
            .unwrap();

        // Not real.
        assert!(matches!(
            graph.mark_log_likelihood(vector),
            Err(SkoroError::NotReal { id }) if id == vector
        ));
        assert!(matches!(
            graph.mark_log_density(vector, stoch),
            Err(SkoroError::NotReal { id }) if id == vector
        ));
        // Not mapped at all.
        assert!(matches!(
            graph.mark_log_likelihood(stoch),
            Err(SkoroError::NotMapped { id }) if id == stoch
        ));
        assert!(matches!(
            graph.mark_log_density(stoch, stoch),
            Err(SkoroError::NotMapped { id }) if id == stoch
        ));
        // Not present.
        assert!(matches!(
            graph.mark_log_likelihood(bogus_id()),
            Err(SkoroError::NodeNotFound { .. })
        ));
        assert!(matches!(
            graph.mark_log_density(bogus_id(), stoch),
            Err(SkoroError::NodeNotFound { .. })
        ));
        graph.validate_indices().unwrap();
    }

    /// Marking a likelihood twice is a no-op; a density may not be demoted to a bare likelihood
    /// while its quantity is still in the graph.
    #[test]
    fn a_density_may_not_be_re_marked_as_a_bare_likelihood() {
        let mut graph = Graph::<B>::new();
        let stoch = stochastic(&mut graph, vec![]);
        let factor = inert(&mut graph, vec![stoch]);

        graph.mark_log_likelihood(factor).unwrap();
        graph.mark_log_likelihood(factor).unwrap(); // idempotent
        assert_eq!(graph.log_likelihoods(), &[factor]);

        let density = inert(&mut graph, vec![stoch]);
        graph.mark_log_density(density, stoch).unwrap();
        assert!(matches!(
            graph.mark_log_likelihood(density),
            Err(SkoroError::MustRemainDensity { id }) if id == density
        ));
        graph.validate_indices().unwrap();
    }

    /// Unmarking retires a factor and clears the reverse index with it.
    #[test]
    fn unmarking_returns_a_factor_to_a_plain_mapped_node() {
        let mut graph = Graph::<B>::new();
        let stoch = stochastic(&mut graph, vec![]);
        let density = inert(&mut graph, vec![stoch]);
        graph.mark_log_density(density, stoch).unwrap();

        graph.unmark_log_likelihood(density).unwrap();
        assert!(matches!(graph.node(density), Some(Node::Mapped(_))));
        assert_eq!(graph.log_density_of(stoch), None);
        assert_eq!(graph.stochastic_of(density), None);
        assert!(graph.log_likelihoods().is_empty());
        graph.validate_indices().unwrap();

        // A node that is not a factor cannot be unmarked.
        assert!(matches!(
            graph.unmark_log_likelihood(density),
            Err(SkoroError::NotLogLikelihood { id }) if id == density
        ));
        assert!(matches!(
            graph.unmark_log_likelihood(bogus_id()),
            Err(SkoroError::NodeNotFound { .. })
        ));
    }
}

/// `demote_to_parameter` re-roots a stochastic quantity in place: it keeps its identity and every
/// dependent, and gives up its sampler, its parents and its own density. Demoting a parameter is a
/// no-op and anything else is rejected, so a caller can demote without first asking what it holds.
mod demotion {
    use super::*;

    /// Demotion keeps the node's identity and everything downstream of it, and severs only what
    /// made it stochastic.
    #[test]
    fn a_stochastic_quantity_becomes_a_free_parameter_in_place() {
        let mut graph = Graph::<B>::new();
        let hyper = graph.add_parameter(Real);
        let stoch = stochastic(&mut graph, vec![hyper]);
        let child = inert(&mut graph, vec![stoch]);

        let orphaned = graph.demote_to_parameter(stoch).unwrap();

        assert!(matches!(graph.node(stoch), Some(Node::Parameter(_))));
        // The edge into the old distribution's parameter is gone...
        assert_eq!(graph.children_of(hyper), None);
        assert_eq!(orphaned, vec![hyper]);
        // ...while the edge from the downstream consumer survives.
        assert!(graph.children_of(stoch).unwrap().contains(&child));
        graph.validate_indices().unwrap();
    }

    /// The severed density is reported too, since it is now a factor scoring nothing.
    #[test]
    fn demotion_reports_the_density_it_retires() {
        let mut graph = Graph::<B>::new();
        let stoch = stochastic(&mut graph, vec![]);
        let density = inert(&mut graph, vec![stoch]);
        graph.mark_log_density(density, stoch).unwrap();

        let orphaned = graph.demote_to_parameter(stoch).unwrap();

        assert_eq!(orphaned, vec![density]);
        assert!(matches!(graph.node(density), Some(Node::Mapped(_))));
        assert_eq!(graph.log_density_of(stoch), None);
        graph.validate_indices().unwrap();
    }

    /// Demoting a free variable is a no-op, so a caller need not track which it holds.
    #[test]
    fn demoting_a_parameter_does_nothing() {
        let mut graph = Graph::<B>::new();
        let p = graph.add_parameter(Real);
        assert_eq!(graph.demote_to_parameter(p).unwrap(), Vec::<Id>::new());
        assert!(matches!(graph.node(p), Some(Node::Parameter(_))));
    }

    #[test]
    fn demoting_anything_else_is_rejected() {
        let mut graph = Graph::<B>::new();
        let c = graph.add_constant(Real, &1.0f64);
        assert!(matches!(
            graph.demote_to_parameter(c),
            Err(SkoroError::NotStochastic { id }) if id == c
        ));
        assert!(matches!(
            graph.demote_to_parameter(bogus_id()),
            Err(SkoroError::NodeNotFound { .. })
        ));
    }

    /// Demotion makes a node non-constant again, so every fold below it must be dropped.
    ///
    /// A stale fold is not a wrong *value* — it is a node the tape skips, so the values stay
    /// right while the derivatives silently come back zero.
    #[test]
    fn demotion_invalidates_the_folds_below_it() {
        let mut graph = Graph::<B>::new();
        let c = graph.add_constant(Real, &2.0f64);
        let stoch = stochastic(&mut graph, vec![]);
        graph.observe::<_, Real, _>(stoch, &3.0f64).unwrap();
        let sum = summed(&mut graph, vec![c, stoch]);
        assert!(folded(&graph, sum));

        graph.unobserve(stoch).unwrap();
        graph.demote_to_parameter(stoch).unwrap();
        assert!(!folded(&graph, sum));
        graph.validate_indices().unwrap();
    }
}

/// Binding a quantity to a value, and the fold that runs when one does. Observation is reversible
/// for both node kinds and re-observing replaces rather than accumulates. The fold is the subtle
/// half: it has to reach a descendant whose parents differ in depth, read a cousin from its cache,
/// walk past a stochastic descendant, and stop at one with a free parent.
mod observation {
    use super::*;

    #[test]
    fn observing_binds_a_stochastic_quantity_to_a_value() {
        let mut graph = Graph::<B>::new();
        let stoch = stochastic(&mut graph, vec![]);
        assert!(matches!(graph.node(stoch).unwrap(), Node::Stochastic(_)));

        graph.observe::<_, Real, _>(stoch, &5.0f64).unwrap();

        let Node::Observed(ObservedNode { value, .. }) = graph.node(stoch).unwrap() else {
            panic!("observation did not work");
        };
        let d = into_vec(
            value
                .as_unbatched()
                .expect("observe stores an unbatched value")
                .clone()
                .unwrap_float(),
        );
        assert_eq!(d, vec![5.0]);
        graph.validate_indices().unwrap();
    }

    /// A parameter observes to a constant, and unobserving reverses both mappings.
    #[test]
    fn observation_is_reversible_for_both_node_kinds() {
        let mut graph = Graph::<B>::new();
        let param = graph.add_parameter(Real);
        let stoch = stochastic(&mut graph, vec![]);

        graph.observe::<_, Real, _>(param, &1.0f64).unwrap();
        graph.observe::<_, Real, _>(stoch, &2.0f64).unwrap();
        assert!(matches!(graph.node(param), Some(Node::Constant(_))));
        assert!(matches!(graph.node(stoch), Some(Node::Observed(_))));

        graph.unobserve(param).unwrap();
        graph.unobserve(stoch).unwrap();
        assert!(matches!(graph.node(param), Some(Node::Parameter(_))));
        assert!(matches!(graph.node(stoch), Some(Node::Stochastic(_))));
        graph.validate_indices().unwrap();
    }

    /// Re-observing an already-observed quantity just replaces the value.
    #[test]
    fn re_observing_replaces_the_value() {
        let mut graph = Graph::<B>::new();
        let param = graph.add_parameter(Real);
        graph.observe::<_, Real, _>(param, &1.0f64).unwrap();
        graph.observe::<_, Real, _>(param, &9.0f64).unwrap();

        let Some(Node::Constant(node)) = graph.node(param) else {
            panic!("expected a constant");
        };
        assert_eq!(
            into_vec(node.value.as_unbatched().unwrap().clone().unwrap_float()),
            vec![9.0]
        );
        graph.validate_indices().unwrap();
    }

    /// Re-observing an already-*observed* quantity is the other half of that.
    ///
    /// A parameter observes to a `Constant` and a stochastic quantity to an `Observed`, so the
    /// two re-observations land in different arms. `Observed` is the one that must keep the
    /// node's sampler: the quantity is still stochastic in the model, it is only conditioned on,
    /// and dropping the sampler would silently make it unsamplable.
    #[test]
    fn re_observing_keeps_the_node_observed_and_samplable() {
        let mut graph = Graph::<B>::new();
        let s = stochastic(&mut graph, vec![]);
        graph.observe::<_, Real, _>(s, &1.0f64).unwrap();
        graph.observe::<_, Real, _>(s, &9.0f64).unwrap();

        let Some(Node::Observed(node)) = graph.node(s) else {
            panic!("expected an observed quantity");
        };
        assert_eq!(
            into_vec(node.value.as_unbatched().unwrap().clone().unwrap_float()),
            vec![9.0]
        );
        graph.validate_indices().unwrap();
    }

    /// A cousin that is already concrete supplies its value straight from the arena.
    ///
    /// The gather walks a descendant's parents and reads whichever kind each one is. A `Mapped`
    /// cousin has an `Option` value that may be empty; a `Constant` or `Observed` one always has
    /// a value, and is read through a different arm. A fold that only understood the first arm
    /// would leave this node unfolded and look merely conservative.
    #[test]
    fn a_constant_cousin_is_read_during_the_fold() {
        let mut graph = Graph::<B>::new();
        let c = graph.add_constant(Real, &10.0f64);
        let p = graph.add_parameter(Real);
        let sum = summed(&mut graph, vec![p, c]);

        assert!(!folded(&graph, sum), "a free parent leaves it open");
        graph.observe::<_, Real, _>(p, &1.0f64).unwrap();
        assert!(folded(&graph, sum), "both parents concrete now");

        let Some(Node::Mapped(MappedNode { value: Some(v), .. })) = graph.node(sum) else {
            panic!("expected a folded node");
        };
        assert_eq!(
            into_vec(v.as_unbatched().unwrap().clone().unwrap_float()),
            vec![11.0]
        );
        graph.validate_indices().unwrap();
    }

    /// A value of the wrong width is refused by the packer, before the graph ever sees it.
    ///
    /// `observe` packs through the node's *own* quantity type, so by the time `observe_impl`
    /// compares component counts they cannot disagree -- its `ShapeMismatch` guard is defensive.
    /// The check that actually fires is `Packable::pack`'s, and it panics rather than returning,
    /// which is the internal-invariant convention: supplying a 2-vector for a 3-dimensional
    /// quantity is a caller bug, not a recoverable model error.
    #[test]
    #[should_panic(expected = "Cannot pack vector data of length 2")]
    fn a_value_of_the_wrong_width_panics_in_the_packer() {
        use skoro_riemann::euclidean::VectorD;

        let mut graph = Graph::<B>::new();
        let v = graph.add_parameter(EuclideanD(3));
        let _ = graph.observe::<_, EuclideanD, _>(v, &VectorD::new(&[1.0f64, 2.0]));
    }

    /// Asking to observe with a quantity type the node does not have is recoverable, though.
    #[test]
    fn a_value_of_the_wrong_quantity_type_is_rejected() {
        let mut graph = Graph::<B>::new();
        let v = graph.add_parameter(EuclideanD(3));
        assert!(matches!(
            graph.observe::<_, Real, _>(v, &1.0f64),
            Err(SkoroError::QuantityTypeMismatch { id }) if id == v
        ));
        graph.validate_indices().unwrap();
    }

    #[test]
    fn only_observable_node_kinds_may_be_observed() {
        let mut graph = Graph::<B>::new();
        let p = graph.add_parameter(Real);
        let mapped = inert(&mut graph, vec![p]);

        assert!(matches!(
            graph.observe::<_, Real, _>(mapped, &1.0f64),
            Err(SkoroError::NotObservable { id }) if id == mapped
        ));
        assert!(matches!(
            graph.unobserve(mapped),
            Err(SkoroError::NotObservable { id }) if id == mapped
        ));
        assert!(matches!(
            graph.unobserve(bogus_id()),
            Err(SkoroError::NodeNotFound { .. })
        ));
        graph.validate_indices().unwrap();
    }

    #[test]
    fn observing_an_absent_node_is_reported() {
        let mut graph = Graph::<B>::new();
        assert!(matches!(
            graph.observe::<_, Real, _>(bogus_id(), &1.0f64),
            Err(SkoroError::NodeNotFound { .. })
        ));
    }

    /// Observing must fold every descendant that became concrete, including one whose parents sit
    /// at different depths.
    #[test]
    fn observing_folds_a_descendant_whose_parents_differ_in_depth() {
        let mut graph = Graph::<B>::new();
        let stoch = stochastic(&mut graph, vec![]);

        let near = summed(&mut graph, vec![stoch]);
        let middle = summed(&mut graph, vec![stoch]);
        let far = summed(&mut graph, vec![middle]);
        // Parents at depth 1 and depth 3.
        let join = summed(&mut graph, vec![near, far]);

        // Nothing is concrete while the root is still stochastic.
        for id in [near, middle, far, join] {
            assert!(
                !folded(&graph, id),
                "{id:?} folded before the root was fixed"
            );
        }

        graph.observe::<_, Real, _>(stoch, &3.0f64).unwrap();
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
        graph.validate_indices().unwrap();
    }

    /// A cousin that already folded is read from its cache rather than recomputed.
    ///
    /// The gather distinguishes a `Mapped` cousin -- whose value is an `Option`, empty until it
    /// folds -- from a `Constant` one, which always has a value. Both arms have to work, and
    /// only the `Mapped` one has a state where the right answer is to supply nothing.
    #[test]
    fn a_folded_cousin_is_read_from_its_cache() {
        let mut graph = Graph::<B>::new();
        let c = graph.add_constant(Real, &4.0f64);
        let d = graph.add_constant(Real, &6.0f64);
        let cousin = summed(&mut graph, vec![c, d]);
        assert!(folded(&graph, cousin), "all-constant parents fold at once");

        let p = graph.add_parameter(Real);
        let joined = summed(&mut graph, vec![p, cousin]);
        assert!(!folded(&graph, joined));

        graph.observe::<_, Real, _>(p, &1.0f64).unwrap();
        let Some(Node::Mapped(MappedNode { value: Some(v), .. })) = graph.node(joined) else {
            panic!("expected a folded node");
        };
        assert_eq!(
            into_vec(v.as_unbatched().unwrap().clone().unwrap_float()),
            vec![11.0]
        );
        graph.validate_indices().unwrap();
    }

    /// A stochastic descendant is walked past by both walks rather than folded.
    ///
    /// A sampled quantity has no cached value to gather and no operation to fold: it is a leaf
    /// of the *concrete* part of the graph even though the graph keeps going below it. Both
    /// walks match on node kind, so each needs its own way past this, and a walk that assumed
    /// every descendant was mapped would reach for an operation that is not there.
    #[test]
    fn a_stochastic_descendant_is_walked_past() {
        let mut graph = Graph::<B>::new();
        let p = graph.add_parameter(Real);
        let z = stochastic(&mut graph, vec![p]);
        let y = summed(&mut graph, vec![z]);

        graph.observe::<_, Real, _>(p, &2.0f64).unwrap();
        assert!(!folded(&graph, y), "a sampled ancestor keeps it open");
        graph.validate_indices().unwrap();
    }

    /// A descendant with a still-free parent stays unfolded, even when its other parent settles.
    #[test]
    fn a_descendant_with_a_free_parent_does_not_fold() {
        let mut graph = Graph::<B>::new();
        let stoch = stochastic(&mut graph, vec![]);
        let free = graph.add_parameter(Real);
        let joined = summed(&mut graph, vec![stoch, free]);

        graph.observe::<_, Real, _>(stoch, &3.0f64).unwrap();
        assert!(!folded(&graph, joined));
        graph.validate_indices().unwrap();
    }
}

/// A pinned batch propagates through a fold and is *checked* rather than broadcast: parents at
/// different batches are rejected, an agnostic fold stays agnostic, and a mix runs at the pinned
/// size. The pin is owned by the observation that set it, so unobserving releases it and
/// re-observing the same quantity is not a self-conflict.
mod batch_pinning {
    use super::*;

    /// The elements of a folded mapped node's stored value, whatever its rank.
    ///
    /// Also asserts which variant it is, because that is half of what these tests are about: a
    /// fold over a batch-pinned parent has to *stay* rank-2. Collapsing it to rank-1 would keep
    /// row 0 and silently discard the rest, and nothing downstream would announce that.
    fn folded_rows(graph: &Graph<B>, id: Id, expect_pinned: Option<usize>) -> Vec<f64> {
        let Some(Node::Mapped(MappedNode { value: Some(v), .. })) = graph.node(id) else {
            panic!("expected a folded mapped node");
        };
        assert_eq!(v.pinned_batch(), expect_pinned, "wrong fold representation");
        match v {
            QuantityValue::Batched(input) => into_vec(input.clone().unwrap_float()),
            QuantityValue::Unbatched(c) => into_vec(c.clone().unwrap_float()),
        }
    }

    /// A fold over batch-pinned parents keeps every row.
    ///
    /// The failure this guards against is the tempting one: folding at batch 1 and `unbatch`ing
    /// the result, which is what the walk did before batch-pinned values existed. That path
    /// silently returns row 0 three times.
    #[test]
    fn a_fold_over_batched_parents_keeps_its_rows() {
        let mut graph = Graph::<B>::new();
        let a = graph
            .add_batched_constant(Real, &[1.0f64, 2.0, 3.0])
            .unwrap();
        let b = graph
            .add_batched_constant(Real, &[10.0f64, 20.0, 30.0])
            .unwrap();
        let sum = summed(&mut graph, vec![a, b]);

        assert!(folded(&graph, sum));
        assert_eq!(folded_rows(&graph, sum, Some(3)), vec![11.0, 22.0, 33.0]);
    }

    /// A batch-agnostic parent broadcasts into a fold that a sibling has pinned.
    ///
    /// This is the case that makes the fold batch a property of the *parent set* rather than of
    /// any one parent: the rank-1 value has to expand to 3, not the pinned one collapse to 1.
    #[test]
    fn a_fold_mixing_pinned_and_agnostic_parents_runs_at_the_pinned_batch() {
        let mut graph = Graph::<B>::new();
        let batched = graph
            .add_batched_constant(Real, &[1.0f64, 2.0, 3.0])
            .unwrap();
        let shared = graph.add_constant(Real, &100.0f64);
        let sum = summed(&mut graph, vec![batched, shared]);

        assert_eq!(folded_rows(&graph, sum, Some(3)), vec![101.0, 102.0, 103.0]);
    }

    /// Where nothing is pinned, folding is exactly what it was: rank-1, and reusable at any batch.
    #[test]
    fn a_fold_over_agnostic_parents_stays_agnostic() {
        let mut graph = Graph::<B>::new();
        let a = graph.add_constant(Real, &2.0f64);
        let b = graph.add_constant(Real, &5.0f64);
        let sum = summed(&mut graph, vec![a, b]);

        assert_eq!(folded_rows(&graph, sum, None), vec![7.0]);
    }

    /// A node which pins a *different* batch-size cannot be added.
    #[test]
    fn pinning_different_batches_are_rejected() {
        let mut graph = Graph::<B>::new();
        let _ = graph
            .add_batched_constant(Real, &[1.0f64, 2.0, 3.0])
            .unwrap();

        assert!(matches!(
            graph.add_batched_constant(Real, &[1.0f64, 2.0, 3.0, 4.0, 5.0]),
            Err(SkoroError::BatchSizeMismatch { .. })
        ));

        graph.validate_indices().unwrap();
        assert_eq!(graph.pinned_batch(), Some(3));
        assert_eq!(graph.node_count(), 1);

        let other = graph.add_parameter(Real);
        assert!(matches!(
            graph.observe_batched::<_, Real, _>(other, &[1.0f64, 2.0, 3.0, 4.0, 5.0]),
            Err(SkoroError::BatchSizeMismatch { .. })
        ));
        graph.validate_indices().unwrap();
        assert_eq!(graph.pinned_batch(), Some(3));
        assert_eq!(graph.node_count(), 2);
    }

    /// Observing folds through a batch-pinned parent, and unobserving clears the fold again.
    #[test]
    fn observing_folds_through_a_pinned_parent_and_unobserving_clears_it() {
        let mut graph = Graph::<B>::new();
        let stoch = stochastic(&mut graph, vec![]);
        let batched = graph
            .add_batched_constant(Real, &[1.0f64, 2.0, 3.0])
            .unwrap();
        let sum = summed(&mut graph, vec![stoch, batched]);

        assert!(!folded(&graph, sum));
        assert_eq!(graph.pinned_batch(), Some(3));

        graph.observe::<_, Real, _>(stoch, &10.0f64).unwrap();
        assert_eq!(graph.pinned_batch(), Some(3));
        assert_eq!(folded_rows(&graph, sum, Some(3)), vec![11.0, 12.0, 13.0]);

        graph.unobserve(stoch).unwrap();
        assert!(!folded(&graph, sum));
        graph.validate_indices().unwrap();
    }

    /// Observing a batch of values pins the graph the same way a batched constant does.
    #[test]
    fn observe_batched_pins_the_graph() {
        let mut graph = Graph::<B>::new();
        let p = graph.add_parameter(Real);
        graph
            .observe_batched::<_, Real, _>(p, &[1.0f64, 2.0])
            .unwrap();
        assert_eq!(graph.pinned_batch(), Some(2));
        graph.validate_indices().unwrap();
    }

    /// A second pin that *agrees* is accepted, and the search stops at the first match.
    #[test]
    fn a_matching_second_pin_is_accepted() {
        let mut graph = Graph::<B>::new();
        let _ = graph
            .add_batched_constant(Real, &[1.0f64, 2.0, 3.0])
            .unwrap();
        let p = graph.add_parameter(Real);
        graph
            .observe_batched::<_, Real, _>(p, &[4.0f64, 5.0, 6.0])
            .unwrap();
        assert_eq!(graph.pinned_batch(), Some(3));
        assert_eq!(graph.node_count(), 2);
        graph.validate_indices().unwrap();
    }

    /// The pin an observation checks against may itself be an *observed* quantity.
    ///
    /// `pinnable` mixes constants and observed quantities, and the check reads the value out of
    /// whichever kind it lands on. Conditioning a second quantity on a batch of data is the
    /// ordinary way a model reaches two observed nodes, so this arm is not an edge case -- it is
    /// what happens on the second `observe_batched` of any real fit.
    #[test]
    fn an_observed_quantity_supplies_the_pin_for_the_next_one() {
        let mut graph = Graph::<B>::new();
        let s = stochastic(&mut graph, vec![]);
        graph
            .observe_batched::<_, Real, _>(s, &[1.0f64, 2.0])
            .unwrap();
        assert_eq!(graph.pinned_batch(), Some(2));

        let t = stochastic(&mut graph, vec![]);
        graph
            .observe_batched::<_, Real, _>(t, &[3.0f64, 4.0])
            .unwrap();
        assert_eq!(graph.pinned_batch(), Some(2));

        // A disagreeing one is still refused, with the observed node as the authority.
        let u = graph.add_parameter(Real);
        assert!(matches!(
            graph.observe_batched::<_, Real, _>(u, &[5.0f64, 6.0, 7.0]),
            Err(SkoroError::BatchSizeMismatch {
                expected: 2,
                found: 3
            })
        ));
        graph.validate_indices().unwrap();
    }

    /// A quantity does not conflict with the pin it is itself holding.
    ///
    /// The loop compares the incoming batch against every *other* pinnable node, and here there
    /// is no other one -- replacing a batch of observed data means meeting only yourself.
    /// Without the `!= id` guard every re-observation would be a self-conflict, and a batch of
    /// data could be conditioned on exactly once.
    #[test]
    fn re_observing_the_pinning_quantity_is_not_a_self_conflict() {
        let mut graph = Graph::<B>::new();
        let s = stochastic(&mut graph, vec![]);
        graph
            .observe_batched::<_, Real, _>(s, &[1.0f64, 2.0])
            .unwrap();
        // A batch-agnostic constant holds no pin, so it is walked past rather than compared.
        let _ = graph.add_constant(Real, &0.0f64);

        graph
            .observe_batched::<_, Real, _>(s, &[7.0f64, 8.0])
            .unwrap();
        assert_eq!(graph.pinned_batch(), Some(2));

        // A different batch on the same quantity is still a change of pin, not a conflict.
        graph
            .observe_batched::<_, Real, _>(s, &[7.0f64, 8.0, 9.0])
            .unwrap();
        assert_eq!(graph.pinned_batch(), Some(3));
        graph.validate_indices().unwrap();
    }

    /// An unpinned graph reports no pin, and unobserving releases one.
    #[test]
    fn unobserving_releases_the_pin() {
        let mut graph = Graph::<B>::new();
        assert_eq!(graph.pinned_batch(), None);
        let p = graph.add_parameter(Real);
        graph
            .observe_batched::<_, Real, _>(p, &[1.0f64, 2.0])
            .unwrap();
        assert_eq!(graph.pinned_batch(), Some(2));
        graph.unobserve(p).unwrap();
        assert_eq!(graph.pinned_batch(), None);
        graph.validate_indices().unwrap();
    }
}

/// A stacked tangent builds internal nodes no caller holds a handle for. Only those internals are
/// private -- the members are not -- and retiring the stacked tangent reclaims them, to a fixpoint,
/// while an unrelated removal reclaims nothing. Without this they would be a leak nothing reports.
mod private_nodes {
    use super::*;

    /// Build the two-member stacked tangent every retirement test below starts from.
    ///
    /// Returns the graph, the two scored parameters, the concatenation, and the node count from
    /// before the call — five nodes later: an anchor and a per-block tangent per member, plus the
    /// concatenation.
    fn stacked_tangent_fixture() -> (Graph<B>, Id, Id, Id, usize) {
        let mut graph = Graph::<B>::new();
        let a = graph.add_parameter(Real);
        let b = graph.add_parameter(Real);
        let at = real_point_map([(a, 1.0), (b, -2.0)]);

        let before = graph.node_count();
        let joint = graph.stacked_tangent(&at, &[a, b]).unwrap();
        assert_eq!(graph.node_count() - before, 5);
        (graph, a, b, joint, before)
    }

    /// The concatenation is the caller's; the anchors and per-block tangents are the graph's.
    #[test]
    fn only_the_internals_are_private() {
        let (graph, a, b, joint, _) = stacked_tangent_fixture();
        assert!(!graph.is_private(joint));
        assert!(!graph.is_private(a));
        assert!(!graph.is_private(b));
        assert_eq!(
            graph
                .nodes()
                .filter(|(id, _)| graph.is_private(**id))
                .count(),
            4,
            "an anchor and a per-block tangent per member"
        );
    }

    /// The stacked tangent evaluates to the displacement from each anchor, in block order.
    ///
    /// The anchors are a *snapshot*: a factor linearized at an estimate must not follow the
    /// quantity it summarizes, so moving the members moves the residual rather than zeroing it.
    #[test]
    fn the_stacked_tangent_is_the_displacement_from_the_anchors() {
        let (graph, a, b, joint, _) = stacked_tangent_fixture();
        // Anchored at (1, -2); evaluated at (4, 1) the residual is (3, 3).
        let point = points([(a, 4.0), (b, 1.0)]);
        assert_eq!(
            into_vec(graph.evaluate(&point, joint).unwrap()),
            vec![3.0, 3.0]
        );

        // At the anchors themselves it vanishes.
        let at_anchor = points([(a, 1.0), (b, -2.0)]);
        assert_eq!(
            into_vec(graph.evaluate(&at_anchor, joint).unwrap()),
            vec![0.0, 0.0]
        );
    }

    #[test]
    fn an_empty_member_list_is_rejected() {
        let mut graph = Graph::<B>::new();
        let at = real_point_map([]);
        assert!(matches!(
            graph.stacked_tangent(&at, &[]),
            Err(SkoroError::ShapeMismatch {
                expected: 1,
                found: 0
            })
        ));
    }

    /// Every member needs a linearization point; there is no default anchor.
    #[test]
    fn a_member_without_a_point_is_rejected() {
        let mut graph = Graph::<B>::new();
        let a = graph.add_parameter(Real);
        let at = real_point_map([]);
        assert!(matches!(
            graph.stacked_tangent(&at, &[a]),
            Err(SkoroError::MissingPoint { id }) if id == a
        ));
        graph.validate_indices().unwrap();
    }

    /// Retiring a joint residual reclaims the `2k` nodes `stacked_tangent` built for itself.
    ///
    /// Those nodes are unreachable by every other route. They sit *upstream* of the
    /// concatenation, so a downstream walk cannot find them, and no handle names them, so an
    /// ordinary cleanup never queues them. Without the private-node sweep a sliding window leaks
    /// `2k` nodes per tick.
    ///
    /// Rooted at a **scored quantity**, which is what a window actually retires. Note that the
    /// factor body is already downstream of it, so what the sweep adds is the surviving member's
    /// half.
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
    /// gone — and the tangent is only reclaimed in the pass before. A single pass over the private
    /// set would strand one anchor, always or never depending on `HashSet` iteration order. This
    /// is the case that catches it: rooting at the concatenation makes *both* per-block tangents
    /// reclaimable in pass one and both anchors only in pass two.
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
    /// removal elsewhere in the model must leave the whole subgraph standing. Without this the
    /// sweep would be free to eat a live factor the first time anything else was retired.
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
}

/// Paths that only a *likelihood factor* reaches.
///
/// `Mapped` and `LogLikelihood` are the same shape and are matched together almost everywhere,
/// which means a walk written for one and never exercised with the other can quietly handle only
/// half its cases. These put a factor in each of those walks.
mod factors_in_the_walks {
    use super::*;

    /// Observing folds a *density* just as it folds a plain mapped node.
    ///
    /// A frozen factor is the whole point of observation: once its quantity is fixed, its score
    /// is a constant and every later pass should read the cache rather than re-run it.
    #[test]
    fn observing_folds_a_likelihood_factor_below_it() {
        let mut graph = Graph::<B>::new();
        let stoch = stochastic(&mut graph, vec![]);
        let factor = summed(&mut graph, vec![stoch]);
        graph.mark_log_density(factor, stoch).unwrap();
        // A plain node downstream of the factor, so the factor is also a *cousin* in the walk.
        let downstream = summed(&mut graph, vec![factor]);

        assert!(!folded(&graph, factor));
        graph.observe::<_, Real, _>(stoch, &3.0f64).unwrap();

        // Both are concrete now, and the factor is folded even though it is a `LogLikelihood`.
        let Some(Node::LogLikelihood(node)) = graph.node(factor) else {
            panic!("expected a log-likelihood node");
        };
        assert!(node.value.is_some(), "an observed factor folds");
        assert!(folded(&graph, downstream));

        // And unobserving clears the factor's fold as well as the plain node's.
        graph.unobserve(stoch).unwrap();
        let Some(Node::LogLikelihood(node)) = graph.node(factor) else {
            panic!("expected a log-likelihood node");
        };
        assert!(node.value.is_none(), "a factor's stale fold is cleared too");
        assert!(!folded(&graph, downstream));
        graph.validate_indices().unwrap();
    }

    /// A folded factor is also a folded *parent*, so a node built on top of it folds at
    /// construction rather than being treated as stochastic.
    #[test]
    fn a_folded_factor_makes_its_children_foldable() {
        let mut graph = Graph::<B>::new();
        let c = graph.add_constant(Real, &2.0f64);
        let factor = summed(&mut graph, vec![c]);
        graph.mark_log_likelihood(factor).unwrap();

        let downstream = summed(&mut graph, vec![factor]);
        assert!(folded(&graph, downstream));
        graph.validate_indices().unwrap();
    }
}

/// Degenerate shapes, where a walk has nothing to walk.
mod degenerate {
    use super::*;

    /// Observing a quantity with no children folds nothing, and does not stumble looking.
    #[test]
    fn observing_a_childless_quantity_is_a_no_op_downstream() {
        let mut graph = Graph::<B>::new();
        let stoch = stochastic(&mut graph, vec![]);
        graph.observe::<_, Real, _>(stoch, &1.0f64).unwrap();
        graph.unobserve(stoch).unwrap();
        assert!(matches!(graph.node(stoch), Some(Node::Stochastic(_))));
        graph.validate_indices().unwrap();
    }

    /// Demoting a childless quantity likewise has no folds to invalidate.
    #[test]
    fn demoting_a_childless_quantity_clears_nothing() {
        let mut graph = Graph::<B>::new();
        let stoch = stochastic(&mut graph, vec![]);
        assert!(graph.demote_to_parameter(stoch).unwrap().is_empty());
        graph.validate_indices().unwrap();
    }

    /// An empty graph sorts to an empty order, and an unknown target contributes nothing.
    #[test]
    fn sorting_an_empty_or_unknown_target_yields_nothing_extra() {
        let graph = Graph::<B>::new();
        assert!(graph.topological_sort(&[]).is_empty());
        // An id the graph never issued has no parents to walk, but is still ordered.
        assert_eq!(graph.topological_sort(&[bogus_id()]), vec![bogus_id()]);
    }
}

mod bindings {
    use super::*;
    use skoro_graph::binding::Role;

    const R1: Role = Role("r1");
    const R2: Role = Role("r2");

    #[test]
    fn bind_refuses_a_missing_endpoint() {
        let mut graph = Graph::<B>::new();
        let a = graph.add_parameter(Real);
        let fresh = bogus_id();

        assert!(matches!(
            graph.bind(a, R1, fresh),
            Err(SkoroError::NodeNotFound { id }) if id == fresh
        ));
        assert!(matches!(
            graph.bind(fresh, R1, a),
            Err(SkoroError::NodeNotFound { id }) if id == fresh
        ));
        graph.validate_indices().unwrap();
    }

    #[test]
    fn bind_is_readable_in_order() {
        let mut graph = Graph::<B>::new();
        let a = graph.add_parameter(Real);
        let b = graph.add_parameter(Real);
        let c = graph.add_parameter(Real);

        graph.bind(a, R1, b).unwrap();
        graph.bind(a, R2, c).unwrap();
        graph.bind(a, R1, c).unwrap();

        assert_eq!(graph.bindings_of(a), &[(R1, b), (R2, c), (R1, c)]);
        assert_eq!(graph.bound(a, R1), vec![b, c]);
        assert_eq!(graph.referrers_of(b), vec![a]);
        graph.validate_indices().unwrap();
    }

    #[test]
    fn remove_refuses_a_bound_target() {
        let mut graph = Graph::<B>::new();
        let a = graph.add_parameter(Real);
        let b = graph.add_parameter(Real);
        graph.bind(a, R1, b).unwrap();

        assert!(matches!(
            graph.remove(&[b]),
            Err(SkoroError::NodeStillBound { id, by }) if id == b && by == a
        ));
        graph.remove(&[a, b]).unwrap();
        graph.validate_indices().unwrap();
    }

    #[test]
    fn remove_source_clears_the_reference() {
        let mut graph = Graph::<B>::new();
        let a = graph.add_parameter(Real);
        let b = graph.add_parameter(Real);
        graph.bind(a, R1, b).unwrap();

        graph.remove(&[a]).unwrap();
        assert!(graph.referrers_of(b).is_empty());
        graph.remove(&[b]).unwrap();
        graph.validate_indices().unwrap();
    }

    #[test]
    fn remove_closure_takes_referrers() {
        let mut graph = Graph::<B>::new();
        let a = graph.add_parameter(Real);
        let b = graph.add_parameter(Real);
        graph.bind(a, R1, b).unwrap();

        let removed = graph.remove_closure(&[b]).unwrap();
        assert_eq!(removed.len(), 2, "b and its referrer a");
        assert!(!graph.contains(a));
        assert!(!graph.contains(b));
        graph.validate_indices().unwrap();
    }

    #[test]
    fn validate_indices_holds_after_binding_mutations() {
        let mut graph = Graph::<B>::new();
        let a = graph.add_parameter(Real);
        let b = graph.add_parameter(Real);
        let c = graph.add_parameter(Real);
        graph.bind(a, R1, b).unwrap();
        graph.bind(a, R2, c).unwrap();
        graph.bind(a, R1, c).unwrap();
        graph.validate_indices().unwrap();

        graph.remove(&[a, b, c]).unwrap();
        graph.validate_indices().unwrap();
    }

    #[test]
    fn terms_classify_proposal_and_weight() {
        let mut graph = Graph::<B>::new();
        let stoch = stochastic(&mut graph, vec![]);
        // `summed`, not `inert`: this density is folded below, once its quantity is observed.
        let density = summed(&mut graph, vec![stoch]);
        graph.mark_log_density(density, stoch).unwrap();

        let obs = stochastic(&mut graph, vec![]);
        let obs_density = summed(&mut graph, vec![obs]);
        graph.mark_log_density(obs_density, obs).unwrap();
        graph.observe::<_, Real, _>(obs, &1.0f64).unwrap();

        let free_standing = graph.add_parameter(Real);
        let free_density = inert(&mut graph, vec![free_standing]);
        graph.mark_log_likelihood(free_density).unwrap();

        let terms = graph.terms();
        assert_eq!(
            terms.iter().map(|t| t.density).collect::<Vec<_>>(),
            vec![density, obs_density, free_density],
            "marking order"
        );
        assert_eq!(
            terms.iter().map(|t| t.kind).collect::<Vec<_>>(),
            vec![TermKind::Proposal, TermKind::Weight, TermKind::Weight]
        );

        graph.observe::<_, Real, _>(stoch, &2.0f64).unwrap();
        assert_eq!(graph.terms()[0].kind, TermKind::Weight);
        graph.validate_indices().unwrap();
    }

    #[test]
    fn log_density_terms_sums_only_the_named_terms() {
        let mut graph = Graph::<B>::new();
        let stoch = stochastic(&mut graph, vec![]);
        // `summed`, not `inert`: this density is actually evaluated below via `log_density`.
        let t1 = summed(&mut graph, vec![stoch]);
        graph.mark_log_density(t1, stoch).unwrap();
        let free_standing = graph.add_parameter(Real);
        let t2 = summed(&mut graph, vec![free_standing]);
        graph.mark_log_likelihood(t2).unwrap();

        // `stoch` is a leaf the compiled program reads from `points`, so it needs a value too.
        let points = points([(free_standing, 3.0), (stoch, 0.5)]);
        let whole = graph.log_density(&points).unwrap();
        let split = (into_vec(graph.log_density_terms(&points, &[t1]).unwrap())[0]
            + into_vec(graph.log_density_terms(&points, &[t2]).unwrap())[0])
            - into_vec(whole)[0];
        approx::assert_abs_diff_eq!(split, 0.0, epsilon = 1e-12);

        let mapped = inert(&mut graph, vec![free_standing]);
        assert!(matches!(
            graph.log_density_terms(&points, &[mapped]),
            Err(SkoroError::NotLogLikelihood { id }) if id == mapped
        ));
    }
}
