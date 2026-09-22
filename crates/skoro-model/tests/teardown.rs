//! Retiring a cone of the model: `remove_closure` from the client side.
//!
//! `ModelBuilderOps::remove` is leaf-only and all-or-nothing — it refuses a batch any survivor
//! still points into — so retiring a state means computing the downward closure by hand at every
//! call site. `ModelBuilderOps::remove_closure` is that walk, done once.
//!
//! Two things are checked here that are not checked at the graph layer, because neither is
//! visible from a `Graph`:
//!
//! - removal is reachable through the **builder** trait and not the read one, which is what
//!   keeps an estimator that holds only `ModelReaderOps` from deleting any of its own evidence. The
//!   negative half is a `trybuild` case, since a test that compiles cannot assert that another
//!   does not;
//! - `cleanup()` survives it. A closure removal deliberately takes nodes the caller *does* hold a
//!   `Q` for, so dropping that handle afterwards leaves a handle naming nothing — and cleanup used
//!   to `expect` on exactly that.
//!
//! The second half of the file is the collector itself: what a mark-and-sweep reclaims that the
//! previous candidate-queue collector could not, and the one mark edge that is not a parent edge.

use skoro_backend::{Flex, Point};
use skoro_core::{error::SkoroResult, id::Id, kind::Continuous};
use skoro_model::{
    ModelBuilderOps, ModelReaderOps,
    distribution::{Distribution, Factor, Sampler},
    map::{register_map_slice, register_map1},
    model::ModelClient,
    q::Q,
};
use skoro_operation::{
    Input,
    ops::{LiteralScalarMultiplication, TangentConcat},
};
use skoro_riemann::{euclidean::EuclideanD, real::Real};

type B = Flex;
skoro_backend::f64_test_binary!(Flex);

/// A mapped node over `blocks`, of the summed width. Any operation would do; `TangentConcat` is
/// the one with no arity limit.
fn concat(blocks: &[&Q<B, EuclideanD>]) -> Q<B, EuclideanD> {
    let widths = vec![1; blocks.len()];
    register_map_slice(EuclideanD(blocks.len()), TangentConcat(widths), blocks).unwrap()
}

fn node_count(model: &ModelClient<B>) -> usize {
    model.server_call(|server| server.graph().node_count())
}

#[derive(Debug, Clone)]
struct LazySampler;

impl Sampler<B, Continuous> for LazySampler {
    fn sample(&self, _batch_size: usize, _parameters: Vec<Input<B>>) -> Point<B> {
        unimplemented!("the test never generates")
    }
}

/// `remove_closure` takes the whole downward cone where the leaf-only `remove` would have refused,
/// and is idempotent -- a sliding window retires by policy, so two policies may name overlapping
/// cones on the same tick. Dropping a `Q` for a node the closure already took must not panic.
mod closures {
    use super::*;

    /// The closure takes everything downstream, where `remove` would have refused.
    #[test]
    fn remove_closure_takes_the_cone_a_leaf_removal_refuses() {
        let model = ModelClient::<B>::default();
        let a = model.add_parameter(EuclideanD(1));
        let b = model.add_parameter(EuclideanD(1));
        let ab = concat(&[&a, &b]);
        let top = concat(&[&ab]);

        // `remove` is the primitive that will not do this: `a` still has a child.
        assert!(ModelBuilderOps::remove(&model, &[a.id()]).is_err());

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
}

/// The shapes that used to break the drop-guard protocol. Nothing in the `Distribution` contract
/// says a density must be expressed over its quantity's parents, so a distribution that skips them
/// is legal -- and under the old `is_mapped_leaf` flags it was fatal. Demotion reclaims a severed
/// parent the orphaned density never named.
mod cleanup {
    use super::*;

    /// A distribution whose `factor` does **not** rebuild over its parents — the one shape
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

        fn factor(&self, x: &Q<B, Real>) -> SkoroResult<Factor<B>> {
            Ok(Factor::new(register_map1(
                Real,
                LiteralScalarMultiplication::new(2.0),
                x,
            )?))
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

    /// Demotion reclaims a severed parent the orphaned density never named.
    ///
    /// This used to be the test that `demote_to_parameter` *reports* what it severed, because a
    /// collector driven by a candidate queue could only examine ids something had handed it. For
    /// an ordinary `Normal` the report was redundant — its `log_density` is built over the same
    /// mean the stochastic node takes as a parent, so collecting the orphaned density cascaded
    /// down to it anyway. `LazyDistr` breaks that coincidence: its `mean` is a parent of the
    /// quantity and of nothing else, so no cascade from the density ever reaches it.
    ///
    /// Under mark-and-sweep nothing needs to report anything. The demote severs the edge, and
    /// the next mark simply does not reach `mean`.
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
}

/// What the mark-and-sweep reclaims that a candidate-queue collector could not. A `Q` is the
/// ownership token and a bare `Id` a weak reference, so reachability decides: an unreachable
/// constant is garbage, a held one is not, and a mutation that orphans a chain reclaims it without
/// reporting anything. The mark also follows `stochastic_of` out of every likelihood term, or a
/// scored quantity whose density never reads it would be swept while still being scored.
mod reachability {
    use super::*;

    /// A constant nothing reaches is garbage; one a handle names is not.
    ///
    /// The case the previous collector could not reach at all: it spared every `Constant` and
    /// `Parameter` unconditionally, on the grounds that a caller might still hold a bare `Id` for
    /// one. That made every temporary constant a permanent leak. The contract now is that a `Q`
    /// is the ownership token and a bare `Id` is a weak reference, so reachability decides.
    #[test]
    fn an_unreachable_constant_is_collected_and_a_held_one_is_not() {
        let model = ModelClient::<B>::default();
        let held = model.add_constant(Real, &1.0f64);
        let dropped = model.add_constant(Real, &2.0f64);
        let dropped_id = dropped.id();
        assert_eq!(node_count(&model), 2);

        drop(dropped);
        model.cleanup();

        assert_eq!(node_count(&model), 1);
        assert!(!model.server_call(|server| server.graph().contains(dropped_id)));
        assert!(
            model.server_call(|server| server.graph().contains(held.id())),
            "a live handle is a root"
        );
    }

    /// A mutation that orphans a subgraph reclaims it without reporting anything.
    ///
    /// The property the candidate-queue collector could not have. Its documented invariant was
    /// that every mutation reducing a node's reachability must *offer* what it orphaned, or the
    /// node was unreachable and uncollected forever — an obligation spread over five call sites
    /// and enforced by nothing. Here the whole chain below `top` is orphaned by a removal that
    /// names only `top`, and no part of the model says so.
    #[test]
    fn a_removal_reclaims_the_chain_it_orphaned_with_no_report() {
        let model = ModelClient::<B>::default();
        let root = model.add_parameter(EuclideanD(1));

        // A chain of mapped nodes whose only handle is the last one.
        let a = concat(&[&root]);
        let b = concat(&[&a]);
        let top = concat(&[&b]);
        let (a_id, b_id, top_id) = (a.id(), b.id(), top.id());
        drop((a, b));
        assert_eq!(node_count(&model), 4);

        // Taking the tip orphans the whole chain between it and `root`, which is still held.
        ModelBuilderOps::remove(&model, &[top_id]).unwrap();

        assert_eq!(node_count(&model), 1, "only the held root survives");
        for id in [a_id, b_id, top_id] {
            assert!(!model.server_call(|server| server.graph().contains(id)));
        }
        assert!(model.server_call(|server| server.graph().contains(root.id())));
    }

    /// A distribution whose `factor` never reads its own quantity.
    ///
    /// Nothing in the `Distribution` contract requires a density to be built over the quantity it
    /// scores, so the density's parents are not a reliable route to `x`. If the mark followed
    /// parent edges alone, dropping `x`'s handle would leave it unmarked, the sweep would take
    /// it, and `Graph::remove` would quietly retire the live scoring density that named it —
    /// changing the objective inside a garbage collection, with no error anywhere. The mark
    /// therefore follows `stochastic_of` out of every likelihood term as well.
    struct DetachedDistr {
        /// A density built over this constant instead of over `x`.
        #[allow(dead_code)]
        elsewhere: Q<B, Real>,
        ids: [Id; 1],
    }

    impl Distribution<B, Continuous> for DetachedDistr {
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

        fn factor(&self, _x: &Q<B, Real>) -> SkoroResult<Factor<B>> {
            Ok(Factor::new(register_map1(
                Real,
                LiteralScalarMultiplication::new(2.0),
                &self.elsewhere,
            )?))
        }
    }

    #[test]
    fn a_scored_quantity_survives_a_density_that_never_reads_it() {
        let model = ModelClient::<B>::default();
        let c = model.add_constant(Real, &2.0f64);
        let x = ModelBuilderOps::add_quantity(
            &model,
            DetachedDistr {
                elsewhere: c.clone(),
                ids: [c.id()],
            },
        )
        .unwrap();
        let x_id = x.id();
        let density = model
            .log_density_of(&x)
            .expect("the distribution's density");

        // The only handle to `x` goes, and `x` is not a parent of its own density.
        drop(x);
        model.cleanup();

        assert!(
            model.server_call(|server| server.graph().contains(x_id)),
            "a quantity a live term scores is reachable, however its density is written"
        );
        assert!(
            model.server_call(|server| server.graph().is_scoring(density)),
            "and the term is still in the objective"
        );
    }

    /// A proposal-only term is a collector root exactly as a scored one is.
    ///
    /// This is what the single term arena buys: `cleanup`'s roots are
    /// `graph.log_likelihoods()`, which holds every term whether or not it is a summand of the
    /// log-density, so a proposal survives a sweep and takes the quantity it scores with it. Had
    /// proposal-only terms been kept in an arena of their own, they would be unrooted here and
    /// the sweep would take them silently.
    #[test]
    fn a_proposal_only_term_is_a_collector_root() {
        let model = ModelClient::<B>::default();
        let c = model.add_constant(Real, &2.0f64);
        let z = ModelBuilderOps::add_proposal(
            &model,
            DetachedDistr {
                elsewhere: c.clone(),
                ids: [c.id()],
            },
        )
        .unwrap();
        let z_id = z.id();
        let proposal = model.log_density_of(&z).expect("the proposal's density");
        assert!(model.server_call(|server| !server.graph().in_log_density(proposal)));

        // The only handle to the proposed quantity goes, and — as in the test above — the
        // proposal's density is not written over it.
        drop(z);
        model.cleanup();

        assert!(
            model.server_call(|server| server.graph().contains(z_id)),
            "a quantity a live proposal generates is reachable"
        );
        assert!(
            model.server_call(|server| server.graph().is_scoring(proposal)),
            "and the proposal is still a term"
        );
        assert!(
            model.server_call(|server| !server.graph().in_log_density(proposal)),
            "still out of the log-density"
        );
        model
            .server_call(|server| server.graph().validate_indices())
            .expect("the sweep leaves the indices coherent");
    }

    /// Demotion through the *client* runs a `cleanup()` of its own, which the graph-layer test of
    /// the same case does not exercise. Retiring the proposal leaves an ordinary mapped node
    /// behind, and the sweep must not trip over it.
    #[test]
    fn demoting_a_proposed_quantity_through_the_client_retires_its_term() {
        let model = ModelClient::<B>::default();
        let c = model.add_constant(Real, &2.0f64);
        let z = ModelBuilderOps::add_proposal(
            &model,
            DetachedDistr {
                elsewhere: c.clone(),
                ids: [c.id()],
            },
        )
        .unwrap();
        let proposal = model.log_density_of(&z).expect("the proposal's density");

        model.demote_to_parameter(&z).unwrap();

        assert!(model.server_call(|server| !server.graph().is_scoring(proposal)));
        assert!(model.terms().is_empty(), "no term is left");
        assert_eq!(model.log_density_of(&z), None);
        model
            .server_call(|server| server.graph().validate_indices())
            .expect("demotion plus cleanup leaves the indices coherent");
    }
}

/// *When* the collector runs, which is the one thing a caller has to know about it. Dropping a
/// handle does not itself collect -- a sweep happens at `cleanup()` and at the mutations that
/// reduce reachability, and nowhere else -- so a workload that only grows holds its garbage until
/// someone asks. A sweep with nothing to do is skipped rather than walked.
mod sweeping {
    use super::*;

    /// Dropping a handle does not itself collect: a sweep runs at `cleanup()` and at the
    /// mutations that reduce reachability, and nowhere else.
    ///
    /// This is the one behaviour a caller has to know about the collector, and it is a
    /// deliberate trade. Node construction used to sweep before every `add_*`, which made a
    /// model of `n` nodes cost `O(n²)` to build. Nothing is swept on the way in now, so a
    /// workload that only *grows* — building temporaries and dropping them, never removing or
    /// demoting anything — holds its garbage until someone asks. `graph_node_count` is the bound
    /// such a caller watches; `cleanup()` is what it calls.
    #[test]
    fn garbage_is_held_until_a_sweep_is_asked_for() {
        let model = ModelClient::<B>::default();
        let root = model.add_parameter(EuclideanD(1));

        for _ in 0..4 {
            // Each temporary is unreachable the moment it drops: nothing else names it.
            drop(concat(&[&root]));
        }
        assert_eq!(
            node_count(&model),
            5,
            "building never sweeps, so the temporaries are still here"
        );

        model.cleanup();
        assert_eq!(
            node_count(&model),
            1,
            "and the sweep takes all four at once"
        );
    }

    /// A sweep with nothing to do is skipped rather than walked.
    ///
    /// The dirty flag can only ever cause *extra* work, never less — it is set by every dropped
    /// handle and by every mutation that reaches `cleanup` — so this is an optimization rather
    /// than a correctness claim. It matters because `cleanup()` is `O(V + E)` and a caller
    /// polling it between ticks should not pay for the walk.
    #[test]
    fn a_clean_model_skips_the_sweep() {
        let model = ModelClient::<B>::default();
        let root = model.add_parameter(EuclideanD(1));
        let held = concat(&[&root]);

        model.cleanup();
        let after_first = node_count(&model);
        model.cleanup();
        assert_eq!(node_count(&model), after_first);
        assert_eq!(after_first, 2, "nothing was garbage to begin with");
        drop(held);
    }
}
