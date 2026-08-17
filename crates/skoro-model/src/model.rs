use dashmap::DashMap;
use lazy_static::lazy_static;
use std::any::Any;
use std::collections::{HashMap, HashSet};
use std::marker::PhantomData;
use std::sync::{Arc, RwLock, Weak};

use skoro_backend::{Backend, CotangentFrame, Point, TangentFrame, TangentVector};
use skoro_core::kind::Kind;
use skoro_core::{
    error::{SkoroError, SkoroResult},
    id::{Id, IdGenerator},
    pack::Packable,
    sampler::Sampler,
};
use skoro_graph::{
    compiler::{ForwardOutput, HvpForwardOutput, JvpForwardOutput, VjpForwardOutput},
    graph::{
        EvalOrder1, EvalOrder2, Graph, PointEvalOrder1, PointEvalOrder2, PointHvpEval,
        PointJvpEval, PointVjpEval,
    },
    node::Node,
    operation::GraphOperation,
    point_map::PointMap,
    quantity::{ContinuousQuantity, QuantityType},
};
use skoro_operation::ValueMap;
use skoro_riemann::euclidean::EuclideanD;
use skoro_riemann::real::Real;

use crate::structure::{JointWhitenedResidual, ResidualCertificate, WhitenedResidual};
use crate::{cleanup::CleanupRegistry, distribution::Distribution, q::Q};

lazy_static! {
    /// Global Id generator of model servers.
    static ref SERVER_GENERATOR: IdGenerator = IdGenerator::new();
    /// Global registry of model servers, holding `Weak` references so that a
    /// server is reclaimed as soon as the last `ModelClient` is dropped. The
    /// `Id` key lets clients identify servers across FFI boundaries.
    static ref MODEL_SERVERS: DashMap<Id, Weak<dyn Any + Send + Sync>> = DashMap::new();
}

/// The stateful engine wrapping the probabilistic DAG.
#[derive(Default)]
pub struct ModelServer<B: Backend> {
    graph: Graph<B>,
    pub device: B::Device,
    cleanup_registry: Arc<CleanupRegistry>,
    /// The evidence that every factor this server holds is least-squares shaped,
    /// pruned against the graph inside every mutating method below — so a record and
    /// the factor it names cannot drift apart between calls, only within one, where
    /// `prune_certificate` is the last thing that runs.
    ///
    /// It is also the pin table: the nodes it names are the ones neither the collector
    /// may reclaim nor [`remove`](Self::remove) may delete. That is one set, owned by
    /// [`ResidualCertificate`] and asked through
    /// [`names`](ResidualCertificate::names), rather than a copy kept here — a
    /// second owner would be a second thing to keep in step.
    certificate: ResidualCertificate,
}

impl<B: Backend> std::fmt::Debug for ModelServer<B> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ModelServer")
            .field("graph", &self.graph)
            .field("device", &self.device)
            .finish_non_exhaustive()
    }
}

impl<B: Backend> ModelServer<B> {
    /// Create a new model server.
    pub fn new(device: B::Device) -> Self {
        Self {
            graph: Graph::new(),
            device,
            cleanup_registry: Arc::new(CleanupRegistry::default()),
            certificate: ResidualCertificate::new(),
        }
    }

    /// Read-only access to the graph.
    ///
    /// Structural mutation goes through the methods on this type, which maintain the
    /// graph's cross-collection invariants; handing out `&mut Graph` would not.
    #[inline(always)]
    pub fn graph(&self) -> &Graph<B> {
        &self.graph
    }

    /// Add a fixed, immutable quantity to the graph of the model server,
    /// continuous or discrete.
    #[inline(always)]
    pub fn add_constant<K, T, D>(&mut self, quantity_type: T, value: &D) -> Id
    where
        K: Kind<B>,
        T: QuantityType<B, K> + Packable<K::Elem, D>,
        D: 'static,
    {
        self.graph.add_constant(quantity_type, value)
    }

    /// Add a fixed, immutable quantity that differs across the batch.
    ///
    /// See [`Graph::add_batched_constant`](skoro_graph::graph::Graph::add_batched_constant).
    #[inline(always)]
    pub fn add_batched_constant<K, T, D>(&mut self, quantity_type: T, values: &[D]) -> Id
    where
        K: Kind<B>,
        T: QuantityType<B, K> + Packable<K::Elem, D>,
        D: 'static,
    {
        self.graph.add_batched_constant(quantity_type, values)
    }

    /// Add a free quantity to the graph of the model server, continuous or
    /// discrete.
    #[inline(always)]
    pub fn add_parameter<K, T>(&mut self, quantity_type: T) -> Id
    where
        K: Kind<B>,
        T: QuantityType<B, K>,
    {
        self.graph.add_parameter(quantity_type)
    }

    /// Adds a stochastic quantity to the graph of the model server, continuous
    /// or discrete.
    #[inline(always)]
    pub fn add_stochastic<K, D>(&mut self, distr: &D) -> SkoroResult<Id>
    where
        K: Kind<B>,
        D: Distribution<B, K>,
    {
        self.graph.add_stochastic(
            distr.quantity_type(),
            distr.sampler(),
            distr.parent_ids().to_vec(),
        )
    }

    /// Add a transformation of other quantities to the graph of the model server.
    #[inline(always)]
    fn add_mapped<K, T>(
        &mut self,
        quantity_type: T,
        op: impl GraphOperation<B>,
        parent_ids: Vec<Id>,
    ) -> SkoroResult<Id>
    where
        K: Kind<B>,
        T: QuantityType<B, K>,
    {
        self.graph.add_mapped(quantity_type, op, parent_ids)
    }

    /// The stacked tangent of several quantities against frozen anchors.
    ///
    /// See [`Graph::stacked_tangent`](skoro_graph::graph::Graph::stacked_tangent).
    #[inline(always)]
    fn stacked_tangent(&mut self, at: &PointMap<B>, ids: &[Id]) -> SkoroResult<Id> {
        self.graph.stacked_tangent(at, ids)
    }

    /// Mark a mapped quantity as a density for the stochastic quantity.
    pub fn mark_density<T>(
        &mut self,
        density: &Q<B, Real>,
        stochastic: &Q<B, T>,
    ) -> SkoroResult<()> {
        self.graph.mark_density(density.id, stochastic.id)
    }

    /// Mark a mapped quantity as a likelihood.
    pub fn mark_likelihood(&mut self, likelihood: &Q<B, Real>) -> SkoroResult<()> {
        self.graph.mark_likelihood(likelihood.id)
    }

    /// Impose a posterior by observing a stochastic quantity at a specified value.
    #[inline(always)]
    pub fn observe<K, T, D>(&mut self, tag: &Q<B, T>, value: &D) -> SkoroResult<()>
    where
        K: Kind<B>,
        T: QuantityType<B, K> + Packable<K::Elem, D>,
        D: 'static,
    {
        self.graph.observe::<K, T, D>(tag.id, value)
    }

    /// Undo the effects of [`observe`](Self::observe).
    #[inline(always)]
    pub fn unobserve<T>(&mut self, tag: &Q<B, T>) -> SkoroResult<()> {
        self.graph.unobserve(tag.id)
    }

    /// Update a stochastic node's sampling and density mechanisms.
    #[inline(always)]
    pub fn update_stochastic<K, T, S>(
        &mut self,
        tag: &Q<B, T>,
        new_sampler: S,
        new_parents: Vec<Id>,
    ) -> SkoroResult<()>
    where
        K: Kind<B>,
        S: Sampler<B, K>,
    {
        let severed = self
            .graph
            .update_stochastic::<K, S>(tag.id, new_sampler, new_parents)?;
        self.cleanup_registry.offer(severed);
        self.prune_certificate();
        self.cleanup();
        Ok(())
    }

    /// Demote a stochastic quantity to a free parameter, keeping its identity.
    ///
    /// The quantity loses its sampler, its parents, and its own log-density; its
    /// [`Id`], quantity type, and every dependent quantity are untouched. Use this
    /// when a quantity's distribution is superseded by a joint likelihood factor
    /// over several quantities — a marginalization whose result does not factorize.
    ///
    /// The demoted quantity's log-density leaves the objective here, so a certificate
    /// record naming it leaves too: `prune_certificate` runs in the same call, and the
    /// rows of `J` shrink exactly when the term stops scoring.
    ///
    /// Everything the demotion made unreachable is collected before returning — the orphaned
    /// density, the subgraph it was holding up, the now-uncertified residual, and the parent
    /// edges the old distribution owned. The graph reports each of those, because none of them
    /// is an event a later pass could reconstruct.
    ///
    /// Returns an error if the quantity does not exist or is neither stochastic nor
    /// already a parameter.
    #[inline(always)]
    pub fn demote_to_parameter<T>(&mut self, tag: &Q<B, T>) -> SkoroResult<()> {
        let severed = self.graph.demote_to_parameter(tag.id)?;
        self.cleanup_registry.offer(severed);
        self.prune_certificate();
        self.cleanup();
        Ok(())
    }

    /// Removes quantities, verifying they have no children which would be kept in the graph.
    ///
    /// Refuses a node the certificate names, for the same reason it refuses one another node
    /// still points into: this is the primitive that will not break a reference, and a record
    /// is a reference. That refusal is what keeps `remove` safe on
    /// [`ModelClientOps`](crate::client_ops::ModelClientOps) — an estimator bounded on
    /// [`NlsStructure`](crate::structure::NlsStructure) holds it, and a certified residual is
    /// childless, so without the check a `remove` could silently take a factor out of the
    /// objective. Retiring a factor is [`remove_closure`](Self::remove_closure), one tier up.
    ///
    /// The removal is also a collection point: anything it orphaned unwinds in the
    /// cascade before returning, so the model is clean after every retire.
    pub fn remove(&mut self, ids: &[Id]) -> SkoroResult<Vec<Node<B>>> {
        if let Some(id) = ids.iter().find(|id| self.certificate.names(**id)) {
            return Err(SkoroError::NodeIsCertified { id: *id });
        }
        let removed = self.graph.remove(ids)?;
        self.prune_certificate();
        self.offer_parents(&removed);
        self.cleanup();
        Ok(removed)
    }

    /// Remove `roots` together with everything downstream of them.
    ///
    /// See [`Graph::remove_closure`](skoro_graph::graph::Graph::remove_closure). Unlike
    /// [`remove`](Self::remove) this cannot fail on a still-referenced node, because it takes
    /// the referents too — which is why it is reachable from
    /// [`ModelBuilderOps`](crate::client_ops::ModelBuilderOps) and not from the client tier.
    ///
    /// The removal is also a collection point: nodes the closure orphaned but did not
    /// take — everything *upstream* of the roots, like a retired factor's body — unwind
    /// in the cascade before returning.
    pub fn remove_closure(&mut self, roots: &[Id]) -> SkoroResult<Vec<Node<B>>> {
        let removed = self.graph.remove_closure(roots)?;
        self.prune_certificate();
        self.offer_parents(&removed);
        self.cleanup();
        Ok(removed)
    }

    /// Queue the parents of `removed` for garbage collection: a node that lost its last
    /// child in the removal may now be collectable, and the handle that once named it is
    /// long gone.
    fn offer_parents(&mut self, removed: &[Node<B>]) {
        self.cleanup_registry.offer(
            removed
                .iter()
                .flat_map(|node| node.parent_ids().iter().copied()),
        );
    }

    /// Drop the certificate entries whose factor no longer exists — and unregister the
    /// density of a record whose *residual* died while the density still scores. Half a
    /// factor is not a factor: a term the objective sums but `J` has no rows for is the
    /// one failure mode here nothing reports, so the two leave in the same breath.
    ///
    /// Liveness is two questions, both asked of the graph, right now: the residual must
    /// be in the arena, and the density must still be a live likelihood term. The second
    /// is the one arena membership cannot answer — `unregister_density` orphans a demoted
    /// or replaced quantity's density without removing it, and an orphaned density scores
    /// nothing.
    ///
    /// **A dropped record's nodes are re-offered to the collector.** Being named by a record is
    /// what protected them, and losing a protector is not an event the graph can see — no edge
    /// changed, so nothing else will ever reconsider them. Every mutation that reduces a node's
    /// reachability has to say so here, or the node is unreachable *and* uncollected, which is
    /// a leak that repeats once per retired factor.
    fn prune_certificate(&mut self) {
        let registry = self.cleanup_registry.clone();
        let graph = &self.graph;
        let mut densities = Vec::new();
        self.certificate.retain(
            |record| graph.contains(record.residual) && graph.is_scoring(record.density),
            |record| {
                if !graph.contains(record.residual) && graph.is_scoring(record.density) {
                    densities.push(record.density);
                }
                registry.offer([record.density, record.residual]);
            },
        );
        for density in densities {
            self.graph.unregister_density_node(density);
        }
        self.debug_validate_state();
    }

    /// Record a factor in the server's certificate. The `D` bound is the whole guarantee;
    /// see [`ResidualCertificate::record`].
    ///
    /// Recording pins the factor's nodes: neither `cleanup` nor [`remove`](Self::remove) may
    /// take one until the record leaves.
    pub fn record_factor<D>(
        &mut self,
        distr: &D,
        density: Id,
        residual: &Q<B, EuclideanD>,
    ) -> SkoroResult<()>
    where
        D: WhitenedResidual<B>,
    {
        self.certificate.record(distr, density, residual)?;
        self.debug_validate_state();
        Ok(())
    }

    /// Record a joint factor over quantities the model already holds. The `F` bound is
    /// the guarantee, exactly as in [`record_factor`](Self::record_factor); see
    /// [`ResidualCertificate::record_joint`].
    pub fn record_joint_factor<F>(
        &mut self,
        factor: &F,
        density: Id,
        residual: &Q<B, EuclideanD>,
    ) -> SkoroResult<()>
    where
        F: JointWhitenedResidual<B>,
    {
        self.certificate.record_joint(factor, density, residual)?;
        self.debug_validate_state();
        Ok(())
    }

    /// The evidence this server maintains about its factors, pruned against the graph at
    /// every mutation above.
    pub fn certificate(&self) -> &ResidualCertificate {
        &self.certificate
    }

    /// The certificate invariants, checked after every certificate mutation in debug
    /// builds — the same regime [`Graph::validate_indices`] runs under, one layer up.
    ///
    /// Only the certificate-against-graph half is asserted here. The certificate's own
    /// index is no longer a second copy to cross-check: it is private to
    /// [`ResidualCertificate`] and written beside every record that enters or leaves.
    fn debug_validate_state(&self) {
        #[cfg(debug_assertions)]
        for record in self.certificate.factors() {
            assert!(
                self.graph.contains(record.residual),
                "certificate names residual {:?}, which is not in the arena",
                record.residual
            );
            assert!(
                self.graph.is_scoring(record.density),
                "certificate names density {:?}, which is not a live likelihood term",
                record.density
            );
        }
    }

    /// Remove all collectable quantities, cascading to any parents the collection
    /// itself orphans.
    ///
    /// Eligibility is decided *at collection time* from the graph's current state, never
    /// from anything recorded when a handle died: a candidate id may name nothing (a
    /// closure removal took it), may name a node that has since gained a child, or may
    /// name a pinned or model-owned node — each is simply skipped. A garbage collector
    /// must treat "already collected" as ordinary, so there is no way for a stale queue
    /// entry to fail here, by construction.
    pub fn cleanup(&mut self) {
        loop {
            let mut batch = Vec::new();
            let mut seen = HashSet::new();
            while let Some(id) = self.cleanup_registry.pop() {
                if seen.insert(id) && self.collectable(id) {
                    batch.push(id);
                }
            }
            if batch.is_empty() {
                return;
            }
            // Eligibility was checked above against the live graph: every id in the
            // batch is present, childless, unowned and unreferenced, so the batch is
            // closed and `remove` cannot refuse it. A failure here is a bug in
            // `collectable`, not bad input.
            let removed = self
                .graph
                .remove(&batch)
                .expect("internal: cleanup collected an ineligible batch");
            for node in &removed {
                self.cleanup_registry
                    .offer(node.parent_ids().iter().copied());
            }
        }
    }

    /// Whether `id` may be collected right now. Re-derived from the graph and the handle
    /// registry on every call — see [`cleanup`](Self::cleanup).
    fn collectable(&self, id: Id) -> bool {
        if self.cleanup_registry.is_tracked(id) || self.certificate.names(id) {
            return false;
        }
        if self
            .graph
            .children_of(id)
            .is_some_and(|children| !children.is_empty())
        {
            return false;
        }
        match self.graph.node(id) {
            Some(Node::Mapped(_)) => true,
            // A density that still scores is a root: the objective sums it, so no absence of
            // handles makes it garbage. One that has been *orphaned* — by a demotion, or by a
            // distribution being replaced — scores nothing and is reachable from nothing, which
            // is the same condition every other node is collected under.
            //
            // This is the arm that makes a demotion actually free anything. The orphan holds
            // the whole subgraph its density was built over, so leaving it to
            // `orphaned_densities` plus an explicit `remove_closure` strands a factor's worth
            // of nodes per demotion — once per tick of a sliding window.
            Some(Node::LogDensity(_)) => !self.graph.is_scoring(id),
            // A graph-created constant (a stacked-tangent anchor) carries no handle and
            // cannot be named once orphaned, so it is collectable where an ordinary
            // constant is not — a caller may still hold an `Id` for one of those.
            Some(Node::Constant(_)) => self.graph.is_private(id),
            _ => false,
        }
    }

    /// Get the quantity type associated with a quantity by its Id.
    ///
    /// Returns an error if the node does not exist or the requested type does
    /// not match the type stored in the model.
    #[inline(always)]
    pub fn get_quantity_type<K, T>(&self, tag: &Q<B, T>) -> SkoroResult<T>
    where
        K: Kind<B>,
        T: QuantityType<B, K>,
    {
        self.graph.get_quantity_type::<K, T>(tag.id)
    }

    /// Evaluate the log-density of the model at the specified points.
    #[inline(always)]
    pub fn log_density(&self, points: &ValueMap<B>) -> SkoroResult<Point<B>> {
        self.graph.log_density(points)
    }

    /// The evaluation of [`log_density`](Self::log_density) with the first-order gradient
    /// calculation.
    #[inline(always)]
    pub fn log_density_ord1(&self, points: &ValueMap<B>) -> SkoroResult<PointEvalOrder1<B>> {
        self.graph.log_density_ord1(points)
    }

    /// The evaluation of [`log_density`](Self::log_density) with the first-order gradient and
    /// second-order Hessian-product calculation.
    #[inline(always)]
    pub fn log_density_ord2(
        &self,
        points: &ValueMap<B>,
        tangents: &HashMap<Id, TangentVector<B>>,
    ) -> SkoroResult<PointEvalOrder2<B>> {
        self.graph.log_density_ord2(points, tangents)
    }

    /// Stochastically generate all values in the model.
    #[inline(always)]
    pub fn generate(&self, batch_size: usize) -> SkoroResult<ForwardOutput<B>> {
        self.graph.generate(batch_size)
    }

    /// Stochastically generate all values in the model.
    #[inline(always)]
    pub fn generate_from(
        &self,
        points: &ValueMap<B>,
        batch_size: usize,
    ) -> SkoroResult<ForwardOutput<B>> {
        self.graph.generate_from(points, batch_size)
    }

    /// Simple interface for evaluating a multiple mapped quantities.
    #[inline(always)]
    pub fn evaluate_ids(&self, points: &ValueMap<B>, ids: &[Id]) -> SkoroResult<ForwardOutput<B>> {
        self.graph.evaluate_ids(points, ids)
    }

    /// The JVP evaluation [`evaluate_ids`](Self::evaluate_ids); if you only want gradients, instead
    /// use [`evaluate_ids_ord1`](Self::evaluate_ids_ord1).
    #[inline(always)]
    pub fn evaluate_ids_jvp(
        &self,
        points: &ValueMap<B>,
        tangents: &HashMap<Id, TangentFrame<B>>,
        ids: &[Id],
    ) -> SkoroResult<JvpForwardOutput<B>> {
        self.graph.evaluate_ids_jvp(points, tangents, ids)
    }

    /// The VJP evaluation [`evaluate_ids`](Self::evaluate_ids); if you only want gradients, instead
    /// use [`evaluate_ids_ord1`](Self::evaluate_ids_ord1).
    #[inline(always)]
    pub fn evaluate_ids_vjp(
        &self,
        points: &ValueMap<B>,
        cotangents: &HashMap<Id, CotangentFrame<B>>,
        ids: &[Id],
    ) -> SkoroResult<VjpForwardOutput<B>> {
        self.graph.evaluate_ids_vjp(points, cotangents, ids)
    }

    /// The HVP evaluation [`evaluate_ids`](Self::evaluate_ids); if you only want gradients and
    /// Hessian-products, instead use [`evaluate_ids_ord2`](Self::evaluate_ids_ord2).
    #[inline(always)]
    pub fn evaluate_ids_hvp(
        &self,
        points: &ValueMap<B>,
        tangents: &HashMap<Id, TangentVector<B>>,
        ord1_cotangents: &HashMap<Id, CotangentFrame<B>>,
        ord2_cotangents: &HashMap<Id, CotangentFrame<B>>,
        ids: &[Id],
    ) -> SkoroResult<HvpForwardOutput<B>> {
        self.graph
            .evaluate_ids_hvp(points, tangents, ord1_cotangents, ord2_cotangents, ids)
    }

    /// The evaluation [`evaluate_ids`](Self::evaluate_ids) with the first-order gradient calculation.
    #[inline(always)]
    pub fn evaluate_ids_ord1(
        &self,
        points: &ValueMap<B>,
        ids: &[Id],
    ) -> SkoroResult<EvalOrder1<B>> {
        self.graph.evaluate_ids_ord1(points, ids)
    }

    /// The evaluation [`evaluate_ids`](Self::evaluate_ids) with the first-order gradient and second-order
    /// Hessian-product calculation.
    #[inline(always)]
    pub fn evaluate_ids_ord2(
        &self,
        points: &ValueMap<B>,
        tangents: &HashMap<Id, TangentVector<B>>,
        ids: &[Id],
    ) -> SkoroResult<EvalOrder2<B>> {
        self.graph.evaluate_ids_ord2(points, tangents, ids)
    }

    /// Simple interface for evaluating a single mapped quantity. This should not be used multiple
    /// times in sequence, since it may lead to inefficient recalculation. In that event, instead
    /// use [`evaluate_ids`](Self::evaluate_ids).
    #[inline(always)]
    pub fn evaluate<M: ContinuousQuantity<B>>(
        &self,
        points: &ValueMap<B>,
        tag: &Q<B, M>,
    ) -> SkoroResult<Point<B>> {
        self.graph.evaluate(points, tag.id)
    }

    /// The JVP evaluation [`evaluate`](Self::evaluate); if you only want gradients, use
    /// [`evaluate_ord1`](Self::evaluate_ord1).
    #[inline(always)]
    pub fn evaluate_jvp<M: ContinuousQuantity<B>>(
        &self,
        points: &ValueMap<B>,
        tangents: &HashMap<Id, TangentFrame<B>>,
        tag: &Q<B, M>,
    ) -> SkoroResult<PointJvpEval<B>> {
        self.graph.evaluate_jvp(points, tangents, tag.id)
    }

    /// The VJP evaluation [`evaluate`](Self::evaluate); if you only want gradients, use
    /// [`evaluate_ord1`](Self::evaluate_ord1).
    #[inline(always)]
    pub fn evaluate_vjp<M: ContinuousQuantity<B>>(
        &self,
        points: &ValueMap<B>,
        cotangents: &HashMap<Id, CotangentFrame<B>>,
        tag: &Q<B, M>,
    ) -> SkoroResult<PointVjpEval<B>> {
        self.graph.evaluate_vjp(points, cotangents, tag.id)
    }

    /// The HVP evaluation [`evaluate`](Self::evaluate); if you only want gradients and
    /// Hessian-products, instead use [`evaluate_ord2`](Self::evaluate_ord2).
    #[inline(always)]
    pub fn evaluate_hvp<M: ContinuousQuantity<B>>(
        &self,
        points: &ValueMap<B>,
        tangents: &HashMap<Id, TangentVector<B>>,
        ord1_cotangents: &HashMap<Id, CotangentFrame<B>>,
        ord2_cotangents: &HashMap<Id, CotangentFrame<B>>,
        tag: &Q<B, M>,
    ) -> SkoroResult<PointHvpEval<B>> {
        self.graph
            .evaluate_hvp(points, tangents, ord1_cotangents, ord2_cotangents, tag.id)
    }

    /// The evaluation [`evaluate`](Self::evaluate) with the first-order gradient calculation.
    #[inline(always)]
    pub fn evaluate_ord1<M: ContinuousQuantity<B>>(
        &self,
        points: &ValueMap<B>,
        tag: &Q<B, M>,
    ) -> SkoroResult<PointEvalOrder1<B>> {
        self.graph.evaluate_ord1(points, tag.id)
    }

    /// The evaluation [`evaluate`](Self::evaluate) with the first-order gradient and second-order
    /// Hessian-product calculation.
    #[inline(always)]
    pub fn evaluate_ord2<M: ContinuousQuantity<B>>(
        &self,
        points: &ValueMap<B>,
        tangents: &HashMap<Id, TangentVector<B>>,
        tag: &Q<B, M>,
    ) -> SkoroResult<PointEvalOrder2<B>> {
        self.graph.evaluate_ord2(points, tangents, tag.id)
    }
}

/// Drop guard for [`Q`] instances, reporting handle liveness to the server's
/// [`CleanupRegistry`].
///
/// One guard exists per handle-family: `Q` clones share it, so a guard is created
/// exactly once per node and dropped when the last clone naming it is gone. Creation
/// registers the id; the drop deregisters it and offers it to `cleanup` as a candidate.
/// The guard carries no opinion about collectability — that is re-derived from the
/// graph at collection time, so this type can never be the reason a live node is
/// collected or a dead one panics the collector.
pub(crate) struct NodeDropGuard {
    id: Id,
    registry: Arc<CleanupRegistry>,
}

impl NodeDropGuard {
    pub(crate) fn new(id: Id, registry: &Arc<CleanupRegistry>) -> Arc<Self> {
        registry.track(id);
        Arc::new(Self {
            id,
            registry: registry.clone(),
        })
    }
}

impl Drop for NodeDropGuard {
    fn drop(&mut self) {
        self.registry.untrack(self.id);
    }
}

/// A lightweight, clonable client handle to a specific `ModelServer`.
#[derive(Debug)]
pub struct ModelClient<B: Backend> {
    pub server_id: Id,
    server: Arc<RwLock<ModelServer<B>>>,
    _phantom: PhantomData<B>,
}

impl<B: Backend> Default for ModelClient<B> {
    fn default() -> Self {
        Self::new(Default::default())
    }
}

impl<B: Backend> ModelClient<B> {
    /// Creates a new server and returns a client handle to it. The server lives as long as at
    /// least one `ModelClient` holds a strong reference.
    pub fn new(device: B::Device) -> Self {
        let server_id = SERVER_GENERATOR.issue_id();
        let server = Arc::new(RwLock::new(ModelServer::<B>::new(device)));
        let any_arc: Arc<dyn Any + Send + Sync> = server.clone();
        MODEL_SERVERS.insert(server_id, Arc::downgrade(&any_arc));
        Self {
            server_id,
            server,
            _phantom: PhantomData,
        }
    }

    /// Re-attach a client to an existing server by its `Id`. Returns `None` if the server has been
    /// dropped (all prior clients released it).
    pub fn from_id(server_id: Id) -> Option<Self> {
        let weak = MODEL_SERVERS.get(&server_id)?.clone();
        let any_arc = weak.upgrade()?;
        let server = any_arc.downcast::<RwLock<ModelServer<B>>>().ok()?;
        Some(Self {
            server_id,
            server,
            _phantom: PhantomData,
        })
    }

    /// Call a function which requires shared reference to server.
    pub fn server_call<O, F: Fn(&ModelServer<B>) -> O>(&self, function: F) -> O {
        let server = self.server.read().unwrap();
        function(&server)
    }

    /// Call a function which requires exclusive reference to server.
    pub fn server_call_mut<O, F: FnOnce(&mut ModelServer<B>) -> O>(&self, function: F) -> O {
        let mut server = self.server.write().unwrap();
        function(&mut server)
    }

    // `get_server` deliberately does not exist. Handing out the `Arc<RwLock<..>>` gave
    // every caller a `&mut ModelServer`, and so a route around any facade that had been
    // careful about what it exposed. `server_call` / `server_call_mut` cover the same
    // ground while keeping the lock scope visible at the call site.

    /// Add a fixed, immutable quantity to the graph of the model server,
    /// continuous or discrete.
    pub fn add_constant<K, T, D>(&self, quantity_type: T, value: &D) -> Q<B, T>
    where
        K: Kind<B>,
        T: QuantityType<B, K> + Packable<K::Elem, D>,
        D: 'static,
    {
        let mut server = self.server.write().unwrap();
        server.cleanup();
        let id = server.add_constant(quantity_type, value);
        Q::new(
            id,
            self.clone(),
            NodeDropGuard::new(id, &server.cleanup_registry),
        )
    }

    /// Add a fixed, immutable quantity that differs across the batch: one value per batch row,
    /// in order, rather than one value broadcast to every row.
    ///
    /// See [`Graph::add_batched_constant`](skoro_graph::graph::Graph::add_batched_constant) for
    /// the batch-pinning this implies.
    pub fn add_batched_constant<K, T, D>(&self, quantity_type: T, values: &[D]) -> Q<B, T>
    where
        K: Kind<B>,
        T: QuantityType<B, K> + Packable<K::Elem, D>,
        D: 'static,
    {
        let mut server = self.server.write().unwrap();
        server.cleanup();
        let id = server.add_batched_constant(quantity_type, values);
        Q::new(
            id,
            self.clone(),
            NodeDropGuard::new(id, &server.cleanup_registry),
        )
    }

    /// Add a free quantity to the graph of the model server, continuous or
    /// discrete.
    pub fn add_parameter<K, T>(&self, quantity_type: T) -> Q<B, T>
    where
        K: Kind<B>,
        T: QuantityType<B, K>,
    {
        let mut server = self.server.write().unwrap();
        server.cleanup();
        let id = server.add_parameter(quantity_type);
        Q::new(
            id,
            self.clone(),
            NodeDropGuard::new(id, &server.cleanup_registry),
        )
    }

    /// Adds a stochastic quantity to the graph of the model server, continuous
    /// or discrete.
    ///
    /// Returns an error if the distribution references a parent that does not
    /// exist in the model.
    pub(crate) fn add_quantity<K, D>(&self, distr: D) -> SkoroResult<Q<B, D::Qty>>
    where
        K: Kind<B>,
        D: Distribution<B, K>,
    {
        // add stochastic quantity
        let mut server = self.server.write().unwrap();
        let id = server.add_stochastic(&distr)?;
        let x = Q::new(
            id,
            self.clone(),
            NodeDropGuard::new(id, &server.cleanup_registry),
        );

        // maybe slow, but currently the price for ergonomic interface: we need to drop the server
        // to add the log-density term (or else deadlock will occur).
        drop(server);
        let density = distr.log_density(&x)?;

        // again, slow... but now we mark the mapped quantity as a density.
        let mut server = self.server.write().unwrap();
        server.cleanup();
        server.mark_density(&density, &x)?;
        Ok(x)
    }

    /// The stacked tangent `δ = [x₁ ⊖ p₁ ; … ; x_k ⊖ p_k]` over a heterogeneous set of
    /// quantities, as one `Q<B, EuclideanD>` of width `Σ dᵢ`.
    ///
    /// This is how a factor scores several quantities at once without a product manifold; see
    /// [`Graph::stacked_tangent`](skoro_graph::graph::Graph::stacked_tangent) for what it builds
    /// and why the anchors are frozen.
    ///
    /// Inherent on [`ModelClient`] and deliberately **not** on [`ModelClientOps`] or
    /// [`ModelBuilderOps`]: the type gate is a statement about what the *traits* offer, and an
    /// estimator bounded on `NlsStructure` holds one of those rather than a `ModelClient`. A
    /// facade that restricts the model therefore still withholds this, while the builder that
    /// assembles a joint factor -- which already holds an unrestricted client -- can reach it.
    pub fn stacked_tangent(&self, at: &PointMap<B>, ids: &[Id]) -> SkoroResult<Q<B, EuclideanD>> {
        let mut server = self.server.write().unwrap();
        server.cleanup();
        let id = server.stacked_tangent(at, ids)?;
        Ok(Q::new(
            id,
            self.clone(),
            NodeDropGuard::new(id, &server.cleanup_registry),
        ))
    }

    pub(crate) fn add_mapped<K, T>(
        &self,
        quantity_type: T,
        op: impl GraphOperation<B>,
        parent_ids: Vec<Id>,
    ) -> SkoroResult<Q<B, T>>
    where
        K: Kind<B>,
        T: QuantityType<B, K>,
    {
        let mut server = self.server.write().unwrap();
        server.cleanup();
        let id = server.add_mapped(quantity_type, op, parent_ids)?;
        Ok(Q::new(
            id,
            self.clone(),
            NodeDropGuard::new(id, &server.cleanup_registry),
        ))
    }

    /// Mark the mapped quantity as a likelihood.
    pub(crate) fn mark_likelihood(&self, likelihood: &Q<B, Real>) -> SkoroResult<()> {
        self.server.write().unwrap().mark_likelihood(likelihood)
    }

    /// Impose a posterior by observing a stochastic quantity at a specified value.
    pub fn observe<K, T, D>(&self, q: &Q<B, T>, value: &D) -> SkoroResult<()>
    where
        K: Kind<B>,
        T: QuantityType<B, K> + Packable<K::Elem, D>,
        D: 'static,
    {
        self.server.write().unwrap().observe::<K, T, D>(q, value)
    }

    /// Undo the effects of [`observe`](Self::observe).
    pub fn unobserve<T>(&self, q: &Q<B, T>) -> SkoroResult<()> {
        self.server.write().unwrap().unobserve(q)
    }

    /// Update a stochastic quantity's distribution.
    ///
    /// Returns an error if the node is not stochastic, a parent is dangling, or
    /// the update would introduce a cycle.
    pub(crate) fn update_quantity<K, D>(&self, tag: &Q<B, D::Qty>, distr: D) -> SkoroResult<()>
    where
        K: Kind<B>,
        D: Distribution<B, K>,
    {
        // update stochastic quantity
        let mut server = self.server.write().unwrap();
        server.update_stochastic(tag, distr.sampler(), distr.parent_ids().to_vec())?;

        // maybe slow, but currently the price for ergonomic interface: we need to drop the server
        // to add the log-density term (or else deadlock will occur).
        drop(server);
        let density = distr.log_density(tag)?;

        // again, slow... but now we mark the mapped quantity as a density.
        let mut server = self.server.write().unwrap();
        server.cleanup();
        server.mark_density(&density, tag)?;
        Ok(())
    }

    /// Demote a stochastic quantity to a free parameter, keeping its identity.
    ///
    /// See [`ModelServer::demote_to_parameter`].
    pub fn demote_to_parameter<T>(&self, tag: &Q<B, T>) -> SkoroResult<()> {
        self.server.write().unwrap().demote_to_parameter(tag)
    }

    /// Removes quantities, verifying they have no children which would be kept in the graph.
    pub fn remove(&self, ids: &[Id]) -> SkoroResult<Vec<Node<B>>> {
        self.server.write().unwrap().remove(ids)
    }

    /// Remove `roots` together with everything downstream of them.
    ///
    /// See [`ModelServer::remove_closure`].
    pub fn remove_closure(&self, roots: &[Id]) -> SkoroResult<Vec<Node<B>>> {
        self.server.write().unwrap().remove_closure(roots)
    }

    /// Remove all non-likelihood mapped quantities which have no more references.
    pub fn cleanup(&self) {
        self.server.write().unwrap().cleanup();
    }

    /// Get the quantity type associated with a quantity.
    ///
    /// Returns an error if the node does not exist or the requested type does
    /// not match the type stored in the model.
    pub fn get_quantity_type<K, T>(&self, tag: &Q<B, T>) -> SkoroResult<T>
    where
        K: Kind<B>,
        T: QuantityType<B, K>,
    {
        self.server.read().unwrap().get_quantity_type(tag)
    }

    /// Evaluate the log-density of the model at the specified points.
    pub fn log_density(&self, points: &ValueMap<B>) -> SkoroResult<Point<B>> {
        self.server.read().unwrap().log_density(points)
    }

    /// The evaluation of [`log_density`](Self::log_density) with the first-order gradient
    /// calculation.
    pub fn log_density_ord1(&self, points: &ValueMap<B>) -> SkoroResult<PointEvalOrder1<B>> {
        self.server.read().unwrap().log_density_ord1(points)
    }

    /// The evaluation of [`log_density`](Self::log_density) with the first-order gradient
    /// calculation and second-order Hessian-product calculation.
    pub fn log_density_ord2(
        &self,
        points: &ValueMap<B>,
        tangents: &HashMap<Id, TangentVector<B>>,
    ) -> SkoroResult<PointEvalOrder2<B>> {
        self.server
            .read()
            .unwrap()
            .log_density_ord2(points, tangents)
    }

    /// Stochastically generate all values in the model.
    pub fn generate(&self, batch_size: usize) -> SkoroResult<ForwardOutput<B>> {
        self.server.read().unwrap().generate(batch_size)
    }

    /// Stochastically generate all values in the model.
    pub fn generate_from(
        &self,
        points: &ValueMap<B>,
        batch_size: usize,
    ) -> SkoroResult<ForwardOutput<B>> {
        self.server
            .read()
            .unwrap()
            .generate_from(points, batch_size)
    }

    /// Simple interface for evaluating a multiple mapped quantities.
    pub fn evaluate_ids(&self, points: &ValueMap<B>, ids: &[Id]) -> SkoroResult<ForwardOutput<B>> {
        self.server.read().unwrap().evaluate_ids(points, ids)
    }

    /// The JVP evaluation [`evaluate_ids`](Self::evaluate_ids); if you only want gradients, instead
    /// use [`evaluate_ids_ord1`](Self::evaluate_ids_ord1).
    #[inline(always)]
    pub fn evaluate_ids_jvp(
        &self,
        points: &ValueMap<B>,
        tangents: &HashMap<Id, TangentFrame<B>>,
        ids: &[Id],
    ) -> SkoroResult<JvpForwardOutput<B>> {
        self.server
            .read()
            .unwrap()
            .evaluate_ids_jvp(points, tangents, ids)
    }

    /// The VJP evaluation [`evaluate_ids`](Self::evaluate_ids); if you only want gradients, instead
    /// use [`evaluate_ids_ord1`](Self::evaluate_ids_ord1).
    #[inline(always)]
    pub fn evaluate_ids_vjp(
        &self,
        points: &ValueMap<B>,
        cotangents: &HashMap<Id, CotangentFrame<B>>,
        ids: &[Id],
    ) -> SkoroResult<VjpForwardOutput<B>> {
        self.server
            .read()
            .unwrap()
            .evaluate_ids_vjp(points, cotangents, ids)
    }

    /// The HVP evaluation [`evaluate_ids`](Self::evaluate_ids); if you only want gradients and
    /// Hessian-products, instead use [`evaluate_ids_ord2`](Self::evaluate_ids_ord2).
    #[inline(always)]
    pub fn evaluate_ids_hvp(
        &self,
        points: &ValueMap<B>,
        tangents: &HashMap<Id, TangentVector<B>>,
        ord1_cotangents: &HashMap<Id, CotangentFrame<B>>,
        ord2_cotangents: &HashMap<Id, CotangentFrame<B>>,
        ids: &[Id],
    ) -> SkoroResult<HvpForwardOutput<B>> {
        self.server.read().unwrap().evaluate_ids_hvp(
            points,
            tangents,
            ord1_cotangents,
            ord2_cotangents,
            ids,
        )
    }

    /// The evaluation [`evaluate_ids`](Self::evaluate_ids) with the first-order gradient calculation.
    pub fn evaluate_ids_ord1(
        &self,
        points: &ValueMap<B>,
        ids: &[Id],
    ) -> SkoroResult<EvalOrder1<B>> {
        self.server.read().unwrap().evaluate_ids_ord1(points, ids)
    }

    /// The evaluation [`evaluate_ids`](Self::evaluate_ids) with the first-order gradient and second-order
    /// Hessian-product calculation.
    pub fn evaluate_ids_ord2(
        &self,
        points: &ValueMap<B>,
        tangents: &HashMap<Id, TangentVector<B>>,
        ids: &[Id],
    ) -> SkoroResult<EvalOrder2<B>> {
        self.server
            .read()
            .unwrap()
            .evaluate_ids_ord2(points, tangents, ids)
    }

    /// Simple interface for evaluating a single mapped quantity. This should not be used multiple
    /// times in sequence, since it may lead to inefficient recalculation. In that event, instead
    /// use [`evaluate_ids`](Self::evaluate_ids).
    pub fn evaluate<T: ContinuousQuantity<B>>(
        &self,
        points: &ValueMap<B>,
        tag: &Q<B, T>,
    ) -> SkoroResult<Point<B>> {
        self.server.read().unwrap().evaluate(points, tag)
    }

    /// The JVP evaluation [`evaluate`](Self::evaluate); if you only want gradients, use
    /// [`evaluate_ord1`](Self::evaluate_ord1).
    #[inline(always)]
    pub fn evaluate_jvp<T: ContinuousQuantity<B>>(
        &self,
        points: &ValueMap<B>,
        tangents: &HashMap<Id, TangentFrame<B>>,
        tag: &Q<B, T>,
    ) -> SkoroResult<PointJvpEval<B>> {
        self.server
            .read()
            .unwrap()
            .evaluate_jvp(points, tangents, tag)
    }

    /// The VJP evaluation [`evaluate`](Self::evaluate); if you only want gradients, use
    /// [`evaluate_ord1`](Self::evaluate_ord1).
    #[inline(always)]
    pub fn evaluate_vjp<T: ContinuousQuantity<B>>(
        &self,
        points: &ValueMap<B>,
        cotangents: &HashMap<Id, CotangentFrame<B>>,
        tag: &Q<B, T>,
    ) -> SkoroResult<PointVjpEval<B>> {
        self.server
            .read()
            .unwrap()
            .evaluate_vjp(points, cotangents, tag)
    }

    /// The HVP evaluation [`evaluate`](Self::evaluate); if you only want gradients and
    /// Hessian-products, instead use [`evaluate_ord2`](Self::evaluate_ord2).
    #[inline(always)]
    pub fn evaluate_hvp<T: ContinuousQuantity<B>>(
        &self,
        points: &ValueMap<B>,
        tangents: &HashMap<Id, TangentVector<B>>,
        ord1_cotangents: &HashMap<Id, CotangentFrame<B>>,
        ord2_cotangents: &HashMap<Id, CotangentFrame<B>>,
        tag: &Q<B, T>,
    ) -> SkoroResult<PointHvpEval<B>> {
        self.server.read().unwrap().evaluate_hvp(
            points,
            tangents,
            ord1_cotangents,
            ord2_cotangents,
            tag,
        )
    }

    /// The evaluation [`evaluate`](Self::evaluate) with the first-order gradient calculation.
    pub fn evaluate_ord1<T: ContinuousQuantity<B>>(
        &self,
        points: &ValueMap<B>,
        tag: &Q<B, T>,
    ) -> SkoroResult<PointEvalOrder1<B>> {
        self.server.read().unwrap().evaluate_ord1(points, tag)
    }

    /// The evaluation [`evaluate`](Self::evaluate) with the first-order gradient and second-order
    /// Hessian-product calculation.
    pub fn evaluate_ord2<T: ContinuousQuantity<B>>(
        &self,
        points: &ValueMap<B>,
        tangents: &HashMap<Id, TangentVector<B>>,
        tag: &Q<B, T>,
    ) -> SkoroResult<PointEvalOrder2<B>> {
        self.server
            .read()
            .unwrap()
            .evaluate_ord2(points, tangents, tag)
    }
}

impl<B: Backend> Clone for ModelClient<B> {
    fn clone(&self) -> Self {
        Self {
            server_id: self.server_id,
            server: self.server.clone(),
            _phantom: PhantomData,
        }
    }
}

impl<B: Backend> Drop for ModelClient<B> {
    fn drop(&mut self) {
        // If we are the last strong reference, purge the (now-dangling) `Weak` from the registry to
        // keep the map from growing unbounded. Any race with a concurrent clone is benign: the
        // worst case is a dead `Weak` left in the map, which `from_id` handles by checking
        // `upgrade()`.
        if Arc::strong_count(&self.server) == 1 {
            MODEL_SERVERS.remove(&self.server_id);
        }
    }
}

impl<B: Backend> PartialEq for ModelClient<B> {
    fn eq(&self, other: &Self) -> bool {
        self.server_id == other.server_id
    }
}
