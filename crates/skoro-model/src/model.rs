use dashmap::DashMap;
use lazy_static::lazy_static;
use std::any::Any;
use std::collections::{HashMap, HashSet, VecDeque};
use std::marker::PhantomData;
use std::sync::{Arc, RwLock, Weak};

use skoro_backend::{Backend, CotangentFrame, Point, TangentFrame, TangentVector};
use skoro_core::kind::Kind;
use skoro_core::{
    error::{SkoroError, SkoroResult},
    id::{Id, IdGenerator},
    pack::Packable,
};
use skoro_graph::{
    compiler::{ForwardOutput, HvpForwardOutput, JvpForwardOutput, VjpForwardOutput},
    graph::{
        EvalOrder1, EvalOrder2, Graph, PointEvalOrder1, PointEvalOrder2, PointHvpEval,
        PointJvpEval, PointVjpEval,
    },
    node::Node,
    operation::DynGraphOperation,
    point_map::PointMap,
    quantity::{ContinuousQuantity, QuantityType},
    tangent_layout::TangentLayout,
};
use skoro_operation::ValueMap;
use skoro_riemann::euclidean::EuclideanD;
use skoro_riemann::real::Real;

use crate::residual::{FactorRecord, ResidualRegistry};
use crate::{cleanup::HandleRegistry, distribution::Distribution, q::Q};

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
    handles: Arc<HandleRegistry>,
    /// What this server has been told about its least-squares factors, pruned against the
    /// graph inside every mutating method below — so a record and the factor it names cannot
    /// drift apart between calls, only within one, where `prune_registry` is the last thing
    /// that runs.
    ///
    /// It is also the pin table: the nodes it names are the ones neither the collector may
    /// reclaim nor [`remove`](Self::remove) may delete. That is one set, owned by
    /// [`ResidualRegistry`] and asked through [`names`](ResidualRegistry::names), rather
    /// than a copy kept here — a second owner would be a second thing to keep in step.
    registry: ResidualRegistry,
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
            handles: Arc::new(HandleRegistry::default()),
            registry: ResidualRegistry::new(),
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
    pub fn add_batched_constant<K, T, D>(
        &mut self,
        quantity_type: T,
        values: &[D],
    ) -> SkoroResult<Id>
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
        op: impl DynGraphOperation<B>,
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
        self.graph.mark_log_density(density.id, stochastic.id)
    }

    /// Mark a mapped quantity as a likelihood.
    pub fn mark_likelihood(&mut self, likelihood: &Q<B, Real>) -> SkoroResult<()> {
        self.graph.mark_log_likelihood(likelihood.id)
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

    /// Demote a stochastic quantity to a free parameter, keeping its identity.
    ///
    /// The quantity loses its sampler, its parents, and its own log-density; its
    /// [`Id`], quantity type, and every dependent quantity are untouched. Use this
    /// when a quantity's distribution is superseded by a joint likelihood factor
    /// over several quantities — a marginalization whose result does not factorize.
    ///
    /// The demoted quantity's log-density leaves the objective here, so a
    /// record naming it leaves too: `prune_registry` runs in the same call, and the
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
        self.graph.demote_to_parameter(tag.id)?;
        self.prune_registry();
        self.handles.mark_dirty();
        self.cleanup();
        Ok(())
    }

    /// Removes quantities, verifying they have no children which would be kept in the graph.
    ///
    /// Refuses a node a record names, for the same reason it refuses one another node
    /// still points into: this is the primitive that will not break a reference, and a record
    /// is a reference. That refusal is what keeps `remove` safe on
    /// [`ModelClientOps`](crate::client_ops::ModelClientOps), which an estimator holds — a
    /// recorded residual is childless, so without the check a `remove` could silently take a
    /// factor out of the objective. Retiring a factor is
    /// [`remove_closure`](Self::remove_closure), one tier up.
    ///
    /// The removal is also a collection point: anything it orphaned is swept before
    /// returning, so the model is clean after every retire.
    pub fn remove(&mut self, ids: &[Id]) -> SkoroResult<Vec<Node<B>>> {
        if let Some(id) = ids.iter().find(|id| self.registry.names(**id)) {
            return Err(SkoroError::NodeIsRecorded { id: *id });
        }
        let removed = self.graph.remove(ids)?;
        self.prune_registry();
        self.handles.mark_dirty();
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
        self.prune_registry();
        self.handles.mark_dirty();
        self.cleanup();
        Ok(removed)
    }

    /// Drop the records whose factor no longer exists — and unmark the density of a record
    /// whose *residual* died while the density still scores. Half a factor is not a factor:
    /// a term the objective sums but `J` has no rows for is the one failure mode here
    /// nothing reports, so the two leave in the same breath.
    ///
    /// Liveness is two questions, both asked of the graph, right now: the residual must be
    /// in the arena, and the density must still be a live likelihood term. The second is the
    /// one arena membership cannot answer — a demotion orphans a density without removing
    /// it, and an orphaned density scores nothing.
    fn prune_registry(&mut self) {
        let graph = &self.graph;
        let mut densities = Vec::new();
        self.registry.retain(
            |record| graph.contains(record.residual) && graph.is_scoring(record.density),
            |record| {
                if !graph.contains(record.residual) && graph.is_scoring(record.density) {
                    densities.push(record.density);
                }
            },
        );
        for density in densities {
            // Infallible in practice: `density` was just observed to be a live scoring term.
            // A `NotLogLikelihood` here would mean the check above and the graph disagree.
            let _ = self.graph.unmark_log_likelihood(density);
        }
        self.debug_validate_state();
    }

    /// Record a least-squares factor, validating it against the graph as it stands.
    ///
    /// This is the runtime replacement for what used to be a trait bound. The bound could
    /// assert that a *distribution* had a residual; it could not assert that the ids handed
    /// over name the nodes the graph actually holds, which is the failure that costs a
    /// caller a `NodeNotFound` at solve time rather than at build time. So:
    ///
    /// - `density` must be in the arena and still scoring — else
    ///   [`SkoroError::NotLogLikelihood`];
    /// - `residual` must be in the arena and continuous — else [`SkoroError::NodeNotFound`]
    ///   or [`SkoroError::NotContinuous`], a discrete node having no tangent space and so
    ///   contributing no rows to `J`;
    /// - every id in `normalizer_parents`, `location` and `whitening` must exist — else
    ///   [`SkoroError::DanglingParent`].
    ///
    /// What is *not* checked is that the density really is `−½‖r‖² + n`. Nothing here can
    /// check that, and it stays the recorder's obligation.
    ///
    /// Recording pins the factor's nodes: neither `cleanup` nor [`remove`](Self::remove) may
    /// take one until the record leaves.
    pub fn record_factor(&mut self, record: FactorRecord) -> SkoroResult<()> {
        if !self.graph.is_scoring(record.density) {
            return Err(if self.graph.contains(record.density) {
                SkoroError::NotLogLikelihood { id: record.density }
            } else {
                SkoroError::NodeNotFound { id: record.density }
            });
        }
        match self.graph.node(record.residual) {
            None => {
                return Err(SkoroError::NodeNotFound {
                    id: record.residual,
                });
            }
            Some(node) if !node.is_continuous() => {
                return Err(SkoroError::NotContinuous {
                    id: record.residual,
                });
            }
            Some(_) => {}
        }
        for parent in record
            .normalizer_parents
            .iter()
            .chain(record.location.iter())
            .chain(record.whitening.iter())
        {
            if !self.graph.contains(*parent) {
                return Err(SkoroError::DanglingParent { parent_id: *parent });
            }
        }
        self.registry.record(record);
        self.debug_validate_state();
        Ok(())
    }

    /// What this server has been told about its factors, pruned against the graph at every
    /// mutation above.
    pub fn registry(&self) -> &ResidualRegistry {
        &self.registry
    }

    /// The rows of `J`, in the order the factors were recorded.
    ///
    /// The registry names the residual nodes and [`TangentLayout::from_ids`] sizes them by
    /// manifold dimension, so an `SE3` residual is 6 rows wide and never its 12-wide storage.
    pub fn residual_layout(&self) -> SkoroResult<TangentLayout> {
        let ids: Vec<Id> = self.registry.residual_ids().collect();
        TangentLayout::from_ids(&self.graph, &ids)
    }

    /// The registry invariants, checked after every registry mutation in debug builds — the
    /// same regime [`Graph::validate_indices`] runs under, one layer up.
    ///
    /// Only the registry-against-graph half is asserted here. The registry's own index is
    /// not a second copy to cross-check: it is private to [`ResidualRegistry`] and written
    /// beside every record that enters or leaves.
    fn debug_validate_state(&self) {
        #[cfg(debug_assertions)]
        for record in self.registry.factors() {
            assert!(
                self.graph.contains(record.residual),
                "a record names residual {:?}, which is not in the arena",
                record.residual
            );
            assert!(
                self.graph.is_scoring(record.density),
                "a record names density {:?}, which is not a live likelihood term",
                record.density
            );
        }
    }

    /// Reclaim every node the model can no longer reach.
    ///
    /// A mark-and-sweep from an explicit root set, run against the graph's shape as it stands
    /// rather than against anything recorded when a handle died. Three consequences worth
    /// stating, because the previous collector had none of them:
    ///
    /// - **Nothing has to report an orphaning.** Losing an edge is not an event a later pass
    ///   could reconstruct, which is why the old candidate-queue collector needed every
    ///   mutation to hand back what it severed. Reachability is not an event either: it is
    ///   just a property of the graph, recomputed here.
    /// - **Every node kind is collectable.** A `Constant` or `Parameter` nothing reaches is
    ///   garbage exactly as a `Mapped` node is. The old rule spared them on the grounds that
    ///   a caller might hold a bare [`Id`], and leaked every temporary constant to do it.
    /// - **The sweep cannot fail.** See below.
    ///
    /// **The roots** are the ids a live [`Q`](crate::q::Q) names, the graph's likelihood
    /// terms, and the nodes a [`FactorRecord`] names. Private nodes — a stacked tangent's
    /// anchors and per-block tangents — are deliberately *not* roots: they are ancestors of
    /// the `δ` node the caller holds, so they are marked while `δ` lives and swept when it
    /// does not. Rooting them would strand `2k + 1` nodes per tick of a sliding window.
    ///
    /// **The one edge that is not a parent edge**: marking a log-likelihood also marks the
    /// quantity it scores. Nothing in the [`Distribution`] contract requires a density to be
    /// built over its own quantity, so a density that scores `x` without reading it would
    /// leave `x` unmarked once `x`'s handle dropped — and taking `x` would make
    /// [`Graph::remove`] retire a live scoring density, changing the objective inside a
    /// garbage collection with no error at all.
    ///
    /// **Why the sweep cannot fail.** [`Graph::remove`] is leaf-only: it refuses a batch any
    /// survivor points into. The garbage set is closed downward, so it never presents one. If
    /// a garbage node `n` had a surviving child `c`, then `c` was marked, and marking walks
    /// `c` to its parents, which would have marked `n`.
    ///
    /// **The contract this fixes on the caller.** A `Q` is the ownership token; a bare `Id`
    /// is a weak reference. An `Id` kept across a `cleanup()` may name nothing afterwards,
    /// which every id-taking method already reports as `NodeNotFound`.
    ///
    /// **When this runs, and when it does not.** Here, and at the end of the mutations that
    /// reduce reachability — [`remove`](Self::remove), [`remove_closure`](Self::remove_closure),
    /// [`demote_to_parameter`](Self::demote_to_parameter). Deliberately **not** on the way in:
    /// adding a node cannot create garbage, and sweeping before every `add_*` — which is what
    /// the previous collector did — makes building an `n`-node model cost `O(n²)`. So a
    /// workload that only grows, building temporaries and dropping their handles without ever
    /// removing or demoting anything, holds that garbage until it asks. `graph_node_count` is
    /// the bound such a caller watches, and this is what it calls. A sliding window, which
    /// retires every tick, never has to.
    pub fn cleanup(&mut self) {
        if !self.handles.take_dirty() {
            return;
        }

        // ---- Mark: the ancestor closure of the roots ----
        let mut marked: HashSet<Id> = HashSet::new();
        let mut queue: VecDeque<Id> = VecDeque::new();
        let mut roots = self.handles.roots();
        roots.extend_from_slice(self.graph.log_likelihoods());
        roots.extend(
            self.registry
                .factors()
                .iter()
                .flat_map(|record| [record.density, record.residual]),
        );
        for id in roots {
            if self.graph.contains(id) && marked.insert(id) {
                queue.push_back(id);
            }
        }
        while let Some(id) = queue.pop_front() {
            let Some(node) = self.graph.node(id) else {
                continue;
            };
            for parent in node.parent_ids() {
                if marked.insert(*parent) {
                    queue.push_back(*parent);
                }
            }
            // The scored quantity of a density, which need not be among its parents.
            if let Some(stochastic) = self.graph.stochastic_of(id)
                && marked.insert(stochastic)
            {
                queue.push_back(stochastic);
            }
        }

        // ---- Sweep: everything else, children before parents ----
        let garbage: Vec<Id> = self
            .graph
            .nodes()
            .map(|(id, _)| *id)
            .filter(|id| !marked.contains(id))
            .collect();
        if garbage.is_empty() {
            return;
        }
        let ordered: Vec<Id> = self
            .graph
            .topological_sort(&garbage)
            .into_iter()
            .filter(|id| !marked.contains(id))
            .rev()
            .collect();
        self.graph
            .remove(&ordered)
            .expect("internal: the garbage set is closed downward, so `remove` cannot refuse it");
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
/// [`HandleRegistry`].
///
/// One guard exists per handle-family: `Q` clones share it, so a guard is created exactly
/// once per node and dropped when the last clone naming it is gone. Creation registers the
/// id as a collector root; the drop removes it.
///
/// The guard carries no opinion about collectability. It cannot: whether the node is still
/// reachable is a property of the graph at sweep time, not of the handle, so this type can
/// never be the reason a live node is collected or a dead one panics the collector.
pub(crate) struct NodeDropGuard {
    id: Id,
    registry: Arc<HandleRegistry>,
}

impl NodeDropGuard {
    pub(crate) fn new(id: Id, registry: &Arc<HandleRegistry>) -> Arc<Self> {
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
        let id = server.add_constant(quantity_type, value);
        Q::new(id, self.clone(), NodeDropGuard::new(id, &server.handles))
    }

    /// Add a fixed, immutable quantity that differs across the batch: one value per batch row,
    /// in order, rather than one value broadcast to every row.
    ///
    /// See [`Graph::add_batched_constant`](skoro_graph::graph::Graph::add_batched_constant) for
    /// the batch-pinning this implies.
    pub fn add_batched_constant<K, T, D>(
        &self,
        quantity_type: T,
        values: &[D],
    ) -> SkoroResult<Q<B, T>>
    where
        K: Kind<B>,
        T: QuantityType<B, K> + Packable<K::Elem, D>,
        D: 'static,
    {
        let mut server = self.server.write().unwrap();
        let id = server.add_batched_constant(quantity_type, values)?;
        Ok(Q::new(
            id,
            self.clone(),
            NodeDropGuard::new(id, &server.handles),
        ))
    }

    /// Add a free quantity to the graph of the model server, continuous or
    /// discrete.
    pub fn add_parameter<K, T>(&self, quantity_type: T) -> Q<B, T>
    where
        K: Kind<B>,
        T: QuantityType<B, K>,
    {
        let mut server = self.server.write().unwrap();
        let id = server.add_parameter(quantity_type);
        Q::new(id, self.clone(), NodeDropGuard::new(id, &server.handles))
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
        let x = Q::new(id, self.clone(), NodeDropGuard::new(id, &server.handles));

        // maybe slow, but currently the price for ergonomic interface: we need to drop the server
        // to add the log-density term (or else deadlock will occur).
        drop(server);
        let density = distr.log_density(&x)?;

        // again, slow... but now we mark the mapped quantity as a density.
        let mut server = self.server.write().unwrap();
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
    /// Also on [`ModelBuilderOps`](crate::client_ops::ModelBuilderOps), which is the bound a
    /// joint-factor builder takes: this adds `2k + 1` nodes, so it belongs on the tier that
    /// grows a model and not on the one an estimator holds.
    pub fn stacked_tangent(&self, at: &PointMap<B>, ids: &[Id]) -> SkoroResult<Q<B, EuclideanD>> {
        let mut server = self.server.write().unwrap();
        let id = server.stacked_tangent(at, ids)?;
        Ok(Q::new(
            id,
            self.clone(),
            NodeDropGuard::new(id, &server.handles),
        ))
    }

    pub(crate) fn add_mapped<K, T>(
        &self,
        quantity_type: T,
        op: impl DynGraphOperation<B>,
        parent_ids: Vec<Id>,
    ) -> SkoroResult<Q<B, T>>
    where
        K: Kind<B>,
        T: QuantityType<B, K>,
    {
        let mut server = self.server.write().unwrap();
        let id = server.add_mapped(quantity_type, op, parent_ids)?;
        Ok(Q::new(
            id,
            self.clone(),
            NodeDropGuard::new(id, &server.handles),
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
