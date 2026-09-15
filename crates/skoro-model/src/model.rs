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
    binding::Role,
    compiler::{ForwardOutput, HvpForwardOutput, JvpForwardOutput, VjpForwardOutput},
    graph::{
        EvalOrder1, EvalOrder2, Graph, PointEvalOrder1, PointEvalOrder2, PointHvpEval,
        PointJvpEval, PointVjpEval,
    },
    node::Node,
    operation::DynGraphOperation,
    point_map::PointMap,
    quantity::{ContinuousQuantity, QuantityType},
};
use skoro_operation::ValueMap;
use skoro_riemann::euclidean::EuclideanD;
use skoro_riemann::real::Real;

use crate::{
    cleanup::HandleRegistry,
    distribution::{Distribution, Factor},
    q::Q,
    roles::ARGUMENT,
};

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

    /// Mark a mapped quantity as the proposal-only density for the stochastic quantity: a
    /// summand of the generative model `q(x|z)` but not of the log-density. See
    /// [`Graph::mark_log_proposal`](skoro_graph::graph::Graph::mark_log_proposal).
    pub fn mark_proposal<T>(
        &mut self,
        density: &Q<B, Real>,
        stochastic: &Q<B, T>,
    ) -> SkoroResult<()> {
        self.graph.mark_log_proposal(density.id, stochastic.id)
    }

    /// Mark a mapped quantity as a likelihood.
    pub fn mark_likelihood(&mut self, likelihood: &Q<B, Real>) -> SkoroResult<()> {
        self.graph.mark_log_likelihood(likelihood.id)
    }

    /// Record a binding: `to` plays `role` for `from`. See
    /// [`Graph::bind`](skoro_graph::graph::Graph::bind).
    pub fn bind(&mut self, from: Id, role: Role, to: Id) -> SkoroResult<()> {
        self.graph.bind(from, role, to)
    }

    /// Mark `factor`'s log-density as a free-standing likelihood term, bind
    /// [`ARGUMENT`] to each of `arguments` in order, then bind every role `factor` declares.
    ///
    /// Returns [`SkoroError::ShapeMismatch`] before changing anything if `arguments` is empty: a
    /// term has at least one argument.
    ///
    /// The server cannot tell whether `factor` was built against it; a handle from another model
    /// names an id in that model's graph. [`ModelClient::add_factor`] checks that, so prefer it.
    pub fn add_factor(&mut self, factor: &Factor<B>, arguments: &[Id]) -> SkoroResult<()> {
        if arguments.is_empty() {
            return Err(SkoroError::ShapeMismatch {
                expected: 1,
                found: 0,
            });
        }
        self.mark_likelihood(&factor.log_density)?;
        self.bind_factor(factor, arguments)
    }

    /// Bind [`ARGUMENT`] to each of `arguments`, then every role `factor` declares.
    fn bind_factor(&mut self, factor: &Factor<B>, arguments: &[Id]) -> SkoroResult<()> {
        let density = factor.log_density.id;
        for argument in arguments {
            self.bind(density, ARGUMENT, *argument)?;
        }
        for (role, handle) in &factor.bindings {
            self.bind(density, *role, handle.id)?;
        }
        Ok(())
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
    /// The demoted quantity's log-density leaves the objective here, and its bindings stop
    /// counting with it: the rows of `J` shrink exactly when the term stops scoring.
    ///
    /// Everything the demotion made unreachable is collected before returning — the orphaned
    /// density, the subgraph it was holding up, the residual and Fisher root it bound, and the
    /// parent edges the old distribution owned. The graph reports each of those, because none of them
    /// is an event a later pass could reconstruct.
    ///
    /// Returns an error if the quantity does not exist or is neither stochastic nor
    /// already a parameter.
    #[inline(always)]
    pub fn demote_to_parameter<T>(&mut self, tag: &Q<B, T>) -> SkoroResult<()> {
        self.graph.demote_to_parameter(tag.id)?;
        self.handles.mark_dirty();
        self.cleanup();
        Ok(())
    }

    /// Removes quantities, verifying they have no children which would be kept in the graph.
    ///
    /// Refuses a node that a node outside the batch still binds
    /// ([`SkoroError::NodeStillBound`]), for the same reason it refuses one another node still
    /// points into as a parent: this is the primitive that will not break a reference, and a
    /// binding is a reference. That refusal is what keeps `remove` safe on
    /// [`ModelClientOps`](crate::client_ops::ModelClientOps), which an estimator holds — a bound
    /// residual is childless, so without the check a `remove` could silently leave a term
    /// binding a node that is gone. Retiring a factor is
    /// [`remove_closure`](Self::remove_closure), one tier up.
    ///
    /// The removal is also a collection point: anything it orphaned is swept before
    /// returning, so the model is clean after every retire.
    pub fn remove(&mut self, ids: &[Id]) -> SkoroResult<Vec<Node<B>>> {
        let removed = self.graph.remove(ids)?;
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
        self.handles.mark_dirty();
        self.cleanup();
        Ok(removed)
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
    /// **The roots** are the ids a live [`Q`](crate::q::Q) names and the graph's likelihood
    /// terms. Private nodes — a stacked tangent's
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
    /// **Bindings are references too.** Marking a node marks every node it binds, so a term's
    /// residual and Fisher root live exactly as long as the term does, with no handle of their
    /// own.
    ///
    /// **Why the sweep cannot fail.** [`Graph::remove`] is leaf-only: it refuses a batch any
    /// survivor points into. The garbage set is closed downward, so it never presents one. If
    /// a garbage node `n` had a surviving child `c`, then `c` was marked, and marking walks
    /// `c` to its parents, which would have marked `n`. The same holds for a surviving node
    /// that binds `n`, so the sweep never trips [`SkoroError::NodeStillBound`] either.
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
            // What the node binds: a binding is a reference, even though it is not a data edge.
            for (_, to) in self.graph.bindings_of(id) {
                if marked.insert(*to) {
                    queue.push_back(*to);
                }
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
    ///
    /// The call is atomic. If the factor cannot be built or recorded — `distr.factor` fails,
    /// the factor belongs to another model, or a binding is refused — the stochastic node and
    /// everything built from it are removed before the error is returned, rather than left for
    /// the next [`cleanup`](Self::cleanup).
    pub(crate) fn add_quantity<K, D>(&self, distr: D) -> SkoroResult<Q<B, D::Qty>>
    where
        K: Kind<B>,
        D: Distribution<B, K>,
    {
        self.add_quantity_with(distr, true)
    }

    /// Adds a stochastic quantity whose density scores it under the *proposal* only.
    ///
    /// The quantity is generated exactly as [`add_quantity`](Self::add_quantity)'s is, but its
    /// density is a summand of `q(x|z)` and not of `p(x|z)`. What the posterior says about the
    /// quantity, if anything, is supplied separately with
    /// [`add_factor`](Self::add_factor) — which is the point: it lets a model sample from a
    /// distribution it can draw from while scoring one it can only evaluate.
    pub(crate) fn add_proposal<K, D>(&self, distr: D) -> SkoroResult<Q<B, D::Qty>>
    where
        K: Kind<B>,
        D: Distribution<B, K>,
    {
        self.add_quantity_with(distr, false)
    }

    fn add_quantity_with<K, D>(&self, distr: D, in_log_density: bool) -> SkoroResult<Q<B, D::Qty>>
    where
        K: Kind<B>,
        D: Distribution<B, K>,
    {
        // add stochastic quantity
        let mut server = self.server.write().unwrap();
        let id = server.add_stochastic(&distr)?;
        let x = Q::new(id, self.clone(), NodeDropGuard::new(id, &server.handles));

        // maybe slow, but currently the price for ergonomic interface: we need to drop the server
        // to build the factor (or else deadlock will occur).
        drop(server);
        match self.record_factor(&distr, &x, in_log_density) {
            Ok(()) => Ok(x),
            Err(error) => {
                // Nothing outside this call can reach `x` or anything built from it yet, so its
                // downstream closure is exactly what the failed call added. Its handle goes first,
                // so the collection that ends the removal does not count it as live. Should the
                // removal itself fail, the nodes stay unreachable and the next `cleanup` takes
                // them; the error worth reporting is still the first one.
                drop(x);
                let _ = self.server.write().unwrap().remove_closure(&[id]);
                Err(error)
            }
        }
    }

    /// Build `distr`'s factor over `x` and record it: the density mark (of the log-density, or of
    /// the proposal alone when `in_log_density` is false), the `ARGUMENT` bindings to `x` and to
    /// each parent, and every binding the factor declares.
    fn record_factor<K, D>(
        &self,
        distr: &D,
        x: &Q<B, D::Qty>,
        in_log_density: bool,
    ) -> SkoroResult<()>
    where
        K: Kind<B>,
        D: Distribution<B, K>,
    {
        let factor = distr.factor(x)?;
        self.check_factor_model(&factor)?;

        // again, slow... but now we mark the mapped quantity as a density and record its
        // bindings. `factor` is still alive here, so nothing it binds can have been collected.
        let mut server = self.server.write().unwrap();
        if in_log_density {
            server.mark_density(&factor.log_density, x)?;
        } else {
            server.mark_proposal(&factor.log_density, x)?;
        }
        let arguments: Vec<Id> = std::iter::once(x.id)
            .chain(distr.parent_ids().iter().copied())
            .collect();
        server.bind_factor(&factor, &arguments)
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

    /// Record a binding: `to` plays `role` for `from`. See [`ModelServer::bind`].
    pub fn bind(&self, from: Id, role: Role, to: Id) -> SkoroResult<()> {
        self.server.write().unwrap().bind(from, role, to)
    }

    /// Attach a free-standing likelihood factor over `arguments`, with the bindings it
    /// declares. See [`ModelServer::add_factor`].
    ///
    /// Returns [`SkoroError::ModelMismatch`] before changing anything if the factor's
    /// log-density or any node it binds belongs to another model.
    pub fn add_factor(&self, factor: Factor<B>, arguments: &[Id]) -> SkoroResult<Q<B, Real>> {
        self.check_factor_model(&factor)?;
        self.server
            .write()
            .unwrap()
            .add_factor(&factor, arguments)?;
        Ok(factor.log_density)
    }

    /// Refuse a factor built against another model. Its handles name ids in that model's graph,
    /// which here would name unrelated nodes — or none — and be bound without complaint.
    fn check_factor_model(&self, factor: &Factor<B>) -> SkoroResult<()> {
        let foreign = std::iter::once(&factor.log_density.client)
            .chain(factor.bindings.iter().map(|(_, handle)| &handle.client))
            .find(|client| client.server_id != self.server_id);
        match foreign {
            Some(client) => Err(SkoroError::ModelMismatch {
                lhs: self.server_id,
                rhs: client.server_id,
            }),
            None => Ok(()),
        }
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
