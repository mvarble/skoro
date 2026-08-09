use crossbeam_queue::SegQueue;
use dashmap::DashMap;
use lazy_static::lazy_static;
use std::any::Any;
use std::collections::HashMap;
use std::marker::PhantomData;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, RwLock, Weak};

use skoro_backend::{Backend, CotangentFrame, Point, TangentFrame, TangentVector};
use skoro_core::kind::Kind;
use skoro_core::{
    error::SkoroResult,
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
    quantity::{ContinuousQuantity, QuantityType},
};
use skoro_operation::ValueMap;
use skoro_riemann::real::Real;

use crate::{distribution::Distribution, q::Q};

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
    pub graph: Graph<B>,
    pub device: B::Device,
    cleanup_queue: Arc<SegQueue<Id>>,
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
            cleanup_queue: Arc::new(SegQueue::new()),
        }
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
    ) -> Id
    where
        K: Kind<B>,
        T: QuantityType<B, K>,
    {
        self.graph.add_mapped(quantity_type, op, parent_ids)
    }

    /// Mark a mapped quantity as a density for the stochastic quantity.
    pub fn mark_density<T>(
        &mut self,
        density: &Q<B, Real>,
        stochastic: &Q<B, T>,
    ) -> SkoroResult<()> {
        self.graph.mark_density(density.id, stochastic.id)?;
        density.guard.is_mapped_leaf.store(false, Ordering::SeqCst);
        Ok(())
    }

    /// Mark a mapped quantity as a likelihood.
    pub fn mark_likelihood(&mut self, likelihood: &Q<B, Real>) -> SkoroResult<()> {
        self.graph.mark_likelihood(likelihood.id)?;
        likelihood
            .guard
            .is_mapped_leaf
            .store(false, Ordering::SeqCst);
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
        self.graph
            .update_stochastic(tag.id, new_sampler, new_parents)
    }

    /// Demote a stochastic quantity to a free parameter, keeping its identity.
    ///
    /// The quantity loses its sampler, its parents, and its own log-density; its
    /// [`Id`], quantity type, and every dependent quantity are untouched. Use this
    /// when a quantity's distribution is superseded by a joint likelihood factor
    /// over several quantities — a marginalization whose result does not factorize.
    ///
    /// Returns an error if the quantity does not exist or is neither stochastic nor
    /// already a parameter.
    #[inline(always)]
    pub fn demote_to_parameter<T>(&mut self, tag: &Q<B, T>) -> SkoroResult<()> {
        self.graph.demote_to_parameter(tag.id)
    }

    /// Removes quantities, verifying they have no children which would be kept in the graph.
    pub fn remove(&mut self, ids: &[Id]) -> SkoroResult<Vec<Node<B>>> {
        self.graph.remove(ids)
    }

    /// Remove all non-likelihood mapped quantities which have no more references.
    pub fn cleanup(&mut self) {
        let mut ids = Vec::with_capacity(self.cleanup_queue.len());
        while let Some(id) = self.cleanup_queue.pop() {
            ids.push(id);
        }
        if !ids.is_empty() {
            // Cleanup only removes leaf mapped nodes the model itself queued; a
            // failure here would indicate an internal bookkeeping bug.
            self.graph
                .remove(&ids)
                .expect("internal: cleanup removed a referenced node");
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

/// Drop guard for [`Q`] instances which allows us to perform graph cleanup of no-longer referenced
/// mapped nodes.
pub(crate) struct NodeDropGuard {
    id: Id,
    is_mapped_leaf: AtomicBool,
    cleanup_queue: Arc<SegQueue<Id>>,
}

impl Drop for NodeDropGuard {
    fn drop(&mut self) {
        if self.is_mapped_leaf.load(Ordering::SeqCst) {
            self.cleanup_queue.push(self.id);
        }
    }
}

impl NodeDropGuard {
    pub(crate) fn set_mapped_leaf(&self, is_mapped_leaf: bool) {
        self.is_mapped_leaf.store(is_mapped_leaf, Ordering::SeqCst);
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

    /// Returns the strong reference to the underlying server.
    pub fn get_server(&self) -> Arc<RwLock<ModelServer<B>>> {
        self.server.clone()
    }

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
            Arc::new(NodeDropGuard {
                id,
                is_mapped_leaf: AtomicBool::new(false),
                cleanup_queue: server.cleanup_queue.clone(),
            }),
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
            Arc::new(NodeDropGuard {
                id,
                is_mapped_leaf: AtomicBool::new(false),
                cleanup_queue: server.cleanup_queue.clone(),
            }),
        )
    }

    /// Adds a stochastic quantity to the graph of the model server, continuous
    /// or discrete.
    ///
    /// Returns an error if the distribution references a parent that does not
    /// exist in the model.
    pub fn add_quantity<K, D>(&self, distr: D) -> SkoroResult<Q<B, D::Qty>>
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
            Arc::new(NodeDropGuard {
                id,
                is_mapped_leaf: AtomicBool::new(false),
                cleanup_queue: server.cleanup_queue.clone(),
            }),
        );

        // maybe slow, but currently the price for ergonomic interface: we need to drop the server
        // to add the log-density term (or else deadlock will occur).
        drop(server);
        let density = distr.log_density(&x);

        // again, slow... but now we mark the mapped quantity as a density.
        let mut server = self.server.write().unwrap();
        server.cleanup();
        // The density is a freshly-created real-valued mapped node, so marking
        // it can only fail through an internal bug.
        server
            .mark_density(&density, &x)
            .expect("internal: failed to mark freshly-created density");
        Ok(x)
    }

    pub(crate) fn add_mapped<K, T>(
        &self,
        quantity_type: T,
        op: impl GraphOperation<B>,
        parent_ids: Vec<Id>,
    ) -> Q<B, T>
    where
        K: Kind<B>,
        T: QuantityType<B, K>,
    {
        let mut server = self.server.write().unwrap();
        server.cleanup();
        let id = server.add_mapped(quantity_type, op, parent_ids);
        Q::new(
            id,
            self.clone(),
            Arc::new(NodeDropGuard {
                id,
                is_mapped_leaf: AtomicBool::new(true),
                cleanup_queue: server.cleanup_queue.clone(),
            }),
        )
    }

    /// Mark the mapped quantity as a likelihood.
    pub fn mark_likelihood(&self, likelihood: &Q<B, Real>) -> SkoroResult<()> {
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
    pub fn update_quantity<K, D>(&self, tag: &Q<B, D::Qty>, distr: D) -> SkoroResult<()>
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
        let density = distr.log_density(tag);

        // again, slow... but now we mark the mapped quantity as a density.
        let mut server = self.server.write().unwrap();
        server.cleanup();
        // The density is a freshly-created real-valued mapped node, so marking
        // it can only fail through an internal bug.
        server
            .mark_density(&density, tag)
            .expect("internal: failed to mark freshly-created density");
        Ok(())
    }

    /// Demote a stochastic quantity to a free parameter, keeping its identity.
    ///
    /// See [`ModelServer::demote_to_parameter`].
    pub fn demote_to_parameter<T>(&self, tag: &Q<B, T>) -> SkoroResult<()> {
        let mut server = self.server.write().unwrap();
        server.demote_to_parameter(tag)?;
        server.cleanup();
        Ok(())
    }

    /// Removes quantities, verifying they have no children which would be kept in the graph.
    pub fn remove(&self, ids: &[Id]) -> SkoroResult<Vec<Node<B>>> {
        self.server.write().unwrap().remove(ids)
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
