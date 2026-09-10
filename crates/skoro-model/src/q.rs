use std::{collections::HashMap, marker::PhantomData, sync::Arc};

use skoro_backend::{Backend, CotangentFrame, Point, TangentFrame, TangentVector};
use skoro_core::{error::SkoroResult, id::Id, kind::Kind, pack::Packable};
use skoro_graph::{
    graph::{PointEvalOrder1, PointEvalOrder2, PointHvpEval, PointJvpEval, PointVjpEval},
    quantity::{ContinuousQuantity, QuantityType},
};
use skoro_operation::ValueMap;

use crate::model::{ModelClient, NodeDropGuard};

/// A lightweight proxy to a specific quantity tracked inside a `ModelClient`. This holds the type
/// state of the underlying quantity type (manifold or discrete).
///
/// `Q` is deliberately generic over an unbounded `T`: the kind of the quantity
/// (continuous or discrete) is fixed by which [`QuantityType`] instantiation
/// `T` implements, and the kind-specific methods live on appropriately-bounded
/// `impl` blocks.
pub struct Q<B: Backend, T> {
    /// The unique identifier of the node in the ModelServer's Graph.
    ///
    /// Private so that `Q` can change representation later without a breaking
    /// release; read it with [`Q::id`].
    pub(crate) id: Id,
    /// The client reference to perform operations against the model server.
    ///
    /// Private, and load-bearing: a public field here would hand every holder of a
    /// quantity the unrestricted model surface, defeating any facade that returned it.
    pub(crate) client: ModelClient<B>,
    pub(crate) guard: Arc<NodeDropGuard>,
    _phantom: PhantomData<T>,
}

impl<B: Backend, T> std::fmt::Debug for Q<B, T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Q")
            .field("id", &self.id)
            .field("server_id", &self.client.server_id)
            .finish_non_exhaustive()
    }
}

impl<B: Backend, T> Clone for Q<B, T> {
    fn clone(&self) -> Self {
        Self {
            id: self.id,
            client: self.client.clone(),
            guard: self.guard.clone(),
            _phantom: PhantomData,
        }
    }
}

impl<B: Backend, T> std::cmp::PartialEq for Q<B, T> {
    fn eq(&self, other: &Self) -> bool {
        self.id == other.id && self.client.server_id == other.client.server_id
    }
}

impl<B: Backend, T> std::hash::Hash for Q<B, T> {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.id.hash(state);
        self.client.server_id.hash(state);
    }
}

impl<B: Backend, T> Q<B, T> {
    /// Creates a new Q proxy.
    pub(crate) fn new(id: Id, client: ModelClient<B>, guard: Arc<NodeDropGuard>) -> Self {
        Self {
            id,
            client,
            guard,
            _phantom: PhantomData,
        }
    }

    /// The identifier of the node this quantity names.
    #[inline(always)]
    pub fn id(&self) -> Id {
        self.id
    }

    /// Render the quantity unobserved.
    #[inline(always)]
    pub fn unobserve(&self) -> SkoroResult<()> {
        self.client.unobserve(self)
    }
}

/// Methods available on quantities of either kind. The kind parameter `K` is
/// inferred from the concrete quantity type `M`.
impl<B: Backend, T> Q<B, T> {
    /// Render the quantity observed at the specified value.
    #[inline(always)]
    pub fn observe<K, D>(&self, value: &D) -> SkoroResult<()>
    where
        K: Kind<B>,
        T: QuantityType<B, K> + Packable<K::Elem, D>,
        D: 'static,
    {
        self.client.observe(self, value)
    }

    /// Get the quantity type within the graph.
    ///
    /// This panics if the node's stored type does not match `M`; since a `Q<B,
    /// M>` proxy is only ever constructed for a node of type `M`, that can only
    /// happen through an internal bug. Use [`ModelClient::get_quantity_type`]
    /// for a recoverable variant.
    #[inline(always)]
    pub fn get_quantity_type<K>(&self) -> T
    where
        K: Kind<B>,
        T: QuantityType<B, K>,
    {
        self.client
            .get_quantity_type(self)
            .expect("internal: quantity type of a Q proxy should always match")
    }
}

/// Evaluations returning continuous (float) tensors are continuous-only.
impl<B: Backend, M: ContinuousQuantity<B>> Q<B, M> {
    /// Evaluate the quantity.
    #[inline(always)]
    pub fn evaluate(&self, points: &ValueMap<B>) -> SkoroResult<Point<B>> {
        self.client.evaluate(points, self)
    }

    /// The JVP evaluation [`evaluate`](Self::evaluate); if you only want gradients, use
    /// [`evaluate_ord1`](Self::evaluate_ord1).
    #[inline(always)]
    pub fn evaluate_jvp(
        &self,
        points: &ValueMap<B>,
        tangents: &HashMap<Id, TangentFrame<B>>,
    ) -> SkoroResult<PointJvpEval<B>> {
        self.client.evaluate_jvp(points, tangents, self)
    }

    /// The VJP evaluation [`evaluate`](Self::evaluate); if you only want gradients, use
    /// [`evaluate_ord1`](Self::evaluate_ord1).
    #[inline(always)]
    pub fn evaluate_vjp(
        &self,
        points: &ValueMap<B>,
        cotangents: &HashMap<Id, CotangentFrame<B>>,
    ) -> SkoroResult<PointVjpEval<B>> {
        self.client.evaluate_vjp(points, cotangents, self)
    }

    /// The HVP evaluation [`evaluate`](Self::evaluate); if you only want gradients and
    /// Hessian-products, instead use [`evaluate_ord2`](Self::evaluate_ord2).
    #[inline(always)]
    pub fn evaluate_hvp(
        &self,
        points: &ValueMap<B>,
        tangents: &HashMap<Id, TangentVector<B>>,
        ord1_cotangents: &HashMap<Id, CotangentFrame<B>>,
        ord2_cotangents: &HashMap<Id, CotangentFrame<B>>,
    ) -> SkoroResult<PointHvpEval<B>> {
        self.client
            .evaluate_hvp(points, tangents, ord1_cotangents, ord2_cotangents, self)
    }

    /// The evaluation [`evaluate`](Self::evaluate) with the first-order gradient calculation.
    #[inline(always)]
    pub fn evaluate_ord1(&self, points: &ValueMap<B>) -> SkoroResult<PointEvalOrder1<B>> {
        self.client.evaluate_ord1(points, self)
    }

    /// The evaluation [`evaluate`](Self::evaluate) with the first-order gradient and second-order
    /// Hessian-product calculation.
    #[inline(always)]
    pub fn evaluate_ord2(
        &self,
        points: &ValueMap<B>,
        tangents: &HashMap<Id, TangentVector<B>>,
    ) -> SkoroResult<PointEvalOrder2<B>> {
        self.client.evaluate_ord2(points, tangents, self)
    }
}

/// A type-erased reference to a quantity: its id, plus the same drop guard the [`Q`] it was made
/// from carries.
///
/// This is what lets one collection hold references to quantities of different types, as a
/// [`Factor`](crate::distribution::Factor)'s bindings do. While any clone of a `Handle` (or of
/// the `Q` it came from) lives, the collector treats the node as a root.
pub struct Handle<B: Backend> {
    pub(crate) id: Id,
    pub(crate) client: ModelClient<B>,
    pub(crate) guard: Arc<NodeDropGuard>,
}

impl<B: Backend> Handle<B> {
    /// The identifier of the node this handle names.
    #[inline(always)]
    pub fn id(&self) -> Id {
        self.id
    }
}

impl<B: Backend, T> From<&Q<B, T>> for Handle<B> {
    fn from(q: &Q<B, T>) -> Self {
        Self {
            id: q.id,
            client: q.client.clone(),
            guard: q.guard.clone(),
        }
    }
}

impl<B: Backend> Clone for Handle<B> {
    fn clone(&self) -> Self {
        Self {
            id: self.id,
            client: self.client.clone(),
            guard: self.guard.clone(),
        }
    }
}

impl<B: Backend> std::fmt::Debug for Handle<B> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Handle")
            .field("id", &self.id)
            .field("server_id", &self.client.server_id)
            .finish_non_exhaustive()
    }
}

// `add_factor` and `bind` deliberately do not live here. Attaching an arbitrary factor to
// any real-valued quantity would let a holder of a `Q` handed out by a restricted model
// widen that model behind its back. They are on `ModelBuilderOps` instead, which a
// restricting facade can decline to implement.
