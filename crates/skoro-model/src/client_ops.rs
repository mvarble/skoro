//! The distribution-agnostic client capability trait.
//!
//! [`ModelClientOps`] factors out the model-interface surface that has nothing
//! to do with which conditional distributions a model supports: adding
//! deterministic/free quantities, observing, evaluation, and generation. It is
//! the base trait that distribution-specific capability traits (e.g.
//! `GaussianModelClientOps` in `skoro-normal`) build on.
//!
//! All quantity-construction methods are *kind-generic*: the same
//! [`add_constant`](ModelClientOps::add_constant),
//! [`add_parameter`](ModelClientOps::add_parameter), and
//! [`observe`](ModelClientOps::observe) serve continuous and discrete
//! quantities alike, with the kind inferred from the quantity type. Only the
//! genuinely kind-specific surface is bounded: manifold/derivative methods
//! require [`ContinuousQuantity`].
//!
//! Code that is generic over "some client one can drive a model through" should
//! bound on `ModelClientOps`. Code that needs a *structural* guarantee about the
//! model (e.g. "all conditionals are Gaussian") should instead take the concrete
//! facade type (e.g. `GaussianModelClient`), not this trait.

use std::collections::HashMap;

use skoro_backend::{Backend, CotangentFrame, Point, TangentFrame, TangentVector};
use skoro_core::{error::SkoroResult, id::Id, kind::Kind, pack::Packable};
use skoro_graph::{
    compiler::{ForwardOutput, HvpForwardOutput, JvpForwardOutput, VjpForwardOutput},
    graph::{
        EvalOrder1, EvalOrder2, PointEvalOrder1, PointEvalOrder2, PointHvpEval, PointJvpEval,
        PointVjpEval,
    },
    node::Node,
    quantity::{ContinuousQuantity, QuantityType},
    tangent_layout::TangentLayout,
};
use skoro_operation::ValueMap;

use skoro_riemann::real::Real;

use crate::distribution::Distribution;
use crate::model::ModelClient;
use crate::q::Q;

/// Distribution-agnostic operations available on any model client.
///
/// This is intentionally "pragmatic and broad": it mirrors the common
/// model-building and evaluation surface of [`ModelClient`] (excluding the
/// lower-level `_jvp`/`_vjp`/`_hvp` directional primitives, which remain inherent
/// methods for now).
pub trait ModelClientOps<B: Backend> {
    // ----- quantity construction -----

    /// Add a fixed, immutable quantity, continuous or discrete.
    fn add_constant<K, T, D>(&self, quantity_type: T, value: &D) -> Q<B, T>
    where
        K: Kind<B>,
        T: QuantityType<B, K> + Packable<K::Elem, D>,
        D: 'static;

    /// Add a fixed, immutable quantity that differs across the batch: one value per batch row,
    /// in order, rather than one value broadcast to every row.
    ///
    /// A batch is a set of *independent* problems, so this is how each row gets its own
    /// constant -- its own linearization point, its own observation. The cost is that the value
    /// pins the batch: a model holding a batch-`n` constant can only be evaluated at batch `n`,
    /// and any other batch size is an error rather than a broadcast. Use
    /// [`add_constant`](Self::add_constant) for a value shared by every row.
    fn add_batched_constant<K, T, D>(&self, quantity_type: T, values: &[D]) -> Q<B, T>
    where
        K: Kind<B>,
        T: QuantityType<B, K> + Packable<K::Elem, D>,
        D: 'static;

    /// Add a free quantity, intended to be inferred, continuous or discrete.
    fn add_parameter<K, T>(&self, quantity_type: T) -> Q<B, T>
    where
        T: QuantityType<B, K>,
        K: Kind<B>;

    // ----- conditioning / mutation -----

    /// Impose a posterior by observing a quantity at a value.
    fn observe<K, T, D>(&self, q: &Q<B, T>, value: &D) -> SkoroResult<()>
    where
        K: Kind<B>,
        T: QuantityType<B, K> + Packable<K::Elem, D>,
        D: 'static;

    /// Undo the effect of [`observe`](Self::observe).
    fn unobserve<T>(&self, q: &Q<B, T>) -> SkoroResult<()>;

    /// Demote a stochastic quantity to a free parameter, keeping its identity.
    ///
    /// The quantity loses its sampler, its parents, and its own log-density, but
    /// keeps its `Id` and quantity type, so every quantity depending on it stays
    /// valid. This is how a quantity is re-rooted when its marginal does not
    /// factorize and the joint is supplied instead as a likelihood factor over
    /// several quantities.
    fn demote_to_parameter<T>(&self, q: &Q<B, T>) -> SkoroResult<()>;

    /// Remove quantities (which must have no retained children).
    fn remove(&self, ids: &[Id]) -> SkoroResult<Vec<Node<B>>>;

    /// Garbage-collect unreferenced mapped quantities.
    fn cleanup(&self);

    /// A [`TangentLayout`] over quantities of this model, in the given order.
    ///
    /// Layouts are built from a [`Graph`](skoro_graph::graph::Graph), which this trait
    /// deliberately does not hand out, so a client that wants one has to offer this. Every
    /// consumer of the flattened tangent API needs it: an optimizer names its free
    /// variables here, a filter its state and measurement blocks.
    ///
    /// Ordering is the caller's, because a block structure they chose is one they can
    /// reason about when reading a covariance or a Jacobian.
    fn tangent_layout(&self, ids: &[Id]) -> SkoroResult<TangentLayout>;

    /// Fetch the quantity type associated with a quantity.
    fn get_quantity_type<K, T>(&self, tag: &Q<B, T>) -> SkoroResult<T>
    where
        K: Kind<B>,
        T: QuantityType<B, K>;

    // ----- evaluation -----

    /// Evaluate the model log-density at the given values.
    fn log_density(&self, points: &ValueMap<B>) -> SkoroResult<Point<B>>;

    /// Log-density with first-order scores.
    fn log_density_ord1(&self, points: &ValueMap<B>) -> SkoroResult<PointEvalOrder1<B>>;

    /// Log-density with first-order scores and second-order sensitivities.
    fn log_density_ord2(
        &self,
        points: &ValueMap<B>,
        tangents: &HashMap<Id, TangentVector<B>>,
    ) -> SkoroResult<PointEvalOrder2<B>>;

    /// Evaluate a single (continuous) quantity.
    fn evaluate<T: ContinuousQuantity<B>>(
        &self,
        points: &ValueMap<B>,
        tag: &Q<B, T>,
    ) -> SkoroResult<Point<B>>;

    /// Evaluate a single quantity with first-order gradients.
    fn evaluate_ord1<T: ContinuousQuantity<B>>(
        &self,
        points: &ValueMap<B>,
        tag: &Q<B, T>,
    ) -> SkoroResult<PointEvalOrder1<B>>;

    /// Evaluate a single quantity with first- and second-order derivatives.
    fn evaluate_ord2<T: ContinuousQuantity<B>>(
        &self,
        points: &ValueMap<B>,
        tangents: &HashMap<Id, TangentVector<B>>,
        tag: &Q<B, T>,
    ) -> SkoroResult<PointEvalOrder2<B>>;

    /// Evaluate several quantities by id.
    fn evaluate_ids(&self, points: &ValueMap<B>, ids: &[Id]) -> SkoroResult<ForwardOutput<B>>;

    /// Evaluate several quantities by id with first-order gradients.
    fn evaluate_ids_ord1(&self, points: &ValueMap<B>, ids: &[Id]) -> SkoroResult<EvalOrder1<B>>;

    /// Evaluate several quantities by id with first- and second-order derivatives.
    fn evaluate_ids_ord2(
        &self,
        points: &ValueMap<B>,
        tangents: &HashMap<Id, TangentVector<B>>,
        ids: &[Id],
    ) -> SkoroResult<EvalOrder2<B>>;

    // ----- generation -----

    /// Sample every quantity in the model.
    fn generate(&self, batch_size: usize) -> SkoroResult<ForwardOutput<B>>;

    /// Sample conditioned on the given values.
    fn generate_from(
        &self,
        points: &ValueMap<B>,
        batch_size: usize,
    ) -> SkoroResult<ForwardOutput<B>>;

    // ----- lower-level directional primitives -----
    // (exposed for completeness; most callers use the `_ord1`/`_ord2` helpers)

    /// JVP of a single quantity.
    fn evaluate_jvp<T: ContinuousQuantity<B>>(
        &self,
        points: &ValueMap<B>,
        tangents: &HashMap<Id, TangentFrame<B>>,
        tag: &Q<B, T>,
    ) -> SkoroResult<PointJvpEval<B>>;

    /// VJP of a single quantity.
    fn evaluate_vjp<T: ContinuousQuantity<B>>(
        &self,
        points: &ValueMap<B>,
        cotangents: &HashMap<Id, CotangentFrame<B>>,
        tag: &Q<B, T>,
    ) -> SkoroResult<PointVjpEval<B>>;

    /// HVP of a single quantity.
    fn evaluate_hvp<T: ContinuousQuantity<B>>(
        &self,
        points: &ValueMap<B>,
        tangents: &HashMap<Id, TangentVector<B>>,
        ord1_cotangents: &HashMap<Id, CotangentFrame<B>>,
        ord2_cotangents: &HashMap<Id, CotangentFrame<B>>,
        tag: &Q<B, T>,
    ) -> SkoroResult<PointHvpEval<B>>;

    /// JVP over several quantities by id.
    fn evaluate_ids_jvp(
        &self,
        points: &ValueMap<B>,
        tangents: &HashMap<Id, TangentFrame<B>>,
        ids: &[Id],
    ) -> SkoroResult<JvpForwardOutput<B>>;

    /// VJP over several quantities by id.
    fn evaluate_ids_vjp(
        &self,
        points: &ValueMap<B>,
        cotangents: &HashMap<Id, CotangentFrame<B>>,
        ids: &[Id],
    ) -> SkoroResult<VjpForwardOutput<B>>;

    /// HVP over several quantities by id.
    fn evaluate_ids_hvp(
        &self,
        points: &ValueMap<B>,
        tangents: &HashMap<Id, TangentVector<B>>,
        ord1_cotangents: &HashMap<Id, CotangentFrame<B>>,
        ord2_cotangents: &HashMap<Id, CotangentFrame<B>>,
        ids: &[Id],
    ) -> SkoroResult<HvpForwardOutput<B>>;
}

/// Capability: this client can grow the model with stochastic quantities and
/// free-standing likelihood factors.
///
/// Deliberately separate from [`ModelClientOps`] rather than folded into it. A
/// restricting facade such as `GaussianModelClient` implements that trait *precisely
/// in order to withhold this surface* — if the two were one trait, "a client that can
/// evaluate a model" and "a client that can add anything to a model" would be the same
/// bound, and no facade could offer the first without the second.
///
/// Note what this trait is not: implementing it says nothing about what a model
/// *contains*. That is a structural claim, and it belongs on the builder that
/// accumulated the model, not on a capability trait anyone may implement.
pub trait ModelBuilderOps<B: Backend>: ModelClientOps<B> {
    /// Add a stochastic quantity with the given conditional distribution.
    fn add_quantity<K, D>(&self, distr: D) -> SkoroResult<Q<B, D::Qty>>
    where
        K: Kind<B>,
        D: Distribution<B, K>;

    /// Replace the conditional distribution of an existing stochastic quantity.
    fn update_quantity<K, D>(&self, tag: &Q<B, D::Qty>, distr: D) -> SkoroResult<()>
    where
        K: Kind<B>,
        D: Distribution<B, K>;

    /// Attach a free-standing likelihood factor to a real-valued quantity.
    fn mark_likelihood(&self, likelihood: &Q<B, Real>) -> SkoroResult<()>;

    /// Remove `roots` together with everything downstream of them.
    ///
    /// The teardown half of a sliding window. [`ModelClientOps::remove`] is leaf-only and
    /// all-or-nothing: it rejects a batch any survivor still points into, so retiring a state
    /// means computing the downward closure by hand at every call site. This is that walk.
    ///
    /// **Why this is a builder capability and `remove` is not.** `remove` refuses to break a
    /// reference, so an algorithm holding it cannot silently delete a factor it was handed —
    /// which is what makes it safe at the [`ModelClientOps`] tier that
    /// [`NlsStructure`](crate::structure::NlsStructure) sits on. This one deliberately does the
    /// opposite: it takes the referents too, and cannot fail on a live child. Deciding that a
    /// whole cone of the model is finished is the same order of authority as adding a factor,
    /// so it sits with the other calls that shape the model rather than with the ones that only
    /// read it.
    ///
    /// Nodes the graph built for itself and never handed out — the anchors of a stacked tangent
    /// — are reclaimed alongside, once nothing points at them. A recording builder must also
    /// drop the certificate entries whose nodes went; see
    /// [`NlsModelClient::remove_closure`](crate::structure::NlsModelClient::remove_closure).
    fn remove_closure(&self, roots: &[Id]) -> SkoroResult<Vec<Node<B>>>;
}

impl<B: Backend> ModelClientOps<B> for ModelClient<B> {
    #[inline(always)]
    fn add_constant<K, T, D>(&self, quantity_type: T, value: &D) -> Q<B, T>
    where
        K: Kind<B>,
        T: QuantityType<B, K> + Packable<K::Elem, D>,
        D: 'static,
    {
        ModelClient::add_constant(self, quantity_type, value)
    }

    #[inline(always)]
    fn add_batched_constant<K, T, D>(&self, quantity_type: T, values: &[D]) -> Q<B, T>
    where
        K: Kind<B>,
        T: QuantityType<B, K> + Packable<K::Elem, D>,
        D: 'static,
    {
        ModelClient::add_batched_constant(self, quantity_type, values)
    }

    #[inline(always)]
    fn add_parameter<K, T>(&self, quantity_type: T) -> Q<B, T>
    where
        K: Kind<B>,
        T: QuantityType<B, K>,
    {
        ModelClient::add_parameter(self, quantity_type)
    }

    #[inline(always)]
    fn observe<K, T, D>(&self, q: &Q<B, T>, value: &D) -> SkoroResult<()>
    where
        K: Kind<B>,
        T: QuantityType<B, K> + Packable<K::Elem, D>,
        D: 'static,
    {
        ModelClient::observe(self, q, value)
    }

    #[inline(always)]
    fn unobserve<T>(&self, q: &Q<B, T>) -> SkoroResult<()> {
        ModelClient::unobserve(self, q)
    }

    #[inline(always)]
    fn demote_to_parameter<T>(&self, q: &Q<B, T>) -> SkoroResult<()> {
        ModelClient::demote_to_parameter(self, q)
    }

    #[inline(always)]
    fn remove(&self, ids: &[Id]) -> SkoroResult<Vec<Node<B>>> {
        ModelClient::remove(self, ids)
    }

    #[inline(always)]
    fn cleanup(&self) {
        ModelClient::cleanup(self)
    }

    #[inline(always)]
    fn tangent_layout(&self, ids: &[Id]) -> SkoroResult<TangentLayout> {
        self.server_call(|server| TangentLayout::from_ids(server.graph(), ids))
    }

    fn get_quantity_type<K, T>(&self, tag: &Q<B, T>) -> SkoroResult<T>
    where
        K: Kind<B>,
        T: QuantityType<B, K>,
    {
        ModelClient::get_quantity_type(self, tag)
    }

    #[inline(always)]
    fn log_density(&self, points: &ValueMap<B>) -> SkoroResult<Point<B>> {
        ModelClient::log_density(self, points)
    }

    #[inline(always)]
    fn log_density_ord1(&self, points: &ValueMap<B>) -> SkoroResult<PointEvalOrder1<B>> {
        ModelClient::log_density_ord1(self, points)
    }

    #[inline(always)]
    fn log_density_ord2(
        &self,
        points: &ValueMap<B>,
        tangents: &HashMap<Id, TangentVector<B>>,
    ) -> SkoroResult<PointEvalOrder2<B>> {
        ModelClient::log_density_ord2(self, points, tangents)
    }

    #[inline(always)]
    fn evaluate<M: ContinuousQuantity<B>>(
        &self,
        points: &ValueMap<B>,
        tag: &Q<B, M>,
    ) -> SkoroResult<Point<B>> {
        ModelClient::evaluate(self, points, tag)
    }

    #[inline(always)]
    fn evaluate_ord1<M: ContinuousQuantity<B>>(
        &self,
        points: &ValueMap<B>,
        tag: &Q<B, M>,
    ) -> SkoroResult<PointEvalOrder1<B>> {
        ModelClient::evaluate_ord1(self, points, tag)
    }

    #[inline(always)]
    fn evaluate_ord2<M: ContinuousQuantity<B>>(
        &self,
        points: &ValueMap<B>,
        tangents: &HashMap<Id, TangentVector<B>>,
        tag: &Q<B, M>,
    ) -> SkoroResult<PointEvalOrder2<B>> {
        ModelClient::evaluate_ord2(self, points, tangents, tag)
    }

    #[inline(always)]
    fn evaluate_ids(&self, points: &ValueMap<B>, ids: &[Id]) -> SkoroResult<ForwardOutput<B>> {
        ModelClient::evaluate_ids(self, points, ids)
    }

    #[inline(always)]
    fn evaluate_ids_ord1(&self, points: &ValueMap<B>, ids: &[Id]) -> SkoroResult<EvalOrder1<B>> {
        ModelClient::evaluate_ids_ord1(self, points, ids)
    }

    #[inline(always)]
    fn evaluate_ids_ord2(
        &self,
        points: &ValueMap<B>,
        tangents: &HashMap<Id, TangentVector<B>>,
        ids: &[Id],
    ) -> SkoroResult<EvalOrder2<B>> {
        ModelClient::evaluate_ids_ord2(self, points, tangents, ids)
    }

    #[inline(always)]
    fn generate(&self, batch_size: usize) -> SkoroResult<ForwardOutput<B>> {
        ModelClient::generate(self, batch_size)
    }

    #[inline(always)]
    fn generate_from(
        &self,
        points: &ValueMap<B>,
        batch_size: usize,
    ) -> SkoroResult<ForwardOutput<B>> {
        ModelClient::generate_from(self, points, batch_size)
    }

    #[inline(always)]
    fn evaluate_jvp<T: ContinuousQuantity<B>>(
        &self,
        points: &ValueMap<B>,
        tangents: &HashMap<Id, TangentFrame<B>>,
        tag: &Q<B, T>,
    ) -> SkoroResult<PointJvpEval<B>> {
        ModelClient::evaluate_jvp(self, points, tangents, tag)
    }

    #[inline(always)]
    fn evaluate_vjp<T: ContinuousQuantity<B>>(
        &self,
        points: &ValueMap<B>,
        cotangents: &HashMap<Id, CotangentFrame<B>>,
        tag: &Q<B, T>,
    ) -> SkoroResult<PointVjpEval<B>> {
        ModelClient::evaluate_vjp(self, points, cotangents, tag)
    }

    #[inline(always)]
    fn evaluate_hvp<T: ContinuousQuantity<B>>(
        &self,
        points: &ValueMap<B>,
        tangents: &HashMap<Id, TangentVector<B>>,
        ord1_cotangents: &HashMap<Id, CotangentFrame<B>>,
        ord2_cotangents: &HashMap<Id, CotangentFrame<B>>,
        tag: &Q<B, T>,
    ) -> SkoroResult<PointHvpEval<B>> {
        ModelClient::evaluate_hvp(
            self,
            points,
            tangents,
            ord1_cotangents,
            ord2_cotangents,
            tag,
        )
    }

    #[inline(always)]
    fn evaluate_ids_jvp(
        &self,
        points: &ValueMap<B>,
        tangents: &HashMap<Id, TangentFrame<B>>,
        ids: &[Id],
    ) -> SkoroResult<JvpForwardOutput<B>> {
        ModelClient::evaluate_ids_jvp(self, points, tangents, ids)
    }

    #[inline(always)]
    fn evaluate_ids_vjp(
        &self,
        points: &ValueMap<B>,
        cotangents: &HashMap<Id, CotangentFrame<B>>,
        ids: &[Id],
    ) -> SkoroResult<VjpForwardOutput<B>> {
        ModelClient::evaluate_ids_vjp(self, points, cotangents, ids)
    }

    #[inline(always)]
    fn evaluate_ids_hvp(
        &self,
        points: &ValueMap<B>,
        tangents: &HashMap<Id, TangentVector<B>>,
        ord1_cotangents: &HashMap<Id, CotangentFrame<B>>,
        ord2_cotangents: &HashMap<Id, CotangentFrame<B>>,
        ids: &[Id],
    ) -> SkoroResult<HvpForwardOutput<B>> {
        ModelClient::evaluate_ids_hvp(
            self,
            points,
            tangents,
            ord1_cotangents,
            ord2_cotangents,
            ids,
        )
    }
}

impl<B: Backend> ModelBuilderOps<B> for ModelClient<B> {
    #[inline(always)]
    fn add_quantity<K, D>(&self, distr: D) -> SkoroResult<Q<B, D::Qty>>
    where
        K: Kind<B>,
        D: Distribution<B, K>,
    {
        ModelClient::add_quantity(self, distr)
    }

    #[inline(always)]
    fn update_quantity<K, D>(&self, tag: &Q<B, D::Qty>, distr: D) -> SkoroResult<()>
    where
        K: Kind<B>,
        D: Distribution<B, K>,
    {
        ModelClient::update_quantity(self, tag, distr)
    }

    #[inline(always)]
    fn mark_likelihood(&self, likelihood: &Q<B, Real>) -> SkoroResult<()> {
        ModelClient::mark_likelihood(self, likelihood)
    }

    #[inline(always)]
    fn remove_closure(&self, roots: &[Id]) -> SkoroResult<Vec<Node<B>>> {
        ModelClient::remove_closure(self, roots)
    }
}
