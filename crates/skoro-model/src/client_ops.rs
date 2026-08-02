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
};
use skoro_operation::ValueMap;

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
    /// Remove quantities (which must have no retained children).
    fn remove(&self, ids: &[Id]) -> SkoroResult<Vec<Node<B>>>;

    /// Garbage-collect unreferenced mapped quantities.
    fn cleanup(&self);

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

impl<B: Backend> ModelClientOps<B> for ModelClient<B> {
    fn add_constant<K, T, D>(&self, quantity_type: T, value: &D) -> Q<B, T>
    where
        K: Kind<B>,
        T: QuantityType<B, K> + Packable<K::Elem, D>,
        D: 'static,
    {
        ModelClient::add_constant(self, quantity_type, value)
    }

    fn add_parameter<K, T>(&self, quantity_type: T) -> Q<B, T>
    where
        K: Kind<B>,
        T: QuantityType<B, K>,
    {
        ModelClient::add_parameter(self, quantity_type)
    }

    fn observe<K, T, D>(&self, q: &Q<B, T>, value: &D) -> SkoroResult<()>
    where
        K: Kind<B>,
        T: QuantityType<B, K> + Packable<K::Elem, D>,
        D: 'static,
    {
        ModelClient::observe(self, q, value)
    }

    fn unobserve<T>(&self, q: &Q<B, T>) -> SkoroResult<()> {
        ModelClient::unobserve(self, q)
    }

    fn remove(&self, ids: &[Id]) -> SkoroResult<Vec<Node<B>>> {
        ModelClient::remove(self, ids)
    }

    fn cleanup(&self) {
        ModelClient::cleanup(self)
    }

    fn get_quantity_type<K, T>(&self, tag: &Q<B, T>) -> SkoroResult<T>
    where
        K: Kind<B>,
        T: QuantityType<B, K>,
    {
        ModelClient::get_quantity_type(self, tag)
    }

    fn log_density(&self, points: &ValueMap<B>) -> SkoroResult<Point<B>> {
        ModelClient::log_density(self, points)
    }

    fn log_density_ord1(&self, points: &ValueMap<B>) -> SkoroResult<PointEvalOrder1<B>> {
        ModelClient::log_density_ord1(self, points)
    }

    fn log_density_ord2(
        &self,
        points: &ValueMap<B>,
        tangents: &HashMap<Id, TangentVector<B>>,
    ) -> SkoroResult<PointEvalOrder2<B>> {
        ModelClient::log_density_ord2(self, points, tangents)
    }

    fn evaluate<M: ContinuousQuantity<B>>(
        &self,
        points: &ValueMap<B>,
        tag: &Q<B, M>,
    ) -> SkoroResult<Point<B>> {
        ModelClient::evaluate(self, points, tag)
    }

    fn evaluate_ord1<M: ContinuousQuantity<B>>(
        &self,
        points: &ValueMap<B>,
        tag: &Q<B, M>,
    ) -> SkoroResult<PointEvalOrder1<B>> {
        ModelClient::evaluate_ord1(self, points, tag)
    }

    fn evaluate_ord2<M: ContinuousQuantity<B>>(
        &self,
        points: &ValueMap<B>,
        tangents: &HashMap<Id, TangentVector<B>>,
        tag: &Q<B, M>,
    ) -> SkoroResult<PointEvalOrder2<B>> {
        ModelClient::evaluate_ord2(self, points, tangents, tag)
    }

    fn evaluate_ids(&self, points: &ValueMap<B>, ids: &[Id]) -> SkoroResult<ForwardOutput<B>> {
        ModelClient::evaluate_ids(self, points, ids)
    }

    fn evaluate_ids_ord1(&self, points: &ValueMap<B>, ids: &[Id]) -> SkoroResult<EvalOrder1<B>> {
        ModelClient::evaluate_ids_ord1(self, points, ids)
    }

    fn evaluate_ids_ord2(
        &self,
        points: &ValueMap<B>,
        tangents: &HashMap<Id, TangentVector<B>>,
        ids: &[Id],
    ) -> SkoroResult<EvalOrder2<B>> {
        ModelClient::evaluate_ids_ord2(self, points, tangents, ids)
    }

    fn generate(&self, batch_size: usize) -> SkoroResult<ForwardOutput<B>> {
        ModelClient::generate(self, batch_size)
    }

    fn generate_from(
        &self,
        points: &ValueMap<B>,
        batch_size: usize,
    ) -> SkoroResult<ForwardOutput<B>> {
        ModelClient::generate_from(self, points, batch_size)
    }

    fn evaluate_jvp<T: ContinuousQuantity<B>>(
        &self,
        points: &ValueMap<B>,
        tangents: &HashMap<Id, TangentFrame<B>>,
        tag: &Q<B, T>,
    ) -> SkoroResult<PointJvpEval<B>> {
        ModelClient::evaluate_jvp(self, points, tangents, tag)
    }

    fn evaluate_vjp<T: ContinuousQuantity<B>>(
        &self,
        points: &ValueMap<B>,
        cotangents: &HashMap<Id, CotangentFrame<B>>,
        tag: &Q<B, T>,
    ) -> SkoroResult<PointVjpEval<B>> {
        ModelClient::evaluate_vjp(self, points, cotangents, tag)
    }

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

    fn evaluate_ids_jvp(
        &self,
        points: &ValueMap<B>,
        tangents: &HashMap<Id, TangentFrame<B>>,
        ids: &[Id],
    ) -> SkoroResult<JvpForwardOutput<B>> {
        ModelClient::evaluate_ids_jvp(self, points, tangents, ids)
    }

    fn evaluate_ids_vjp(
        &self,
        points: &ValueMap<B>,
        cotangents: &HashMap<Id, CotangentFrame<B>>,
        ids: &[Id],
    ) -> SkoroResult<VjpForwardOutput<B>> {
        ModelClient::evaluate_ids_vjp(self, points, cotangents, ids)
    }

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
