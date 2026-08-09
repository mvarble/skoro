//! Gaussian model facades.
//!
//! These restrict model-building to Gaussian conditionals. The capability trait
//! [`GaussianModelClientOps`] adds
//! [`add_gaussian_quantity`](GaussianModelClientOps::add_gaussian_quantity)
//! on top of the distribution-agnostic [`ModelClientOps`]; it is implemented for both the general
//! [`ModelClient`] and the restricted [`GaussianModelClient`].
//!
//! - **Trait [`GaussianModelClientOps`]** is the *capability* "can add Gaussian
//!   quantities" — used as a bound by adapter/producer code that builds models.
//! - **Type [`GaussianModelClient`]** is the *structural guarantee* "every conditional in this
//!   model is Gaussian" — used by inference algorithms (e.g. EKF / Levenberg–Marquardt) that need
//!   the nonlinear-least-squares structure. The guarantee holds because `GaussianModelClient` does
//!   not expose the general `add_quantity`/`get_server` surface (it intentionally does **not**
//!   `Deref` to [`ModelClient`]).

use std::collections::HashMap;

use skoro_backend::{
    Backend, CotangentFrame, ElementConversion, Point, TangentFrame, TangentVector,
};
use skoro_core::{error::SkoroResult, id::Id, kind::Kind, pack::Packable};
use skoro_graph::compiler::{ForwardOutput, HvpForwardOutput, JvpForwardOutput, VjpForwardOutput};
use skoro_graph::{
    graph::{
        EvalOrder1, EvalOrder2, PointEvalOrder1, PointEvalOrder2, PointHvpEval, PointJvpEval,
        PointVjpEval,
    },
    node::Node,
    quantity::{ContinuousQuantity, QuantityType},
};
use skoro_linalg::tril::Tril;
use skoro_model::{
    ModelClientOps,
    model::{ModelClient, ModelServer},
    q::Q,
};
use skoro_operation::ValueMap;
use skoro_riemann::euclidean::VectorD;

use crate::distribution::Normal;

/// Trait bound bundle required to build a [`Normal`] for a given backend.
///
/// Mirrors the `where` clause on [`Normal::new`].
pub trait GaussianElem: Backend
where
    Self::FloatElem: std::ops::Div<Self::FloatElem, Output = Self::FloatElem>
        + std::ops::Mul<Self::FloatElem, Output = Self::FloatElem>
        + std::ops::Add<Self::FloatElem, Output = Self::FloatElem>
        + std::ops::Sub<Self::FloatElem, Output = Self::FloatElem>
        + std::ops::Neg<Output = Self::FloatElem>
        + std::ops::AddAssign<Self::FloatElem>,
{
}

impl<B: Backend> GaussianElem for B where
    B::FloatElem: std::ops::Div<Self::FloatElem, Output = Self::FloatElem>
        + std::ops::Mul<Self::FloatElem, Output = Self::FloatElem>
        + std::ops::Add<Self::FloatElem, Output = Self::FloatElem>
        + std::ops::Sub<Self::FloatElem, Output = Self::FloatElem>
        + std::ops::Neg<Output = Self::FloatElem>
        + std::ops::AddAssign<Self::FloatElem>
{
}

/// Capability: a client that can add Gaussian-distributed quantities.
///
/// Extends [`ModelClientOps`] with [`add_gaussian_quantity`](Self::add_gaussian_quantity),
/// which is semantically `add_quantity` with a [`Normal`] distribution.
pub trait GaussianModelClientOps<B: Backend>: ModelClientOps<B>
where
    B::FloatElem: std::ops::Div<B::FloatElem, Output = B::FloatElem>
        + std::ops::Mul<B::FloatElem, Output = B::FloatElem>
        + std::ops::Add<B::FloatElem, Output = B::FloatElem>
        + std::ops::Sub<B::FloatElem, Output = B::FloatElem>
        + std::ops::Neg<Output = B::FloatElem>
        + std::ops::AddAssign<B::FloatElem>,
{
    /// Add a quantity with a `Normal(mean, std_dev)` conditional distribution,
    /// where `std_dev` is a lower-triangular covariance factor.
    fn add_gaussian_quantity<T: ContinuousQuantity<B>>(
        &self,
        mean: &Q<B, T>,
        std_dev: &Q<B, Tril>,
    ) -> SkoroResult<Q<B, T>>;
}

impl<B: Backend> GaussianModelClientOps<B> for ModelClient<B>
where
    B::FloatElem: std::ops::Div<B::FloatElem, Output = B::FloatElem>
        + std::ops::Mul<B::FloatElem, Output = B::FloatElem>
        + std::ops::Add<B::FloatElem, Output = B::FloatElem>
        + std::ops::Sub<B::FloatElem, Output = B::FloatElem>
        + std::ops::Neg<Output = B::FloatElem>
        + std::ops::AddAssign<B::FloatElem>,
{
    fn add_gaussian_quantity<T: ContinuousQuantity<B>>(
        &self,
        mean: &Q<B, T>,
        std_dev: &Q<B, Tril>,
    ) -> SkoroResult<Q<B, T>> {
        self.add_quantity(Normal::new(mean.clone(), std_dev.clone()))
    }
}

/// The stateful engine for a Gaussian-only model.
///
/// Currently a thin newtype over [`ModelServer`]; it exists so the Gaussian-only
/// invariant has an end-to-end home and so future bookkeeping (e.g. caching the
/// ids of mean nodes for fast EKF Jacobian assembly) can be added without an API
/// break.
#[derive(Debug, Default)]
pub struct GaussianModelServer<B: Backend> {
    server: ModelServer<B>,
}

impl<B: Backend> GaussianModelServer<B> {
    /// Create a new Gaussian model server.
    pub fn new(device: B::Device) -> Self {
        Self {
            server: ModelServer::new(device),
        }
    }

    /// Borrow the underlying general server (read-only access for inference).
    pub fn inner(&self) -> &ModelServer<B> {
        &self.server
    }
}

/// A model client restricted to Gaussian conditionals.
///
/// Wraps a [`ModelClient`] but deliberately does **not** expose the general
/// `add_quantity` / `get_server` surface, so the only way a stochastic node
/// enters the model is via [`add_gaussian_quantity`](GaussianModelClientOps::add_gaussian_quantity).
/// This makes "all conditionals are Gaussian" a compile-time guarantee carried by
/// the type, which inference algorithms can require.
#[derive(Debug, Clone)]
pub struct GaussianModelClient<B: Backend> {
    client: ModelClient<B>,
}

impl<B: Backend> Default for GaussianModelClient<B> {
    fn default() -> Self {
        Self::new(Default::default())
    }
}

impl<B: Backend> GaussianModelClient<B> {
    /// Create a new Gaussian model and a client handle to it.
    pub fn new(device: B::Device) -> Self {
        Self {
            client: ModelClient::new(device),
        }
    }

    /// Consume the facade and return the underlying general client.
    ///
    /// This is an explicit escape hatch: once you hold a general
    /// [`ModelClient`], the Gaussian-only guarantee no longer applies.
    pub fn into_client(self) -> ModelClient<B> {
        self.client
    }
}

/// Build a constant lower-triangular covariance factor `Q<B, Tril>` whose
/// diagonal is `diag` and whose off-diagonal entries are zero.
///
/// `diag` are the standard deviations along each axis (the Cholesky factor of a
/// diagonal covariance). The resulting factor has side `diag.len()`.
pub fn constant_covariance_diagonal<B, C>(client: &C, diag: &[f64]) -> Q<B, Tril>
where
    B: Backend,
    C: ModelClientOps<B>,
{
    let n = diag.len();
    let k = n * (n + 1) / 2;
    let mut packed = vec![B::FloatElem::from_elem(0.0); k];
    for (i, &d) in diag.iter().enumerate() {
        // Diagonal entry of row `i` in the row-major packed lower-triangular layout.
        packed[i * (i + 3) / 2] = B::FloatElem::from_elem(d);
    }
    client.add_constant(Tril(n), &VectorD::from_vec(packed))
}

/// Build a constant identity covariance factor `Q<B, Tril>` of side `n`.
pub fn constant_covariance_identity<B, C>(client: &C, n: usize) -> Q<B, Tril>
where
    B: Backend,
    C: ModelClientOps<B>,
{
    constant_covariance_diagonal(client, &vec![1.0; n])
}

// `GaussianModelClient` exposes the distribution-agnostic surface by delegating
// to its inner `ModelClient`. It deliberately does NOT expose `add_quantity`,
// `update_quantity`, or `get_server`, so the only way to add a stochastic node
// is via `add_gaussian_quantity`.
impl<B: Backend> ModelClientOps<B> for GaussianModelClient<B> {
    fn add_constant<K, T, D>(&self, quantity_type: T, value: &D) -> Q<B, T>
    where
        K: Kind<B>,
        T: QuantityType<B, K> + Packable<K::Elem, D>,
        D: 'static,
    {
        self.client.add_constant(quantity_type, value)
    }

    fn add_parameter<K, T>(&self, quantity_type: T) -> Q<B, T>
    where
        K: Kind<B>,
        T: QuantityType<B, K>,
    {
        self.client.add_parameter(quantity_type)
    }

    fn observe<K, T, D>(&self, q: &Q<B, T>, value: &D) -> SkoroResult<()>
    where
        K: Kind<B>,
        T: QuantityType<B, K> + Packable<K::Elem, D>,
        D: 'static,
    {
        self.client.observe(q, value)
    }

    fn unobserve<M>(&self, q: &Q<B, M>) -> SkoroResult<()> {
        self.client.unobserve(q)
    }

    fn demote_to_parameter<M>(&self, q: &Q<B, M>) -> SkoroResult<()> {
        self.client.demote_to_parameter(q)
    }

    fn remove(&self, ids: &[Id]) -> SkoroResult<Vec<Node<B>>> {
        self.client.remove(ids)
    }

    fn cleanup(&self) {
        self.client.cleanup()
    }

    fn get_quantity_type<K, T>(&self, tag: &Q<B, T>) -> SkoroResult<T>
    where
        K: Kind<B>,
        T: QuantityType<B, K>,
    {
        self.client.get_quantity_type(tag)
    }

    fn log_density(&self, points: &ValueMap<B>) -> SkoroResult<Point<B>> {
        self.client.log_density(points)
    }

    fn log_density_ord1(&self, points: &ValueMap<B>) -> SkoroResult<PointEvalOrder1<B>> {
        self.client.log_density_ord1(points)
    }

    fn log_density_ord2(
        &self,
        points: &ValueMap<B>,
        tangents: &HashMap<Id, TangentVector<B>>,
    ) -> SkoroResult<PointEvalOrder2<B>> {
        self.client.log_density_ord2(points, tangents)
    }

    fn evaluate<T: ContinuousQuantity<B>>(
        &self,
        points: &ValueMap<B>,
        tag: &Q<B, T>,
    ) -> SkoroResult<Point<B>> {
        self.client.evaluate(points, tag)
    }

    fn evaluate_ord1<T: ContinuousQuantity<B>>(
        &self,
        points: &ValueMap<B>,
        tag: &Q<B, T>,
    ) -> SkoroResult<PointEvalOrder1<B>> {
        self.client.evaluate_ord1(points, tag)
    }

    fn evaluate_ord2<T: ContinuousQuantity<B>>(
        &self,
        points: &ValueMap<B>,
        tangents: &HashMap<Id, TangentVector<B>>,
        tag: &Q<B, T>,
    ) -> SkoroResult<PointEvalOrder2<B>> {
        self.client.evaluate_ord2(points, tangents, tag)
    }

    fn evaluate_ids(&self, points: &ValueMap<B>, ids: &[Id]) -> SkoroResult<ForwardOutput<B>> {
        self.client.evaluate_ids(points, ids)
    }

    fn evaluate_ids_ord1(&self, points: &ValueMap<B>, ids: &[Id]) -> SkoroResult<EvalOrder1<B>> {
        self.client.evaluate_ids_ord1(points, ids)
    }

    fn evaluate_ids_ord2(
        &self,
        points: &ValueMap<B>,
        tangents: &HashMap<Id, TangentVector<B>>,
        ids: &[Id],
    ) -> SkoroResult<EvalOrder2<B>> {
        self.client.evaluate_ids_ord2(points, tangents, ids)
    }

    fn generate(&self, batch_size: usize) -> SkoroResult<ForwardOutput<B>> {
        self.client.generate(batch_size)
    }

    fn generate_from(
        &self,
        points: &ValueMap<B>,
        batch_size: usize,
    ) -> SkoroResult<ForwardOutput<B>> {
        self.client.generate_from(points, batch_size)
    }

    fn evaluate_jvp<T: ContinuousQuantity<B>>(
        &self,
        points: &ValueMap<B>,
        tangents: &HashMap<Id, TangentFrame<B>>,
        tag: &Q<B, T>,
    ) -> SkoroResult<PointJvpEval<B>> {
        self.client.evaluate_jvp(points, tangents, tag)
    }

    fn evaluate_vjp<T: ContinuousQuantity<B>>(
        &self,
        points: &ValueMap<B>,
        cotangents: &HashMap<Id, CotangentFrame<B>>,
        tag: &Q<B, T>,
    ) -> SkoroResult<PointVjpEval<B>> {
        self.client.evaluate_vjp(points, cotangents, tag)
    }

    fn evaluate_hvp<T: ContinuousQuantity<B>>(
        &self,
        points: &ValueMap<B>,
        tangents: &HashMap<Id, TangentVector<B>>,
        ord1_cotangents: &HashMap<Id, CotangentFrame<B>>,
        ord2_cotangents: &HashMap<Id, CotangentFrame<B>>,
        tag: &Q<B, T>,
    ) -> SkoroResult<PointHvpEval<B>> {
        self.client
            .evaluate_hvp(points, tangents, ord1_cotangents, ord2_cotangents, tag)
    }

    fn evaluate_ids_jvp(
        &self,
        points: &ValueMap<B>,
        tangents: &HashMap<Id, TangentFrame<B>>,
        ids: &[Id],
    ) -> SkoroResult<JvpForwardOutput<B>> {
        self.client.evaluate_ids_jvp(points, tangents, ids)
    }

    fn evaluate_ids_vjp(
        &self,
        points: &ValueMap<B>,
        cotangents: &HashMap<Id, CotangentFrame<B>>,
        ids: &[Id],
    ) -> SkoroResult<VjpForwardOutput<B>> {
        self.client.evaluate_ids_vjp(points, cotangents, ids)
    }

    fn evaluate_ids_hvp(
        &self,
        points: &ValueMap<B>,
        tangents: &HashMap<Id, TangentVector<B>>,
        ord1_cotangents: &HashMap<Id, CotangentFrame<B>>,
        ord2_cotangents: &HashMap<Id, CotangentFrame<B>>,
        ids: &[Id],
    ) -> SkoroResult<HvpForwardOutput<B>> {
        self.client
            .evaluate_ids_hvp(points, tangents, ord1_cotangents, ord2_cotangents, ids)
    }
}

impl<B: Backend> GaussianModelClientOps<B> for GaussianModelClient<B>
where
    B::FloatElem: std::ops::Div<B::FloatElem, Output = B::FloatElem>
        + std::ops::Mul<B::FloatElem, Output = B::FloatElem>
        + std::ops::Add<B::FloatElem, Output = B::FloatElem>
        + std::ops::Sub<B::FloatElem, Output = B::FloatElem>
        + std::ops::Neg<Output = B::FloatElem>
        + std::ops::AddAssign<B::FloatElem>,
{
    fn add_gaussian_quantity<T: ContinuousQuantity<B>>(
        &self,
        mean: &Q<B, T>,
        std_dev: &Q<B, Tril>,
    ) -> SkoroResult<Q<B, T>> {
        self.client
            .add_quantity(Normal::new(mean.clone(), std_dev.clone()))
    }
}
