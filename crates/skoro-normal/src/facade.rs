//! Gaussian model facades.
//!
//! These restrict model-building to Gaussian conditionals. The capability trait
//! [`GaussianModelClientOps`] adds
//! [`add_gaussian_quantity`](GaussianModelClientOps::add_gaussian_quantity)
//! on top of the distribution-agnostic [`ModelClientOps`]; it is implemented for both the general
//! [`ModelClient`] and the restricted [`GaussianModelClient`].
//!
//! - **Trait [`GaussianModelClientOps`]** is the *capability* "can add Gaussian
//!   quantities" — used as a bound by adapter/producer code that builds models. It is
//!   implemented for the unrestricted [`ModelClient`] too, so it says *can add Gaussians*
//!   and never *only ever did*. It claims nothing about what a model contains.
//! - **Type [`GaussianModelClient`]** is the *builder*: it restricts model growth to
//!   Gaussian conditionals and records what it built. It does not expose the general
//!   `add_quantity` surface, and intentionally does **not** `Deref` to [`ModelClient`].
//!
//! **The structural claim lives elsewhere.** "Every conditional in this model is Gaussian"
//! is what an inference algorithm bounds on, and it belongs on
//! [`GaussianStructure`](crate::structure::GaussianStructure), whose methods return a
//! certificate nothing outside a bounded constructor can populate. A capability trait
//! cannot carry that claim, however it is documented — see
//! [`skoro_model::structure`] for why the guarantee is on the returned evidence rather
//! than on the trait.

use std::collections::HashMap;

use skoro_backend::{
    Backend, CotangentFrame, ElementConversion, Point, TangentFrame, TangentVector,
};
use skoro_core::{
    error::{SkoroError, SkoroResult},
    id::Id,
    kind::Kind,
    pack::Packable,
};
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
use skoro_model::ModelBuilderOps;
use skoro_model::{
    ModelClientOps, NlsStructure, ResidualCertificate, TangentLayout, WhitenedResidual,
    model::{ModelClient, ModelServer},
    q::Q,
    structure::retire_recorded,
};
use skoro_operation::ValueMap;
use skoro_riemann::euclidean::{EuclideanD, VectorD};

use crate::distribution::Normal;
use crate::structure::{GaussianBlock, GaussianStructure};

/// Capability: a client that can add Gaussian-distributed quantities.
///
/// Extends [`ModelClientOps`] with [`add_gaussian_quantity`](Self::add_gaussian_quantity),
/// which is semantically `add_quantity` with a [`Normal`] distribution.
///
/// This is a *capability*, not a guarantee: it is implemented for the unrestricted
/// [`ModelClient`] as well, so a `C: GaussianModelClientOps<B>` bound says the client can
/// add Gaussians, never that it only ever did. For the latter, bound on
/// [`GaussianStructure`](crate::structure::GaussianStructure).
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
    ///
    /// Takes `&mut self` because an implementor may *record* what it built —
    /// [`GaussianModelClient`] does, and the cache it accumulates is the evidence an
    /// inference algorithm consumes. An implementor that records nothing, such as the
    /// blanket impl for [`ModelClient`], simply ignores the stronger receiver.
    fn add_gaussian_quantity<T: ContinuousQuantity<B>>(
        &mut self,
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
        &mut self,
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

/// A builder restricted to Gaussian conditionals, which records what it built.
///
/// Wraps a [`ModelClient`] but deliberately does **not** expose the general
/// `add_quantity` / `get_server` surface, so the only way a stochastic node enters the
/// model is via
/// [`add_gaussian_quantity`](GaussianModelClientOps::add_gaussian_quantity) — which
/// records a [`GaussianBlock`] and a certificate entry as it goes.
///
/// The recording is the whole point. A distribution is consumed when a quantity is added
/// and nothing identifying it survives into the graph, so the moment of building is the
/// only moment the family is knowable. What is accumulated here is handed to inference
/// algorithms through [`NlsStructure`] and [`GaussianStructure`].
///
/// **Not [`Clone`].** Two clones would share one graph while owning separate caches, so
/// each could record half the factors and both would report a certificate that omitted the
/// other's — exactly the forgery the certificate exists to prevent.
#[derive(Debug)]
pub struct GaussianModelClient<B: Backend> {
    client: ModelClient<B>,
    certificate: ResidualCertificate,
    blocks: Vec<GaussianBlock>,
    /// Keeps the recorded residual nodes alive.
    ///
    /// A residual is a `Mapped` node with no children, so dropping the last [`Q`] naming it
    /// queues it for garbage collection and the ids in `certificate` would dangle after the
    /// next `cleanup()`.
    residuals: Vec<Q<B, EuclideanD>>,
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
            certificate: ResidualCertificate::new(),
            blocks: Vec::new(),
            residuals: Vec::new(),
        }
    }

    /// Consume the facade and return the underlying general client.
    ///
    /// **This dissolves the certificate.** It is an explicit exit from the guarantee, not a
    /// view onto it: a caller holding the general [`ModelClient`] can add factors of any
    /// family, and the cache accumulated here — which is dropped along with the facade —
    /// would never see them. Anything already handed a [`ResidualCertificate`] keeps a
    /// snapshot that was honest when it was taken and says nothing about the model
    /// afterwards.
    pub fn into_client(self) -> ModelClient<B> {
        self.client
    }

    /// Remove `roots` with everything downstream, and drop the records that went with them.
    ///
    /// The teardown half of a sliding window, and the reason this facade can host one: retiring a
    /// state deletes the factors that scored it, and both caches here name those factors by id.
    /// [`retire_recorded`] prunes the certificate; the [`GaussianBlock`] list is pruned beside it,
    /// against the same liveness question, so the two cannot come to disagree about which factors
    /// the model holds.
    ///
    /// Inherent rather than a [`ModelBuilderOps`](skoro_model::ModelBuilderOps) impl — this facade
    /// withholds that trait deliberately, exactly as it withholds `add_quantity`.
    pub fn remove_closure(&mut self, roots: &[Id]) -> SkoroResult<Vec<Node<B>>> {
        let removed = retire_recorded(
            &self.client,
            &mut self.certificate,
            &mut self.residuals,
            roots,
        )?;
        let graph_has = |id: Id| {
            self.client
                .server_call(|server| server.graph().contains(id))
        };
        self.blocks
            .retain(|block| graph_has(block.density) && graph_has(block.residual));
        Ok(removed)
    }
}

impl<B: Backend> NlsStructure<B> for GaussianModelClient<B> {
    fn certificate(&self) -> &ResidualCertificate {
        &self.certificate
    }
}

impl<B: Backend> GaussianStructure<B> for GaussianModelClient<B> {
    fn blocks(&self) -> &[GaussianBlock] {
        &self.blocks
    }
}

/// Build a constant lower-triangular covariance factor `Q<B, Tril>` whose diagonal is `diag` and
/// whose off-diagonal entries are zero.
///
/// `diag` are the standard deviations along each axis (the Cholesky factor of a diagonal
/// covariance). The resulting factor has side `diag.len()`.
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

/// Build a constant lower-triangular covariance factor `Q<B, Tril>` whose diagonal isotropic with
/// value `scale`, size `n`, and off-diagonal entries are zero.
///
/// `diag` are the standard deviations along each axis (the Cholesky factor of a diagonal
/// covariance). The resulting factor has side `diag.len()`.
pub fn constant_covariance_isotropic<B, C>(client: &C, scale: f64, n: usize) -> Q<B, Tril>
where
    B: Backend,
    C: ModelClientOps<B>,
{
    let k = n * (n + 1) / 2;
    let mut packed = vec![B::FloatElem::from_elem(0.0); k];
    let s = B::FloatElem::from_elem(scale);
    for i in 0..n {
        // Diagonal entry of row `i` in the row-major packed lower-triangular layout.
        packed[i * (i + 3) / 2] = s
    }
    client.add_constant(Tril(n), &VectorD::from_vec(packed))
}

/// Build a constant identity covariance factor `Q<B, Tril>` of on space of dimension `n`.
#[inline(always)]
pub fn constant_covariance_identity<B, C>(client: &C, n: usize) -> Q<B, Tril>
where
    B: Backend,
    C: ModelClientOps<B>,
{
    constant_covariance_isotropic(client, 1.0, n)
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

    fn add_batched_constant<K, T, D>(&self, quantity_type: T, values: &[D]) -> Q<B, T>
    where
        K: Kind<B>,
        T: QuantityType<B, K> + Packable<K::Elem, D>,
        D: 'static,
    {
        self.client.add_batched_constant(quantity_type, values)
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

    fn tangent_layout(&self, ids: &[Id]) -> SkoroResult<TangentLayout> {
        self.client.tangent_layout(ids)
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
    /// Adds the factor **and records it**, which is what a plain [`ModelClient`] does not
    /// do.
    ///
    /// The residual node built here is a second copy of the one `Normal::log_density`
    /// builds internally. That is not free — it is an extra subgraph — but it is the node
    /// whose id the certificate names, and a least-squares consumer evaluates it on every
    /// iteration anyway.
    fn add_gaussian_quantity<T: ContinuousQuantity<B>>(
        &mut self,
        mean: &Q<B, T>,
        std_dev: &Q<B, Tril>,
    ) -> SkoroResult<Q<B, T>> {
        let distr = Normal::new(mean.clone(), std_dev.clone());
        let x = self.client.add_quantity(distr.clone())?;
        let residual = WhitenedResidual::whitened_residual(&distr, &x)?;
        let density = self
            .client
            .server_call(|server| server.graph().density_of(x.id()))
            .ok_or(SkoroError::NotStochastic { id: x.id() })?;

        self.certificate.record(&distr, density, &residual)?;
        self.blocks.push(GaussianBlock {
            density,
            residual: residual.id(),
            mean: mean.id(),
            whitening: std_dev.id(),
        });
        self.residuals.push(residual);
        Ok(x)
    }
}
