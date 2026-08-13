//! Structural claims about a model, and the evidence that carries them.
//!
//! An inference algorithm has preconditions — Levenberg-Marquardt needs the negative
//! log-density to be a sum of squared residuals — and something has to carry those facts
//! from where they are known to where they are needed. That "somewhere" is never the built
//! graph: a [`Distribution`] is consumed by value when a quantity is added and what remains
//! records nothing identifying it, so **"was this factor a `Normal`?" has no typed answer
//! after the model is built.** Not an expensive one — none.
//!
//! So the claim is made in two places, at two scopes:
//!
//! - A **factor-local** claim — *this factor's negative log-density is `½‖r‖²` plus a
//!   normalizer* — is a bound on the [`Distribution`] type, checked where the factor is
//!   built. That is [`WhitenedResidual`].
//! - A **model-global** claim — *every factor in this model is like that* — is the
//!   accumulation of factor-local claims, carried as data by the builder that enforced
//!   them. That is [`ResidualCertificate`].
//!
//! And the property that makes this pay rather than merely typecheck: **the certificate and
//! the algorithm's input are the same object.** What a restricted builder accumulates is a
//! list of residual nodes, which is exactly the row layout of the Jacobian an optimizer
//! needs. [`NlsStructure::residual_layout`] returns the rows of `J`.
//!
//! ## Why the trait is open and the evidence is not
//!
//! The obvious move is to *seal* [`NlsStructure`], so only builders that enforced the bound
//! can implement it. It does not work: sealing confines an implementation to the defining
//! crate, and this trait must live here — it bounds [`Distribution`] and
//! [`ModelClientOps`] — while `GaussianModelClient` lives in `skoro-normal`. Sealing would
//! forbid exactly the implementation the trait exists to admit.
//!
//! So the guarantee moves into the value the trait returns. [`ResidualCertificate::record`]
//! demands a [`WhitenedResidual`] distribution and there is no constructor that accepts a
//! fabricated list of ids, so a hand-rolled implementor can produce only an empty or an
//! honest certificate. **The trait is forgeable; the evidence is not**, and the evidence is
//! what the algorithm consumes.
//!
//! ## Naming
//!
//! An `…Ops` trait answers "can this client *add* X?" and returns whatever the action
//! produced. A `…Structure` trait answers "is *every* factor in this model X?" and returns
//! a value whose constructor is bounded. That is a mechanical test rather than a matter of
//! taste: a method returning a `bool`, or a `Vec<Id>` the implementor assembled, proves
//! nothing and belongs on an `Ops` trait no matter what it is called.

use std::collections::HashMap;

use skoro_backend::{Backend, CotangentFrame, Point, TangentFrame, TangentVector};
use skoro_core::{
    error::{SkoroError, SkoroResult},
    id::Id,
    kind::{Continuous, Kind},
    pack::Packable,
};
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
use skoro_riemann::euclidean::EuclideanD;

use crate::{client_ops::ModelClientOps, distribution::Distribution, model::ModelClient, q::Q};

/// A continuous distribution whose negative log-density is `½‖r(x)‖² + n(θ)`.
///
/// Bounded on `Distribution<B, Continuous>` because a discrete factor has no tangent space,
/// `inverse_retract` is undefined for it, and no residual vector exists for it at all.
///
/// The residual type is forced rather than chosen: for a generic mean manifold `M`,
/// `inverse_retract` resolves only through the blanket
/// `AssociatedTangentVectorBuilder<EuclideanD, B>`, so [`Q<B, EuclideanD>`] needs no bound
/// on `M` beyond the one a distribution already carries.
pub trait WhitenedResidual<B: Backend>: Distribution<B, Continuous> {
    /// The residual node `r(x)`. The sum of squares of these is the objective.
    ///
    /// This *builds a graph node*; it does not compute a number. The node is what a
    /// Jacobian's rows are taken over.
    fn whitened_residual(&self, x: &Q<B, Self::Qty>) -> SkoroResult<Q<B, EuclideanD>>;

    /// The subset of [`parent_ids`](Distribution::parent_ids) on which the normalizer
    /// `n(θ)` depends.
    ///
    /// `parent_ids` is positional and untyped, so there is no way to ask "which parent is
    /// the covariance". This asks the narrower question a least-squares consumer actually
    /// has, and the answer is what the objective's validity turns on: minimizing `½Σ‖rᵢ‖²`
    /// is minimizing `−log p` exactly when no id returned here is being optimized.
    fn normalizer_parents(&self) -> &[Id];
}

/// What a builder recorded about one factor, at the moment it built it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FactorRecord {
    /// The `LogDensity` node.
    pub density: Id,
    /// The whitened-residual node — a row block of `J`.
    pub residual: Id,
    /// See [`WhitenedResidual::normalizer_parents`].
    pub normalizer_parents: Vec<Id>,
}

/// The accumulated evidence that every factor in a model is least-squares shaped.
///
/// Unforgeable not because the type is sealed, but because entries can only be added
/// through [`record`](Self::record), which demands a [`WhitenedResidual`] distribution. A
/// hand-built certificate is empty, and there is no way to put a fabricated id into one.
///
/// **If `record` ever grows an unbounded sibling, every claim in this module lapses.**
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct ResidualCertificate {
    factors: Vec<FactorRecord>,
}

impl ResidualCertificate {
    /// An empty certificate — the only one anyone outside a recording builder can make.
    pub fn new() -> Self {
        Self::default()
    }

    /// Record a factor. The `D` bound is the whole guarantee.
    ///
    /// `distr` is read only for [`WhitenedResidual::normalizer_parents`]; requiring it by
    /// reference is what makes the bound load-bearing rather than decorative.
    pub fn record<B, D>(
        &mut self,
        distr: &D,
        density: Id,
        residual: &Q<B, EuclideanD>,
    ) -> SkoroResult<()>
    where
        B: Backend,
        D: WhitenedResidual<B>,
    {
        self.factors.push(FactorRecord {
            density,
            residual: residual.id(),
            normalizer_parents: distr.normalizer_parents().to_vec(),
        });
        Ok(())
    }

    pub fn factors(&self) -> &[FactorRecord] {
        &self.factors
    }

    /// The residual nodes, in the order they were recorded — the rows of `J`.
    pub fn residual_ids(&self) -> impl Iterator<Item = Id> + '_ {
        self.factors.iter().map(|f| f.residual)
    }
}

/// A model whose negative log-density is a sum of squared residuals.
///
/// The supertrait is deliberate: [`ModelClientOps`] supplies the whole evaluation surface
/// an optimizer needs — `evaluate_ids`, `evaluate_ids_vjp` and the rest — while containing
/// no way to *grow* the model. So an algorithm bounded on `NlsStructure` can evaluate and
/// differentiate without ever holding something that could invalidate the certificate it
/// was given, and needs no second client parameter.
pub trait NlsStructure<B: Backend>: ModelClientOps<B> {
    /// The evidence. See [`ResidualCertificate`] for why it cannot be forged.
    fn certificate(&self) -> &ResidualCertificate;

    /// The rows of `J`, in the order the factors were recorded.
    ///
    /// The default body is the whole of it: the certificate names the residual nodes, and
    /// [`ModelClientOps::tangent_layout`] sizes them by manifold dimension. An implementor
    /// overrides this only to change the row *order*.
    fn residual_layout(&self) -> SkoroResult<TangentLayout> {
        let ids: Vec<Id> = self.certificate().residual_ids().collect();
        self.tangent_layout(&ids)
    }
}

/// The least-squares builder. Accepts any [`WhitenedResidual`] factor.
///
/// The Gaussian sibling in `skoro-normal` is the same shape with a stronger bound. This one
/// exists because Levenberg-Marquardt needs only that residual vectors exist — it is not
/// `Normal`-specific, and a model mixing Gaussian and non-Gaussian least-squares factors is
/// an ordinary case.
///
/// Note there is no `into_client()`. The Gaussian facade has one and it dissolves the
/// guarantee; this builder does not repeat that.
#[derive(Debug)]
pub struct NlsModelClient<B: Backend> {
    client: ModelClient<B>,
    certificate: ResidualCertificate,
    /// Keeps the recorded residual nodes alive.
    ///
    /// A residual is a `Mapped` node with no children, so dropping the last [`Q`] naming it
    /// queues it for garbage collection and the ids in `certificate` would dangle after the
    /// next `cleanup()`. The certificate itself stays non-generic and id-only; the handles
    /// live here.
    residuals: Vec<Q<B, EuclideanD>>,
}

impl<B: Backend> Default for NlsModelClient<B> {
    fn default() -> Self {
        Self::new(Default::default())
    }
}

impl<B: Backend> NlsModelClient<B> {
    pub fn new(device: B::Device) -> Self {
        Self {
            client: ModelClient::new(device),
            certificate: ResidualCertificate::new(),
            residuals: Vec::new(),
        }
    }

    /// Add a factor and record it. The bound is what a plain [`ModelClient`] does not
    /// enforce.
    ///
    /// `Clone` is required because [`ModelBuilderOps::add_quantity`] consumes the
    /// distribution, while the residual node can only be built afterwards — it is a
    /// function of the stochastic quantity that call returns.
    ///
    /// The residual node built here is a *second* copy of the one the distribution's
    /// `log_density` builds internally. That is correct but not free: it is an extra
    /// subgraph, evaluated only when the residual is itself an evaluation target, which for
    /// a least-squares consumer it always is.
    pub fn add_residual_quantity<D>(&mut self, distr: D) -> SkoroResult<Q<B, D::Qty>>
    where
        D: WhitenedResidual<B> + Clone,
    {
        let x = self.client.add_quantity(distr.clone())?;
        let residual = distr.whitened_residual(&x)?;
        let density = self
            .client
            .server_call(|server| server.graph().density_of(x.id()))
            .ok_or(SkoroError::NotStochastic { id: x.id() })?;
        self.certificate.record(&distr, density, &residual)?;
        self.residuals.push(residual);
        Ok(x)
    }
}

impl<B: Backend> NlsStructure<B> for NlsModelClient<B> {
    fn certificate(&self) -> &ResidualCertificate {
        &self.certificate
    }
}

// `NlsModelClient` exposes the distribution-agnostic surface by delegating to its inner
// `ModelClient`. It deliberately does NOT expose `ModelBuilderOps`, so the only way a
// stochastic node enters the model is via `add_residual_quantity`, which records it.
impl<B: Backend> ModelClientOps<B> for NlsModelClient<B> {
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
