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
//! ## The joint sibling
//!
//! [`ResidualCertificate::record_joint`] is a **bounded** sibling of `record`, demanding a
//! [`JointWhitenedResidual`]. The warning below — that the claims here lapse if `record` grows an
//! *unbounded* sibling — is untouched by it: a second door with the same lock on it is not a hole.
//! What it admits is the one factor shape a [`Distribution`] cannot describe, a joint over several
//! quantities that already exist, which has no single `Qty` and no sampler.
//!
//! ## Retiring a factor
//!
//! Evidence can leave a certificate as well as enter it. This does not weaken anything, and
//! the asymmetry is the reason: entry is bounded because a fabricated id must never get *in*,
//! while pruning can only ever take ids *out*. A certificate that has been pruned is a subset
//! of one that was honest, which is still honest.
//!
//! It is also not optional. A sliding window deletes factor nodes every tick, and a record
//! naming a deleted node is worse than no record — [`NlsStructure::residual_layout`] would
//! resolve it and fail. So the certificate lives in the
//! [`ModelServer`](crate::model::ModelServer), and every mutation that can invalidate a
//! record — `remove`, `remove_closure`, `demote_to_parameter`, `update_stochastic` — prunes
//! it inside the same locked call, against two questions asked of the graph: is the residual
//! in the arena, and is the density still a live likelihood term. The second is the one arena
//! membership cannot answer: demotion orphans a density without removing it, and an orphaned
//! density scores nothing, so a record kept on `contains` alone would hand `J` rows for a
//! term the objective no longer holds.
//!
//! ## Naming
//!
//! An `…Ops` trait answers "can this client *add* X?" and returns whatever the action
//! produced. A `…Structure` trait answers "is *every* factor in this model X?" and returns
//! a value whose constructor is bounded. That is a mechanical test rather than a matter of
//! taste: a method returning a `bool`, or a `Vec<Id>` the implementor assembled, proves
//! nothing and belongs on an `Ops` trait no matter what it is called.

use std::collections::{HashMap, HashSet};

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
use skoro_riemann::real::Real;

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

/// A factor over quantities the model **already holds**, whose negative log-density is
/// `½‖r‖² + n`.
///
/// The sibling of [`WhitenedResidual`] for factors that are not distributions. A joint likelihood
/// over a Markov blanket — what a marginalized-out quantity leaves behind — scores several
/// quantities at once, so it has no single [`Distribution::Qty`], no sampler, and adds no
/// stochastic node. It cannot implement `WhitenedResidual`, and the reason is structural rather
/// than an oversight: `Distribution` is a conditional over *one* quantity.
///
/// This deliberately does **not** need a product manifold. The residual is built by stacking each
/// member's own tangent — see
/// [`ModelClient::stacked_tangent`](crate::model::ModelClient::stacked_tangent) — so the members
/// keep their ids, their storage, and their children, and only the *tangent* is joint.
///
/// The split between the two methods mirrors [`Distribution::log_density`] and
/// [`WhitenedResidual::whitened_residual`] exactly, and carries the same obligation: an
/// implementor must make them agree, since nothing here checks that `log_density` really is
/// `−½‖whitened_residual‖² + n`. That obligation is easier to discharge here than for `Normal`,
/// whose normalizer varies with its covariance node — a joint factor's whitening is a frozen
/// constant, so a concrete implementor can derive both from one expression.
pub trait JointWhitenedResidual<B: Backend> {
    /// The residual node `r`, built over quantities the model already holds.
    ///
    /// Like [`WhitenedResidual::whitened_residual`] this *builds a graph node* rather than
    /// computing a number, and the node is what a Jacobian's rows are taken over.
    fn whitened_residual(&self, client: &ModelClient<B>) -> SkoroResult<Q<B, EuclideanD>>;

    /// The factor's contribution to the model's log-density, `−½‖r‖² + n`.
    fn log_density(&self, client: &ModelClient<B>) -> SkoroResult<Q<B, Real>>;

    /// The quantities the normalizer `n` depends on.
    ///
    /// See [`WhitenedResidual::normalizer_parents`]: minimizing `½Σ‖rᵢ‖²` is minimizing `−log p`
    /// exactly when nothing named here is being optimized.
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
    /// Every node the records name, for O(1) [`names`](Self::names).
    ///
    /// Derived from `factors` and maintained beside every write to it, in this type only —
    /// which is the point of it living here rather than beside the collector that reads it. A
    /// second owner would be a second thing to keep in step, and `docs/skoro.md` §4.2 is
    /// already a list of what that costs.
    named: HashSet<Id>,
}

impl ResidualCertificate {
    /// An empty certificate — the only one anyone outside a recording builder can make.
    pub fn new() -> Self {
        Self::default()
    }

    /// Whether `id` is a node some record names — a certified density or residual.
    ///
    /// Two callers, and they want it for opposite reasons: the collector must not *reclaim*
    /// such a node, and [`ModelServer::remove`](crate::model::ModelServer::remove) must not
    /// *delete* one. Both are the same question, so both ask it here.
    pub fn names(&self, id: Id) -> bool {
        self.named.contains(&id)
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
        self.push(FactorRecord {
            density,
            residual: residual.id(),
            normalizer_parents: distr.normalizer_parents().to_vec(),
        });
        Ok(())
    }

    /// Record a joint factor over quantities the model already holds. The `F` bound is the
    /// guarantee, exactly as in [`record`](Self::record).
    ///
    /// A **bounded** sibling, not an unbounded one: there is still no way to put a fabricated id
    /// into a certificate, because the only argument that carries ids is a `Q` the factor itself
    /// built. What it widens is the *shape* of factor admitted, not the evidence required.
    pub fn record_joint<B, F>(
        &mut self,
        factor: &F,
        density: Id,
        residual: &Q<B, EuclideanD>,
    ) -> SkoroResult<()>
    where
        B: Backend,
        F: JointWhitenedResidual<B>,
    {
        self.push(FactorRecord {
            density,
            residual: residual.id(),
            normalizer_parents: factor.normalizer_parents().to_vec(),
        });
        Ok(())
    }

    /// The only way a record enters, so the only place `named` has to be extended.
    fn push(&mut self, record: FactorRecord) {
        self.named.insert(record.density);
        self.named.insert(record.residual);
        self.factors.push(record);
    }

    pub fn factors(&self) -> &[FactorRecord] {
        &self.factors
    }

    /// Drop every record `keep` rejects, reporting each dropped record to `dropped`.
    ///
    /// Retiring part of a model deletes factor nodes, and a record naming a deleted one is not
    /// evidence of anything: [`residual_layout`](NlsStructure::residual_layout) would resolve it
    /// through `TangentLayout::from_ids` and fail with `NodeNotFound` at the next solve. Pruning
    /// is the only way an entry ever leaves a certificate, and it can only ever *shrink* one —
    /// so the guarantee that no fabricated id can enter is untouched.
    ///
    /// **Order survives.** The rows of `J` are the factors in the order they were recorded, so
    /// the survivors must keep their relative positions; a caller reading a residual block by
    /// index would otherwise be reading a different factor's rows.
    ///
    /// The caller decides what "live" means per record, and learns which records died so it can
    /// act on the manner of death: a record whose density was removed but whose residual
    /// survives is not a half-valid factor — it no longer contributes to the log-density at
    /// all, and leaving it would put rows in `J` for a term the objective does not contain;
    /// the mirror case, a scoring density whose residual is gone, must leave the objective in
    /// the same breath. `ModelServer::prune_certificate` encodes both.
    ///
    /// A dropped record's nodes stop being [`named`](Self::names), which is what un-protects
    /// them — so `dropped` is also where a caller must re-offer them to its collector. Nothing
    /// else will: losing a protector is not an event the graph can see.
    pub fn retain(
        &mut self,
        mut keep: impl FnMut(&FactorRecord) -> bool,
        mut dropped: impl FnMut(&FactorRecord),
    ) {
        let before = self.factors.len();
        self.factors.retain(|record| {
            let keep = keep(record);
            if !keep {
                dropped(record);
            }
            keep
        });
        if self.factors.len() != before {
            self.named = self
                .factors
                .iter()
                .flat_map(|record| [record.density, record.residual])
                .collect();
        }
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
    ///
    /// Returned by value: the certificate lives in the [`ModelServer`](crate::model::ModelServer),
    /// pruned inside the same locked mutation as any graph change, so a reference cannot escape
    /// the lock. A clone is cheap — a handful of ids per factor — and this is called at
    /// solve-setup, not per iteration.
    fn certificate(&self) -> ResidualCertificate;

    /// The rows of `J`, in the order the factors were recorded.
    ///
    /// The default body is the whole of it: the certificate names the residual nodes, and
    /// [`ModelClientOps::tangent_layout`] sizes them by manifold dimension. An implementor
    /// overrides this only to change the row *order*.
    fn residual_layout(&self) -> SkoroResult<TangentLayout> {
        let certificate = self.certificate();
        let ids: Vec<Id> = certificate.residual_ids().collect();
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
/// The certificate it records into lives in the server, so it is pruned inside every
/// mutation the server performs — `remove`, `remove_closure`, `demote_to_parameter`,
/// `update_stochastic` — no matter which client tier reached them.
///
/// Note there is no `into_client()`. The Gaussian facade has one and it dissolves the
/// guarantee; this builder does not repeat that.
#[derive(Debug)]
pub struct NlsModelClient<B: Backend> {
    client: ModelClient<B>,
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
        }
    }

    /// Add a factor and record it. The bound is what a plain [`ModelClient`] does not
    /// enforce.
    ///
    /// `Clone` is required because [`ModelBuilderOps::add_quantity`](crate::client_ops::ModelBuilderOps::add_quantity)
    /// consumes the distribution, while the residual node can only be built afterwards — it
    /// is a function of the stochastic quantity that call returns.
    ///
    /// The residual node built here is a *second* copy of the one the distribution's
    /// `log_density` builds internally. That is correct but not free: it is an extra
    /// subgraph, evaluated only when the residual is itself an evaluation target, which for
    /// a least-squares consumer it always is.
    ///
    /// The residual's `Q` is deliberately *not* kept: recording pins the node against
    /// `cleanup()` on the server side, and the pin leaves when the record does.
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
        self.client
            .server_call_mut(|server| server.record_factor(&distr, density, &residual))?;
        Ok(x)
    }

    /// Add a factor over quantities the model **already holds**, and record it.
    ///
    /// This is what a marginalized-out quantity leaves behind: a joint over its Markov blanket,
    /// which is not a distribution over any new quantity and so arrives through
    /// [`JointWhitenedResidual`] rather than [`WhitenedResidual`]. It adds no stochastic node —
    /// the factor is attached with `mark_likelihood`, and the quantities it scores are untouched.
    ///
    /// The returned handle is the log-density node, for a caller that wants to inspect or remove
    /// the factor later.
    pub fn add_joint_factor<F>(&mut self, factor: F) -> SkoroResult<Q<B, Real>>
    where
        F: JointWhitenedResidual<B>,
    {
        let residual = factor.whitened_residual(&self.client)?;
        let density = factor.log_density(&self.client)?;
        self.client.mark_likelihood(&density)?;
        self.client.server_call_mut(|server| {
            server.record_joint_factor(&factor, density.id(), &residual)
        })?;
        Ok(density)
    }

    /// Remove `roots` with everything downstream; the server drops the evidence that went
    /// with them in the same locked call.
    ///
    /// Inherent rather than a [`ModelBuilderOps`](crate::client_ops::ModelBuilderOps) impl for
    /// the same reason [`add_joint_factor`](Self::add_joint_factor) is: this builder withholds
    /// that trait.
    pub fn remove_closure(&mut self, roots: &[Id]) -> SkoroResult<Vec<Node<B>>> {
        self.client.remove_closure(roots)
    }

    /// The number of nodes in the underlying graph — the bound a retention policy watches.
    pub fn graph_node_count(&self) -> usize {
        self.client
            .server_call(|server| server.graph().node_count())
    }
}

impl<B: Backend> NlsStructure<B> for NlsModelClient<B> {
    fn certificate(&self) -> ResidualCertificate {
        self.client
            .server_call(|server| server.certificate().clone())
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
