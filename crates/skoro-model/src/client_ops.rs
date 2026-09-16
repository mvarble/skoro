//! The distribution-agnostic client capability trait.
//!
//! [`ModelClientOps`] factors out the model-interface surface that has nothing
//! to do with which conditional distributions a model supports: adding
//! deterministic/free quantities, observing, evaluation, and generation.
//! [`ModelBuilderOps`] adds the calls that *grow* a model.
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
//! bound on `ModelClientOps`; code that also builds should bound on
//! [`ModelBuilderOps`]. Neither says anything about what a model *contains* —
//! for that, read its terms and their bindings, through
//! [`ModelClientOps::factors`] and [`ModelClientOps::bound`].

use std::collections::HashMap;

use skoro_backend::{Backend, CotangentFrame, Point, TangentFrame, TangentVector};
use skoro_core::{error::SkoroResult, id::Id, kind::Kind, pack::Packable};
use skoro_graph::{
    binding::Role,
    compiler::{ForwardOutput, HvpForwardOutput, JvpForwardOutput, VjpForwardOutput},
    graph::{
        EvalOrder1, EvalOrder2, PointEvalOrder1, PointEvalOrder2, PointHvpEval, PointJvpEval,
        PointVjpEval, Term,
    },
    node::Node,
    point_map::PointMap,
    quantity::{ContinuousQuantity, QuantityType},
    tangent_layout::TangentLayout,
};
use skoro_operation::ValueMap;

use skoro_riemann::{euclidean::EuclideanD, real::Real};

use crate::distribution::{Distribution, Factor};
use crate::model::ModelClient;
use crate::q::Q;
use crate::roles::ARGUMENT;

/// One term of a model's log-density, with the arguments it is expressed over.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct FactorInfo {
    /// The log-likelihood node.
    pub density: Id,
    /// The quantity the term is the log-density of; `None` for a free-standing factor.
    pub scored: Option<Id>,
    /// A summand of the model's log-density `p(x|z)`.
    pub in_density: bool,
    /// A summand of the generative model `q(x|z)`.
    pub in_proposal: bool,
    /// The nodes bound under [`ARGUMENT`], in binding order.
    pub arguments: Vec<Id>,
}

/// Distribution-agnostic operations available on any model client.
///
/// This is intentionally "pragmatic and broad": it mirrors the common model-building and evaluation
/// surface of [`ModelClient`] (excluding the lower-level `_jvp`/`_vjp`/`_hvp` directional
/// primitives, which remain inherent methods for now).
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
    fn add_batched_constant<K, T, D>(&self, quantity_type: T, values: &[D]) -> SkoroResult<Q<B, T>>
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
    /// The quantity loses its sampler, its parents, and its own log-density, but keeps its `Id`
    /// and quantity type, so every quantity depending on it stays valid. This is how a quantity
    /// is re-rooted when its marginal does not factorize and the joint is supplied instead as a
    /// likelihood factor over several quantities.
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

    /// The number of nodes in the underlying graph — the bound a retention policy watches.
    ///
    /// A sliding window retires by policy rather than by bookkeeping, and "how big has this
    /// got" is the question the policy asks. It is a count, not a handle, so it sits on the
    /// client tier.
    fn graph_node_count(&self) -> usize;

    /// The log-density node scoring a stochastic quantity, if it has one.
    ///
    /// A builder that has just added a quantity needs this to name the term it created:
    /// `add_quantity` returns the quantity, and the density it built is reachable only through the
    /// graph. Read-only, so it sits on the client tier.
    fn log_density_of<T>(&self, q: &Q<B, T>) -> Option<Id>;

    /// One [`FactorInfo`] per term of the log-density, in term order.
    fn factors(&self) -> Vec<FactorInfo>;

    /// Every `(role, to)` pair `id` has bound, in binding order.
    fn bindings_of(&self, id: Id) -> Vec<(Role, Id)>;

    /// The nodes `id` has bound under `role`, in binding order.
    fn bound(&self, id: Id, role: Role) -> Vec<Id>;

    /// One [`Term`] per term of the log-density, in term order.
    fn terms(&self) -> Vec<Term>;

    /// The generative model `log q(x|z)` at `points`: the sum of every term with
    /// [`Term::in_proposal`] set.
    fn log_proposal(&self, points: &ValueMap<B>) -> SkoroResult<Point<B>>;

    /// The log-weight `log p(x|z) − log q(x|z)` at `points`, the discrepancy between the density
    /// the model scores and the mechanism it generates from.
    ///
    /// This is a difference, not a sub-sum. It coincides with "the terms outside the proposal"
    /// only when every proposal term is also a summand of the log-density; a proposal-only term
    /// enters it negatively.
    fn log_weight(&self, points: &ValueMap<B>) -> SkoroResult<Point<B>>;

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

/// Capability: this client can grow the model with stochastic quantities and free-standing
/// likelihood factors.
///
/// Deliberately separate from [`ModelClientOps`] rather than folded into it. If the two were one
/// trait, "a client that can evaluate a model" and "a client that can add anything to a model"
/// would be the same bound — so an estimator, which needs only the first, could not be given
/// the first without the second. That split is what `remove` / `remove_closure` sit on either side
/// of.
///
/// Note what this trait is not: implementing it says nothing about what a model *contains*. What a
/// model contains is read from its terms and their bindings, via [`ModelClientOps::factors`].
pub trait ModelBuilderOps<B: Backend>: ModelClientOps<B> {
    /// Add a stochastic quantity with the given conditional distribution.
    fn add_quantity<K, D>(&self, distr: D) -> SkoroResult<Q<B, D::Qty>>
    where
        K: Kind<B>,
        D: Distribution<B, K>;

    /// Add a stochastic quantity whose density belongs to the *proposal* alone.
    ///
    /// The quantity is generated exactly as [`add_quantity`](Self::add_quantity)'s is — it is a
    /// stochastic node with the distribution's sampler — but its density is a summand of
    /// `q(x|z)` and not of `p(x|z)`. Whatever the posterior says about the quantity is attached
    /// separately with [`add_factor`](Self::add_factor), and
    /// [`log_weight`](ModelClientOps::log_weight) accounts for the difference.
    ///
    /// This is what lets a Monte-Carlo algorithm work over a distribution whose density is known
    /// but which offers no sampler: propose from something it can draw from, and let the weight
    /// correct.
    fn add_proposal<K, D>(&self, distr: D) -> SkoroResult<Q<B, D::Qty>>
    where
        K: Kind<B>,
        D: Distribution<B, K>;

    /// The stacked tangent `δ = [x₁ ⊖ p₁ ; … ; x_k ⊖ p_k]` over a heterogeneous set of
    /// quantities, as one `Q<B, EuclideanD>` of width `Σ dᵢ`.
    ///
    /// This is how a factor scores several quantities at once without a product manifold;
    /// see [`Graph::stacked_tangent`](skoro_graph::graph::Graph::stacked_tangent) for what it
    /// builds and why the anchors are frozen. It sits on the *builder* tier because it grows
    /// the model — it adds `2k + 1` nodes — and for no other reason.
    fn stacked_tangent(&self, at: &PointMap<B>, ids: &[Id]) -> SkoroResult<Q<B, EuclideanD>>;

    /// Attach a free-standing likelihood factor over `arguments`, bound as its
    /// [`ARGUMENT`]s in order, together with the bindings it declares.
    /// Returns the factor's log-density.
    fn add_factor(&self, factor: Factor<B>, arguments: &[Id]) -> SkoroResult<Q<B, Real>>;

    /// Record a binding: `to` plays `role` for `from`.
    fn bind(&self, from: Id, role: Role, to: Id) -> SkoroResult<()>;

    /// Remove `roots` together with everything downstream of them.
    ///
    /// The teardown half of a sliding window. [`ModelClientOps::remove`] is leaf-only and
    /// all-or-nothing: it rejects a batch any survivor still points into, so retiring a state
    /// means computing the downward closure by hand at every call site. This is that walk.
    ///
    /// **Why this is a builder capability and `remove` is not.** `remove` refuses to break a
    /// reference, so an estimator holding only [`ModelClientOps`] cannot silently delete a
    /// factor it was handed. This one deliberately does the opposite: it takes the referents
    /// too, and cannot fail on a live child. Deciding that a whole cone of the model is
    /// finished is the same order of authority as adding a factor, so it sits with the other
    /// calls that shape the model rather than with the ones that only read it.
    ///
    /// A term bound to any node in the cone goes with it, and anything the removal
    /// orphaned upstream — a retired factor's body, the anchors of a stacked tangent — is
    /// swept before it returns.
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
    fn add_batched_constant<K, T, D>(&self, quantity_type: T, values: &[D]) -> SkoroResult<Q<B, T>>
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
    fn graph_node_count(&self) -> usize {
        self.server_call(|server| server.graph().node_count())
    }

    #[inline(always)]
    fn log_density_of<T>(&self, q: &Q<B, T>) -> Option<Id> {
        self.server_call(|server| server.graph().log_density_of(q.id()))
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

    fn factors(&self) -> Vec<FactorInfo> {
        self.server_call(|server| {
            let graph = server.graph();
            graph
                .terms()
                .into_iter()
                .map(|term| FactorInfo {
                    density: term.density,
                    scored: term.scored,
                    in_density: term.in_density,
                    in_proposal: term.in_proposal,
                    arguments: graph.bound(term.density, ARGUMENT),
                })
                .collect()
        })
    }

    #[inline(always)]
    fn bindings_of(&self, id: Id) -> Vec<(Role, Id)> {
        self.server_call(|server| server.graph().bindings_of(id).to_vec())
    }

    #[inline(always)]
    fn bound(&self, id: Id, role: Role) -> Vec<Id> {
        self.server_call(|server| server.graph().bound(id, role))
    }

    #[inline(always)]
    fn terms(&self) -> Vec<Term> {
        self.server_call(|server| server.graph().terms())
    }

    fn log_proposal(&self, points: &ValueMap<B>) -> SkoroResult<Point<B>> {
        // One lock for both the classification and the evaluation: reading `terms()` under a
        // lock of its own and then evaluating under a second would sum a set the graph may no
        // longer agree with.
        self.server_call(|server| {
            let graph = server.graph();
            graph.log_density_terms(points, &term_ids(&graph.terms(), |t| t.in_proposal))
        })
    }

    fn log_weight(&self, points: &ValueMap<B>) -> SkoroResult<Point<B>> {
        self.server_call(|server| {
            let graph = server.graph();
            let terms = graph.terms();
            // One compiled pass over the union of the two sides. Summing them separately
            // compiled a second program to walk most of the same graph, and the terms in both
            // then cancelled to within rounding rather than exactly.
            graph.log_density_terms_signed(
                points,
                &term_ids(&terms, |t| t.in_density),
                &term_ids(&terms, |t| t.in_proposal),
            )
        })
    }
}

/// The density ids of the terms `keep` selects, in term order.
fn term_ids(terms: &[Term], keep: impl Fn(&Term) -> bool) -> Vec<Id> {
    terms
        .iter()
        .filter(|term| keep(term))
        .map(|term| term.density)
        .collect()
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
    fn add_proposal<K, D>(&self, distr: D) -> SkoroResult<Q<B, D::Qty>>
    where
        K: Kind<B>,
        D: Distribution<B, K>,
    {
        ModelClient::add_proposal(self, distr)
    }

    #[inline(always)]
    fn stacked_tangent(&self, at: &PointMap<B>, ids: &[Id]) -> SkoroResult<Q<B, EuclideanD>> {
        ModelClient::stacked_tangent(self, at, ids)
    }

    #[inline(always)]
    fn add_factor(&self, factor: Factor<B>, arguments: &[Id]) -> SkoroResult<Q<B, Real>> {
        ModelClient::add_factor(self, factor, arguments)
    }

    #[inline(always)]
    fn bind(&self, from: Id, role: Role, to: Id) -> SkoroResult<()> {
        ModelClient::bind(self, from, role, to)
    }

    #[inline(always)]
    fn remove_closure(&self, roots: &[Id]) -> SkoroResult<Vec<Node<B>>> {
        ModelClient::remove_closure(self, roots)
    }
}
