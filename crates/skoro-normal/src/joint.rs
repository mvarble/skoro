//! A Gaussian factor over several existing quantities, linearized at a frozen estimate.
//!
//! This is what a marginalized-out quantity leaves behind. Retiring a state from a sliding window
//! turns its own density into a joint over its Markov blanket, and once the blanket holds more
//! than one quantity that joint has no [`Distribution`](skoro_model::distribution::Distribution)
//! to describe it — a distribution is a conditional over *one* quantity type.
//!
//! `docs/skoro.md` §4.1 reads that as needing a **product manifold**. It does not. What the factor
//! needs is a joint *tangent*, and stacking each member's own `x ⊖ p` gives one without merging
//! anything: the members keep their ids, their storage, and their other children. A product
//! manifold would have to re-home them into a single quantity, which is exactly what
//! `demote_to_parameter` exists to avoid.

use skoro_backend::Backend;
use skoro_core::{error::SkoroResult, id::Id};
use skoro_graph::point_map::PointMap;
use skoro_linalg::{
    matrix::{SparseLinearMap, sparse_linear_map},
    tril::{Tril, TrilOps},
    vector::LinalgVectorOps,
};
use skoro_model::{
    ModelBuilderOps,
    distribution::Factor,
    q::Q,
    roles::{FISHER_ROOT, NORMALIZER_PARENT, RESIDUAL},
};
use skoro_riemann::{euclidean::EuclideanD, matrix::EuclideanMatrixD, real::Real};

/// A Gaussian factor `−½ δᵀ Λ δ + n` over a set of quantities, where `δ` stacks their tangents
/// against a frozen linearization point.
///
/// The precision `Λ` enters as its **whitening** `W`, a lower-triangular `Tril(Σdᵢ)` constant with
/// `WᵀW = Λ`, so the residual is `r = W δ` and `‖r‖² = δᵀ Λ δ`. That is the same convention
/// [`Normal`](crate::distribution::Normal) uses — there `W = L⁻¹` with `Σ = L Lᵀ` — which is why
/// this reuses [`TrilOps::vector_product`] unchanged.
///
/// `W` is taken already computed rather than derived from `Λ` here, and deliberately: turning a
/// precision into a whitening is dense host linear algebra, and `skoro-dense` is the crate that
/// pays for `faer`. See `skoro_dense::factor::whitening_from_precision`.
///
/// **The block order is load-bearing.** `ids` fixes the order the tangents are stacked in, and `W`
/// must be expressed in those same coordinates. Nothing checks that correspondence, because
/// nothing can — a `Tril` of the right width is a valid whitening for any ordering.
pub struct JointNormal<B: Backend> {
    ids: Vec<Id>,
    at: PointMap<B>,
    whitening: Q<B, Tril>,
    normalizer_parents: [Id; 1],
    normalizer: f64,
}

impl<B: Backend> std::fmt::Debug for JointNormal<B> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("JointNormal")
            .field("ids", &self.ids)
            .field("whitening", &self.whitening)
            .finish_non_exhaustive()
    }
}

impl<B: Backend> Clone for JointNormal<B> {
    fn clone(&self) -> Self {
        Self {
            ids: self.ids.clone(),
            at: self.at.clone(),
            whitening: self.whitening.clone(),
            normalizer_parents: self.normalizer_parents,
            normalizer: self.normalizer,
        }
    }
}

impl<B: Backend> JointNormal<B> {
    /// A factor over `ids`, linearized at `at`, whitened by `whitening`.
    ///
    /// `at` supplies one linearization point per quantity, and is snapshotted into frozen
    /// constants when the residual is built — so the factor does not follow the quantities it
    /// summarizes. Because those anchors are batch-pinned, a batch of `N` independent problems
    /// carries `N` independent linearization points, and the model is thereafter pinned to that
    /// batch.
    ///
    /// `normalizer` is the constant term of the log-density. It does not affect any least-squares
    /// estimate — minimizing `½Σ‖rᵢ‖²` is unchanged by it — but it does affect the value of
    /// `log_density`, so a caller comparing model evidence should pass the real one:
    /// `−½ d ln(2π) + ln|det W|`, both of which
    /// `skoro_dense::factor::whitening_from_precision` returns.
    pub fn new(ids: Vec<Id>, at: PointMap<B>, whitening: Q<B, Tril>, normalizer: f64) -> Self {
        Self {
            normalizer_parents: [whitening.id()],
            ids,
            at,
            whitening,
            normalizer,
        }
    }

    /// The quantities this factor scores, in the order `W` is expressed in.
    pub fn ids(&self) -> &[Id] {
        &self.ids
    }

    /// The whitening `W`, with `WᵀW = Λ`.
    pub fn whitening(&self) -> &Q<B, Tril> {
        &self.whitening
    }
}

impl<B: Backend> JointNormal<B>
where
    B::FloatElem: std::ops::Div<B::FloatElem, Output = B::FloatElem>
        + std::ops::Mul<B::FloatElem, Output = B::FloatElem>
        + std::ops::Add<B::FloatElem, Output = B::FloatElem>
        + std::ops::Sub<B::FloatElem, Output = B::FloatElem>
        + std::ops::Neg<Output = B::FloatElem>
        + std::ops::AddAssign<B::FloatElem>,
{
    /// `r = W · [x₁ ⊖ p₁ ; … ; x_k ⊖ p_k]`.
    ///
    /// The single expression both public methods are built from, so the residual and the
    /// density cannot come to disagree about what this factor scores.
    fn residual<C: ModelBuilderOps<B>>(&self, client: &C) -> SkoroResult<Q<B, EuclideanD>> {
        let delta = client.stacked_tangent(&self.at, &self.ids)?;
        self.whitening.vector_product(&delta)
    }

    /// The residual node `r`, built over quantities the model already holds.
    ///
    /// This *builds a graph node* rather than computing a number, and the node is what a
    /// Jacobian's rows are taken over.
    pub fn whitened_residual<C: ModelBuilderOps<B>>(
        &self,
        client: &C,
    ) -> SkoroResult<Q<B, EuclideanD>> {
        self.residual(client)
    }

    /// The factor's contribution to the model's log-density, `−½‖r‖² + n`.
    ///
    /// Derived from the same [`residual`](Self::residual) as
    /// [`whitened_residual`](Self::whitened_residual), so the obligation that the two agree
    /// is discharged by construction rather than left to a caller.
    pub fn log_density<C: ModelBuilderOps<B>>(&self, client: &C) -> SkoroResult<Q<B, Real>> {
        let quadratic = (self.residual(client)?.norm_squared()? * -0.5)?;
        quadratic + self.normalizer
    }

    /// The quantities the normalizer `n` depends on: the whitening constant, and nothing
    /// else.
    ///
    /// Unlike [`Normal`](crate::distribution::Normal), whose normalizer carries a `log|L⁻¹|`
    /// term that moves with its covariance node, this factor's `W` is a frozen constant and
    /// its normalizer is a genuine constant. Reporting the id anyway is not decoration: it
    /// means the least-squares precondition still bites if anyone ever makes the whitening
    /// free.
    pub fn normalizer_parents(&self) -> &[Id] {
        &self.normalizer_parents
    }

    /// The factor this joint contributes: the log-density `−½‖r‖² + n`, built from one residual
    /// node `r = W δ`, with bindings
    ///
    /// - [`RESIDUAL`] to `r`;
    /// - [`NORMALIZER_PARENT`] to the whitening `W`;
    /// - [`FISHER_ROOT`] to `W` itself, unpacked to a dense `EuclideanMatrixD(D, D)`, so that
    ///   `CᵀC = WᵀW = Λ` over the stacked tangent of [`ids`](Self::ids).
    ///
    /// `D` is the summed manifold dimension of the members. A whitening of any other width is
    /// refused with [`SkoroError::ShapeMismatch`](skoro_core::error::SkoroError::ShapeMismatch).
    pub fn factor<C: ModelBuilderOps<B>>(&self, client: &C) -> SkoroResult<Factor<B>> {
        let residual = self.residual(client)?;
        let density = ((residual.norm_squared()? * -0.5)? + self.normalizer)?;
        let d = client.tangent_layout(&self.ids)?.total();
        let root = sparse_linear_map(&self.whitening, EuclideanMatrixD(d, d), unpack_map(d))?;
        Ok(Factor::new(density)
            .bind(RESIDUAL, &residual)
            .bind(NORMALIZER_PARENT, &self.whitening)
            .bind(FISHER_ROOT, &root))
    }
}

/// The `[d², d(d+1)/2]` map unpacking a row-major packed lower triangle into its dense `d × d`
/// row-major form: entry `i·d + j` of the output reads entry `i(i+1)/2 + j` of the input, for
/// `j ≤ i`, and the upper triangle stays zero.
fn unpack_map(d: usize) -> SparseLinearMap {
    let triplets = (0..d).flat_map(|i| (0..=i).map(move |j| (i * d + j, i * (i + 1) / 2 + j, 1.0)));
    SparseLinearMap::from_triplets(d * d, d * (d + 1) / 2, triplets)
}

/// Add a joint Gaussian factor over quantities the model **already holds**.
///
/// This is what a marginalized-out quantity leaves behind: a joint over its Markov blanket,
/// which is not a distribution over any new quantity. It adds no stochastic node — the
/// factor is attached with [`ModelBuilderOps::add_factor`] over the members as its arguments,
/// and the quantities it scores are untouched.
///
/// The returned handle is the log-density node, for a caller that wants to inspect or retire
/// the factor later. The factor binds no `LOCATION` or `WHITENING`: a joint factor has no single
/// mean node, which is the whole reason it is not a [`Normal`](crate::distribution::Normal).
pub fn add_joint_factor<B, C>(client: &C, joint: &JointNormal<B>) -> SkoroResult<Q<B, Real>>
where
    B: Backend,
    C: ModelBuilderOps<B>,
    B::FloatElem: std::ops::Div<B::FloatElem, Output = B::FloatElem>
        + std::ops::Mul<B::FloatElem, Output = B::FloatElem>
        + std::ops::Add<B::FloatElem, Output = B::FloatElem>
        + std::ops::Sub<B::FloatElem, Output = B::FloatElem>
        + std::ops::Neg<Output = B::FloatElem>
        + std::ops::AddAssign<B::FloatElem>,
{
    client.add_factor(joint.factor(client)?, joint.ids())
}
