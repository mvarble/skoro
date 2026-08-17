//! The Gaussian refinement of the least-squares structure claims.
//!
//! [`WhitenedResidual`] says a factor's negative log-density is `½‖r‖²` plus a normalizer.
//! [`GaussianFactor`] adds one thing and one thing only: that `r(x) = L⁻¹ (x ⊖ μ)` with the
//! mean and the whitening reachable **separately**. That is the entire difference between
//! "least-squares shaped" and "hierarchical Gaussian", and it is what a filter needs, since
//! `H = ∂μ/∂x` and `R = L Lᵀ` are different objects.
//!
//! The *nonlinearity* of the mean needs no expression here. [`GaussianFactor::mean`] returns
//! a graph node, which may be the output of arbitrarily many operations — the graph already
//! holds it, so the refinement is two accessors rather than a new algebra.

use skoro_backend::Backend;
use skoro_core::{error::SkoroResult, id::Id};
use skoro_graph::quantity::ContinuousQuantity;
use skoro_linalg::tril::Tril;
use skoro_model::{NlsStructure, WhitenedResidual, q::Q};
use skoro_riemann::euclidean::EuclideanD;

use crate::distribution::Normal;

/// Additionally: `r(x) = L⁻¹ (x ⊖ μ)`, with the mean and whitening separately reachable.
pub trait GaussianFactor<B: Backend>: WhitenedResidual<B> {
    /// The mean node `μ`, which may itself be a mapped function of other quantities.
    fn mean(&self) -> &Q<B, Self::Qty>;

    /// The whitening factor `L⁻¹`'s source: the lower-triangular `L` with `R = L Lᵀ`.
    fn whitening(&self) -> &Q<B, Tril>;
}

/// One factor of a hierarchical Gaussian model, as its builder recorded it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GaussianBlock {
    /// The `LogDensity` node.
    pub density: Id,
    /// The whitened-residual node — a row block of `J`.
    pub residual: Id,
    /// `μ`. `H = ∂μ/∂x` is a Jacobian row block over this.
    pub mean: Id,
    /// `L`, with `R = L Lᵀ`.
    pub whitening: Id,
}

/// A model in which every conditional is Gaussian.
///
/// The refinement is the point: `GaussianStructure: NlsStructure` **is** the statement
/// "hierarchical Gaussian ⇒ least-squares capable", so an algorithm bounded on
/// [`NlsStructure`] accepts a Gaussian model without a cast, a check, or a walk.
///
/// **Phase 1 has no consumer for [`blocks`](Self::blocks).** Its consumer is the EKF, which
/// needs `μ` and `L` apart because it forms `H P Hᵀ + R`. It is declared now because the
/// refinement is the thing being validated, and declaring it later would mean revisiting
/// the facade a second time.
pub trait GaussianStructure<B: Backend>: NlsStructure<B> {
    /// The certified factors with their Gaussian refinements, in certificate order.
    ///
    /// Returned by value: membership in this list is decided by the certificate, which
    /// lives behind the server lock, and the refinement is derived per call rather than
    /// kept as a parallel list that could disagree with it.
    fn blocks(&self) -> Vec<GaussianBlock>;
}

impl<B: Backend, M: ContinuousQuantity<B>> WhitenedResidual<B> for Normal<B, M>
where
    B::FloatElem: std::ops::Div<B::FloatElem, Output = B::FloatElem>
        + std::ops::Mul<B::FloatElem, Output = B::FloatElem>
        + std::ops::Add<B::FloatElem, Output = B::FloatElem>
        + std::ops::Sub<B::FloatElem, Output = B::FloatElem>
        + std::ops::Neg<Output = B::FloatElem>
        + std::ops::AddAssign<B::FloatElem>,
{
    fn whitened_residual(&self, x: &Q<B, M>) -> SkoroResult<Q<B, EuclideanD>> {
        Normal::whitened_residual(self, x)
    }

    /// `Normal`'s normalizer is `constant + log|L⁻¹|`, so it depends on the covariance
    /// factor and on nothing else. `parent_ids()` is `[mean, std_dev]` positionally, which
    /// makes this `&self.parent_ids()[1..]`.
    fn normalizer_parents(&self) -> &[Id] {
        &<Self as skoro_model::distribution::Distribution<B, skoro_core::kind::Continuous>>::parent_ids(self)[1..]
    }
}

impl<B: Backend, M: ContinuousQuantity<B>> GaussianFactor<B> for Normal<B, M>
where
    B::FloatElem: std::ops::Div<B::FloatElem, Output = B::FloatElem>
        + std::ops::Mul<B::FloatElem, Output = B::FloatElem>
        + std::ops::Add<B::FloatElem, Output = B::FloatElem>
        + std::ops::Sub<B::FloatElem, Output = B::FloatElem>
        + std::ops::Neg<Output = B::FloatElem>
        + std::ops::AddAssign<B::FloatElem>,
{
    fn mean(&self) -> &Q<B, M> {
        Normal::mean(self)
    }

    fn whitening(&self) -> &Q<B, Tril> {
        Normal::std_dev(self)
    }
}
