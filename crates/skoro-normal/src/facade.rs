//! Building Gaussian factors, and recording what was built.
//!
//! A [`Normal`] conditional is added through [`add_gaussian_quantity`], which does three
//! things in one call: it adds the stochastic quantity, it builds the whitened residual
//! `r = L⁻¹ (x ⊖ μ)`, and it records the resulting factor with its `μ` and `L` named
//! separately. The recording is the point. A distribution is consumed when a quantity is
//! added and nothing identifying it survives into the graph, so the moment of building is
//! the only moment the family is knowable.
//!
//! **This is bookkeeping, not proof.** Nothing here restricts what else a caller may add to
//! the same model — the record says "this factor is a Gaussian whitened residual", and says
//! nothing at all about the factors it does not cover. An estimator reading
//! [`ModelBuilderOps::residual_layout`] gets the rows for the factors that *were* recorded,
//! and it is the caller's business to know whether that is the whole model.

use skoro_backend::{Backend, ElementConversion};
use skoro_core::error::SkoroResult;
use skoro_graph::quantity::ContinuousQuantity;
use skoro_linalg::tril::Tril;
use skoro_model::{FactorRecord, ModelBuilderOps, ModelClientOps, q::Q};
use skoro_riemann::euclidean::VectorD;

use crate::distribution::Normal;
use crate::structure::GaussianBlock;

/// Add a quantity with a `Normal(mean, std_dev)` conditional, and record the factor.
///
/// `std_dev` is the lower-triangular covariance factor `L`, with `Σ = L Lᵀ`.
///
/// The residual node built here is a *second* copy of the one `Normal::log_density` builds
/// internally. That is not free — it is an extra subgraph — but it is the node the record
/// names, and a least-squares consumer evaluates it on every iteration anyway.
///
/// Recording pins the density and the residual against the collector and against
/// [`ModelClientOps::remove`]; retiring the factor is a closure removal at `x`.
pub fn add_gaussian_quantity<B, C, T>(
    client: &C,
    mean: &Q<B, T>,
    std_dev: &Q<B, Tril>,
) -> SkoroResult<Q<B, T>>
where
    B: Backend,
    C: ModelBuilderOps<B>,
    T: ContinuousQuantity<B>,
    B::FloatElem: std::ops::Div<B::FloatElem, Output = B::FloatElem>
        + std::ops::Mul<B::FloatElem, Output = B::FloatElem>
        + std::ops::Add<B::FloatElem, Output = B::FloatElem>
        + std::ops::Sub<B::FloatElem, Output = B::FloatElem>
        + std::ops::Neg<Output = B::FloatElem>
        + std::ops::AddAssign<B::FloatElem>,
{
    let distr = Normal::new(mean.clone(), std_dev.clone());
    let x = client.add_quantity(distr.clone())?;
    let residual = distr.whitened_residual(&x)?;
    let density = client
        .log_density_of(&x)
        .ok_or(skoro_model::SkoroError::NotStochastic { id: x.id() })?;

    // `Normal`'s normalizer is `constant + log|L⁻¹|`: it depends on the covariance factor
    // and on nothing else, which is what makes the least-squares precondition checkable
    // without inspecting an operation.
    client.record_factor(
        FactorRecord::new(density, residual.id(), vec![std_dev.id()])
            .with_refinement(mean.id(), std_dev.id()),
    )?;
    Ok(x)
}

/// The model's recorded factors that carry a `μ`/`L` refinement, in record order.
///
/// Derived from the registry on every call rather than kept as a parallel list that could
/// disagree with it: membership is the registry's decision, and a retired factor's block
/// simply stops appearing.
pub fn gaussian_blocks<B, C>(client: &C) -> Vec<GaussianBlock>
where
    B: Backend,
    C: ModelBuilderOps<B>,
{
    GaussianBlock::all(&ModelBuilderOps::factors(client))
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

/// Build a constant lower-triangular covariance factor `Q<B, Tril>` whose diagonal is isotropic
/// with value `scale` and side `n`, and whose off-diagonal entries are zero.
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

/// Build a constant identity covariance factor `Q<B, Tril>` on a space of dimension `n`.
#[inline(always)]
pub fn constant_covariance_identity<B, C>(client: &C, n: usize) -> Q<B, Tril>
where
    B: Backend,
    C: ModelClientOps<B>,
{
    constant_covariance_isotropic(client, 1.0, n)
}
