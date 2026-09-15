//! Constant covariance factors for building Gaussians.
//!
//! A Gaussian enters a model through `client.add_quantity(Normal::new(mean, std_dev))`: its
//! [`Factor`](skoro_model::distribution::Factor) binds the residual, the mean and the whitening
//! separately, and [`GaussianBlock`](crate::structure::GaussianBlock) reads them back. What is
//! left here are the helpers that build the lower-triangular covariance factor `std_dev` for the
//! common constant cases.

use skoro_backend::Backend;
use skoro_linalg::tril::Tril;
use skoro_model::{ModelClientOps, q::Q};
use skoro_riemann::euclidean::VectorD;

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
    let mut packed = vec![0.0f64; k];
    for (i, &d) in diag.iter().enumerate() {
        // Diagonal entry of row `i` in the row-major packed lower-triangular layout.
        packed[i * (i + 3) / 2] = d;
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
    let mut packed = vec![0.0f64; k];
    for i in 0..n {
        // Diagonal entry of row `i` in the row-major packed lower-triangular layout.
        packed[i * (i + 3) / 2] = scale;
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
