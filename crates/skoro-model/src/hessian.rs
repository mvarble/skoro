//! Dense Hessian assembly of the log-density over a flattened tangent space.
//!
//! The Hessian–vector pass (`log_density_ord2`) pushes one tangent vector per pass, so a dense
//! `D × D` matrix is `D` passes, one per coordinate of the layout. This assembles them, with the
//! same layout flattening as [`jacobian`](crate::jacobian).
//!
//! # What it is on a curved manifold
//!
//! Column `i` is `d/dt g(x ⊕ t eᵢ)` at `t = 0`, where `g(x) = ∇(log p ∘ R_x)(0)` is the gradient in
//! the chart at `x`. That is the ordinary Hessian on flat manifolds. On a curved one the chart moves
//! along the path, so the matrix need not be symmetric, and it equals the Hessian of the pullback
//! `log p ∘ R_x` only at a stationary point, where the terms from the moving chart multiply `g = 0`.
//! Nothing here symmetrizes it: a caller who wants a symmetric matrix away from a stationary point
//! has to decide which one.
//!
//! This is the *observed* curvature. The *expected* one, the Fisher information, is assembled from
//! the terms' Fisher roots by [`information`](crate::information), is positive semi-definite by
//! construction, and costs no second-order pass.
//!
//! Batch and device are taken from the first entry of `points`, as in
//! [`jacobian`](crate::jacobian).

use std::collections::HashMap;

use skoro_backend::{Backend, Tensor};
use skoro_core::error::SkoroResult;
use skoro_graph::tangent_layout::TangentLayout;
use skoro_operation::ValueMap;

use crate::client_ops::ModelReaderOps;
use crate::jacobian::batch_and_device;

/// The Hessian of `log p` over `free`: `[batch, free.total(), free.total()]`, column `i` the
/// Hessian–vector product with the `i`th unit vector of the layout.
///
/// Costs `free.total()` Hessian–vector passes over one compiled program. Columns follow the rule of
/// [`jacobian`](crate::jacobian): a `Mapped` or `LogLikelihood` column is
/// [`SkoroError::MappedColumn`](skoro_core::error::SkoroError::MappedColumn), and a constant or
/// observed one is zero. A free quantity the log-density does not reach gives a zero row and
/// column. See the [module documentation](self) for what the matrix is on a curved manifold.
pub fn hessian_hvp<B: Backend, C: ModelReaderOps<B>>(
    client: &C,
    points: &ValueMap<B>,
    free: &TangentLayout,
) -> SkoroResult<Tensor<B, 3>> {
    client.ensure_leaf_columns(free)?;
    let (batch, device) = batch_and_device(points);
    let identity = Tensor::<B, 2>::eye(free.total(), &device);
    let mut columns = Vec::with_capacity(free.total());
    for block in free.blocks() {
        for k in 0..block.dimension {
            let unit = identity
                .clone()
                .slice_dim(0, block.offset + k..block.offset + k + 1)
                .slice_dim(1, block.offset..block.offset + block.dimension)
                .repeat_dim(0, batch);
            let tangents = HashMap::from([(block.id, unit)]);
            let out = client.log_density_ord2(points, &tangents)?;
            columns.push(free.flatten_vectors(&out.ord2_cotangents, batch, &device)?);
        }
    }
    if columns.is_empty() {
        return Ok(Tensor::zeros([batch, 0, 0], &device));
    }
    Ok(Tensor::stack(columns, 2))
}
