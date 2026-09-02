//! Assembling the Fisher information (or its root) from every term's `FISHER_ROOT` binding.

use skoro_backend::{Backend, Tensor};
use skoro_core::{
    error::{SkoroError, SkoroResult},
    id::Id,
};
use skoro_graph::tangent_layout::TangentLayout;
use skoro_operation::ValueMap;
use skoro_riemann::matrix::EuclideanMatrixD;

use crate::client_ops::ModelClientOps;
use crate::jacobian::jacobian_vjp;
use crate::roles::FISHER_ROOT;

/// One term's contribution to the stacked Fisher root, `[batch, k, free.total()]`.
pub struct InformationBlock<B: Backend> {
    pub factor: Id,
    pub root: Tensor<B, 3>,
}

/// Batch size and device, chosen the same way `jacobian.rs` does.
fn batch_and_device<B: Backend>(points: &ValueMap<B>) -> (usize, B::Device) {
    match points.values().next() {
        Some(point) => (point.dims()[0], point.device()),
        None => (1, B::Device::default()),
    }
}

/// One [`InformationBlock`] per log-density term that carries a `FISHER_ROOT` binding, skipping
/// every term that does not — and every proposal-only term, whose density the model does not
/// score.
pub fn information_blocks<B: Backend, C: ModelClientOps<B>>(
    client: &C,
    points: &ValueMap<B>,
    free: &TangentLayout,
) -> SkoroResult<Vec<InformationBlock<B>>> {
    let (batch, _device) = batch_and_device(points);
    let mut blocks = Vec::new();
    for f in client.factors() {
        // A proposal-only term is not a summand of the log-density, so it contributes no
        // expected information — even though it binds a `FISHER_ROOT` like any other factor.
        if !f.in_density {
            continue;
        }
        let roots = client.bound(f.density, FISHER_ROOT);
        let root_id = match roots.len() {
            0 => continue,
            1 => roots[0],
            n => {
                return Err(SkoroError::ShapeMismatch {
                    expected: 1,
                    found: n,
                });
            }
        };
        if f.arguments.is_empty() {
            return Err(SkoroError::ShapeMismatch {
                expected: 1,
                found: 0,
            });
        }
        let args = client.tangent_layout(&f.arguments)?;
        let j = jacobian_vjp(client, points, free, &args)?;

        let out = client.evaluate_ids(points, &[root_id])?;
        let (value, qty) = match out.evaluations.get(&root_id) {
            Some(pair) => pair,
            None => out
                .constants
                .get(&root_id)
                .ok_or(SkoroError::NodeNotFound { id: root_id })?,
        };
        let matrix_type = qty
            .as_any()
            .downcast_ref::<EuclideanMatrixD>()
            .ok_or(SkoroError::QuantityTypeMismatch { id: root_id })?;
        let (k, cols) = (matrix_type.0, matrix_type.1);
        if cols != args.total() {
            return Err(SkoroError::LayoutDimensionMismatch {
                expected: args.total(),
                found: cols,
            });
        }
        let root_batch = value.unwrap_float_ref().dims()[0];
        let mut root = value
            .unwrap_float_ref()
            .clone()
            .reshape([root_batch, k, cols]);
        if root_batch == 1 && batch > 1 {
            root = root.repeat_dim(0, batch);
        }

        let block = root.matmul(j);
        blocks.push(InformationBlock {
            factor: f.density,
            root: block,
        });
    }
    Ok(blocks)
}

/// The blocks concatenated along dim 1: `[batch, Σk, free.total()]`. `ShapeMismatch { expected:
/// 1, found: 0 }` if there are no blocks.
pub fn information_root<B: Backend, C: ModelClientOps<B>>(
    client: &C,
    points: &ValueMap<B>,
    free: &TangentLayout,
) -> SkoroResult<Tensor<B, 3>> {
    let blocks = information_blocks(client, points, free)?;
    if blocks.is_empty() {
        return Err(SkoroError::ShapeMismatch {
            expected: 1,
            found: 0,
        });
    }
    Ok(Tensor::cat(blocks.into_iter().map(|b| b.root).collect(), 1))
}

/// `Aᵀ A` for the stacked root `A`: `[batch, free.total(), free.total()]`.
pub fn information<B: Backend, C: ModelClientOps<B>>(
    client: &C,
    points: &ValueMap<B>,
    free: &TangentLayout,
) -> SkoroResult<Tensor<B, 3>> {
    let a = information_root(client, points, free)?;
    Ok(a.clone().swap_dims(1, 2).matmul(a))
}
