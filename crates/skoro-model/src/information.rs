//! Assembling the expected (Fisher) information from the terms' Fisher roots.
//!
//! A term that binds a [`FISHER_ROOT`] `C` declares that `CᵀC` is its expected information over
//! its [`ARGUMENT`](crate::roles::ARGUMENT)s. Over a free layout `x`, the chain rule gives the
//! term's information as `(C J)ᵀ (C J)`, with `J = ∂arguments/∂x`. Stacking the `C J` blocks of
//! every term gives a root `A` of the whole model's information `AᵀA`.
//!
//! Assembly compiles twice per call, however many terms there are: one reverse pass over the union
//! of every term's arguments, from which each term's `J` is sliced, and one evaluation of every
//! root.
//!
//! Batch and device are taken from the first entry of `points`, as in [`jacobian`](crate::jacobian).

use std::collections::HashSet;

use skoro_backend::{Backend, Tensor};
use skoro_core::{
    error::{SkoroError, SkoroResult},
    id::Id,
};
use skoro_graph::tangent_layout::TangentLayout;
use skoro_operation::ValueMap;
use skoro_riemann::matrix::EuclideanMatrixD;

use crate::client_ops::ModelClientOps;
use crate::jacobian::{batch_and_device, jacobian_vjp};
use crate::roles::FISHER_ROOT;

/// One term's contribution to the stacked root: `C J`, shape `[batch, k, free.total()]`.
pub struct InformationBlock<B: Backend> {
    /// The term's log-likelihood node.
    pub factor: Id,
    pub root: Tensor<B, 3>,
}

/// One [`InformationBlock`] per term that binds a [`FISHER_ROOT`], in term order.
///
/// Terms without a root are skipped, and so are proposal-only terms, whose density the model does
/// not score. Returns [`SkoroError::ShapeMismatch`] for a term with more
/// than one root or with no arguments, [`SkoroError::QuantityTypeMismatch`] for a root that is
/// not an `EuclideanMatrixD`, and [`SkoroError::LayoutDimensionMismatch`] for a root whose
/// column count is not the arguments' total dimension. The first two are checked for every term
/// before anything is evaluated.
pub fn information_blocks<B: Backend, C: ModelClientOps<B>>(
    client: &C,
    points: &ValueMap<B>,
    free: &TangentLayout,
) -> SkoroResult<Vec<InformationBlock<B>>> {
    // Every term with a root, as (density, root, arguments), checked before anything is compiled.
    let mut terms = Vec::new();
    for factor in client.factors() {
        // A proposal-only term is not a summand of the log-density, so it contributes no
        // expected information — even though it binds a `FISHER_ROOT` like any other factor.
        // Skipping it here also keeps its arguments out of the union layout below.
        if !factor.in_density {
            continue;
        }
        let root_id = match client.bound(factor.density, FISHER_ROOT).as_slice() {
            [] => continue,
            [root] => *root,
            roots => {
                return Err(SkoroError::ShapeMismatch {
                    expected: 1,
                    found: roots.len(),
                });
            }
        };
        if factor.arguments.is_empty() {
            return Err(SkoroError::ShapeMismatch {
                expected: 1,
                found: 0,
            });
        }
        let arguments = client.tangent_layout(&factor.arguments)?;
        terms.push((factor.density, root_id, arguments));
    }
    if terms.is_empty() {
        return Ok(Vec::new());
    }

    // One reverse pass over the union of the terms' arguments. A row of `J` depends only on its
    // argument, not on which term lists it, so each term's `J` is its own blocks of this one.
    let mut seen = HashSet::new();
    let union_ids: Vec<Id> = terms
        .iter()
        .flat_map(|(_, _, arguments)| arguments.ids())
        .filter(|id| seen.insert(*id))
        .collect();
    let union = client.tangent_layout(&union_ids)?;
    let j_union = jacobian_vjp(client, points, free, &union)?;

    // One evaluation of every root. A mapped root is evaluated, a constant (or constant-folded)
    // one is read from the constants, and a parameter or stochastic one from the points it was
    // given.
    let mut seen = HashSet::new();
    let root_ids: Vec<Id> = terms
        .iter()
        .map(|(_, root, _)| *root)
        .filter(|id| seen.insert(*id))
        .collect();
    let out = client.evaluate_ids(points, &root_ids)?;

    let (batch, _) = batch_and_device(points);
    terms
        .into_iter()
        .map(|(factor, root_id, arguments)| {
            let (value, quantity_type) = out
                .evaluations
                .get(&root_id)
                .or_else(|| out.constants.get(&root_id))
                .or_else(|| out.variables.get(&root_id))
                .ok_or(SkoroError::NodeNotFound { id: root_id })?;
            let &EuclideanMatrixD(rows, cols) = quantity_type
                .as_any()
                .downcast_ref::<EuclideanMatrixD>()
                .ok_or(SkoroError::QuantityTypeMismatch { id: root_id })?;
            if cols != arguments.total() {
                return Err(SkoroError::LayoutDimensionMismatch {
                    expected: arguments.total(),
                    found: cols,
                });
            }
            let value = value.unwrap_float_ref().clone();
            let value_batch = value.dims()[0];
            let mut root = value.reshape([value_batch, rows, cols]);
            if value_batch == 1 && batch > 1 {
                root = root.repeat_dim(0, batch);
            }

            let j = Tensor::cat(
                arguments
                    .blocks()
                    .iter()
                    .map(|block| {
                        let at = union
                            .block(block.id)
                            .expect("the union holds every term's arguments");
                        j_union.clone().narrow(1, at.offset, at.dimension)
                    })
                    .collect(),
                1,
            );
            Ok(InformationBlock {
                factor,
                root: root.matmul(j),
            })
        })
        .collect()
}

/// The blocks stacked along the row axis: `[batch, Σk, free.total()]`.
///
/// Returns [`SkoroError::ShapeMismatch`] if no term binds a root.
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
    Ok(Tensor::cat(
        blocks.into_iter().map(|block| block.root).collect(),
        1,
    ))
}

/// The expected information `AᵀA` over `free`, `A` the [`information_root`]:
/// `[batch, free.total(), free.total()]`.
pub fn information<B: Backend, C: ModelClientOps<B>>(
    client: &C,
    points: &ValueMap<B>,
    free: &TangentLayout,
) -> SkoroResult<Tensor<B, 3>> {
    let a = information_root(client, points, free)?;
    Ok(a.clone().swap_dims(1, 2).matmul(a))
}
