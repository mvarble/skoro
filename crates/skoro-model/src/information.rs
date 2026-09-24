//! Assembling the expected (Fisher) information from the terms' Fisher roots.
//!
//! A term that binds a [`FISHER_ROOT`] `C` declares that `CᵀC` is its expected information over
//! its [`ARGUMENT`](crate::roles::ARGUMENT)s. Over a free layout `x`, the chain rule gives the
//! term's information as `(C J)ᵀ (C J)`, with `J = ∂arguments/∂x`. Stacking the `C J` blocks of
//! every term gives a root `A` of the whole model's information `AᵀA`.
//!
//! Assembly is block-sparse: each term's block is `C J` over only the free leaves its arguments
//! reach — its [`keys`](InformationBlock::keys) — never over the whole of `free`. It compiles twice
//! per call, however many terms there are: one compressed reverse pass over the distinct arguments
//! ([`jacobian_blocks`]), from which each term's `J` is gathered, and one evaluation of every root.
//!
//! Batch and device are taken from the first entry of `points`, as in [`jacobian`](crate::jacobian).

use std::collections::{HashMap, HashSet};

use skoro_backend::{Backend, Tensor};
use skoro_core::{
    error::{SkoroError, SkoroResult},
    id::Id,
};
use skoro_graph::tangent_layout::TangentLayout;
use skoro_operation::ValueMap;
use skoro_riemann::matrix::EuclideanMatrixD;

use crate::client_ops::ModelReaderOps;
use crate::jacobian::{batch_and_device, in_layout_order, jacobian_blocks};
use crate::roles::FISHER_ROOT;

/// One term's contribution to the stacked root: `C J` over the free leaves the term reaches.
pub struct InformationBlock<B: Backend> {
    /// The term's log-likelihood node.
    pub factor: Id,
    /// The free leaves the term's arguments depend on, in `free`'s order. The term's row of the
    /// stacked root is zero in every other column.
    pub keys: TangentLayout,
    /// `C J`: `[batch, k, keys.total()]`.
    pub root: Tensor<B, 3>,
}

impl<B: Backend> InformationBlock<B> {
    /// [`root`](Self::root) scattered into the columns of `free`: `[batch, k, free.total()]`.
    ///
    /// Errors [`SkoroError::IdNotInLayout`] if a key is not in `free`.
    pub fn dense(&self, free: &TangentLayout) -> SkoroResult<Tensor<B, 3>> {
        let [batch, rows, _] = self.root.dims();
        let mut out = Tensor::zeros([batch, rows, free.total()], &self.root.device());
        for key in self.keys.blocks() {
            let at = free
                .block(key.id)
                .ok_or(SkoroError::IdNotInLayout { id: key.id })?;
            out = out.slice_assign(
                [0..batch, 0..rows, at.offset..at.offset + at.dimension],
                self.root.clone().narrow(2, key.offset, key.dimension),
            );
        }
        Ok(out)
    }
}

/// One [`InformationBlock`] per term that binds a [`FISHER_ROOT`], in term order.
///
/// Terms without a root are skipped, and so are proposal-only terms, whose density the model does
/// not score. The skip is deliberate — a model may mix terms whose expected information is known
/// with terms that offer none — but it means the blocks then cover a *sub*-model, and nothing in
/// the returned value says so. A caller that cannot accept that calls
/// [`require_role`](crate::coverage::require_role) first; one that wants to report the omissions
/// reads them from [`terms_missing`](crate::coverage::terms_missing), or diffs its own
/// [`factor`](InformationBlock::factor) ids against
/// [`factors`](ModelReaderOps::factors). Returns [`SkoroError::ShapeMismatch`] for a term with more
/// than one root or with no arguments, [`SkoroError::QuantityTypeMismatch`] for a root that is
/// not an `EuclideanMatrixD`, and [`SkoroError::LayoutDimensionMismatch`] for a root whose
/// column count is not the arguments' total dimension. The first two are checked for every term
/// before anything is evaluated.
pub fn information_blocks<B: Backend, C: ModelReaderOps<B>>(
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

    // One compressed reverse pass over the distinct arguments. A row of `J` depends only on its
    // argument, not on which term lists it, so each term's `J` is gathered from these rows.
    let argument_ids: Vec<Id> = terms
        .iter()
        .flat_map(|(_, _, arguments)| arguments.ids())
        .collect();
    let argument_rows: HashMap<Id, _> = jacobian_blocks(client, points, free, &argument_ids)?
        .rows
        .into_iter()
        .map(|row| (row.row, row))
        .collect();

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

            // The term's keys: the union of its arguments' keys, in `free`'s order.
            let key_ids = in_layout_order(
                free,
                arguments
                    .ids()
                    .flat_map(|id| argument_rows[&id].keys.ids().collect::<Vec<_>>()),
            );
            let keys = client.tangent_layout(&key_ids)?;
            let (_, device) = batch_and_device(points);
            let j = Tensor::cat(
                arguments
                    .blocks()
                    .iter()
                    .map(|argument| {
                        let row = &argument_rows[&argument.id];
                        let mut block =
                            Tensor::zeros([batch, argument.dimension, keys.total()], &device);
                        for key in row.keys.blocks() {
                            let at = keys.block(key.id).expect("a row's keys are the term's");
                            block = block.slice_assign(
                                [
                                    0..batch,
                                    0..argument.dimension,
                                    at.offset..at.offset + at.dimension,
                                ],
                                row.jacobian.clone().narrow(2, key.offset, key.dimension),
                            );
                        }
                        block
                    })
                    .collect(),
                1,
            );
            Ok(InformationBlock {
                factor,
                keys,
                root: root.matmul(j),
            })
        })
        .collect()
}

/// The blocks scattered into `free`'s columns and stacked along the row axis:
/// `[batch, Σk, free.total()]` — the dense root, for models small enough to want it.
///
/// Returns [`SkoroError::ShapeMismatch`] if no term binds a root. A model where only *some* terms
/// do is not an error here: it yields a root of those terms alone, per
/// [`information_blocks`]. Pair this with [`require_role`](crate::coverage::require_role) whenever
/// the root has to cover the whole log-density — as it must when it is solved against a gradient,
/// since the gradient sums over every term.
pub fn information_root<B: Backend, C: ModelReaderOps<B>>(
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
        blocks
            .iter()
            .map(|block| block.dense(free))
            .collect::<SkoroResult<_>>()?,
        1,
    ))
}

/// The expected information `AᵀA` over `free`, `A` the [`information_root`]:
/// `[batch, free.total(), free.total()]`.
///
/// The information of the terms that bind a root, which is the model's own only when every scored
/// term does; see [`information_blocks`] and [`require_role`](crate::coverage::require_role).
pub fn information<B: Backend, C: ModelReaderOps<B>>(
    client: &C,
    points: &ValueMap<B>,
    free: &TangentLayout,
) -> SkoroResult<Tensor<B, 3>> {
    let a = information_root(client, points, free)?;
    Ok(a.clone().swap_dims(1, 2).matmul(a))
}
