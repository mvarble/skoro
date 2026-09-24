//! Reading a least-squares structure off the model's bindings.

use skoro_backend::{Backend, Tensor};
use skoro_core::{
    error::{SkoroError, SkoroResult},
    id::Id,
};
use skoro_graph::{compiler::ForwardOutput, tangent_layout::TangentLayout};
use skoro_operation::ValueMap;

use crate::client_ops::ModelReaderOps;
use crate::jacobian::jacobian_blocks;
use crate::roles::RESIDUAL;

/// The rows of `J`: a layout over every term's [`RESIDUAL`] binding, in term order.
///
/// A term with no residual is skipped, as is a proposal-only term: it is no part of the objective.
/// A term with more than one residual is [`SkoroError::ShapeMismatch`].
///
/// As with [`information_blocks`](crate::information::information_blocks), skipping means the
/// layout can describe a sub-objective without saying so.
/// [`require_role`](crate::coverage::require_role) is the guard for a caller that needs every
/// scored term represented.
pub fn residual_layout<B: Backend, C: ModelReaderOps<B>>(client: &C) -> SkoroResult<TangentLayout> {
    client.tangent_layout(
        &residual_terms(client)?
            .iter()
            .map(|(_, r)| *r)
            .collect::<Vec<_>>(),
    )
}

/// `(density, residual)` for every scored term that binds a [`RESIDUAL`], in term order.
fn residual_terms<B: Backend, C: ModelReaderOps<B>>(client: &C) -> SkoroResult<Vec<(Id, Id)>> {
    let mut ids: Vec<(Id, Id)> = Vec::new();
    for factor in client.factors() {
        // Only the log-density's terms make up the sum of squares; a proposal-only term binds a
        // `RESIDUAL` too, but it is no part of the objective.
        if !factor.in_density {
            continue;
        }
        match client.bound(factor.density, RESIDUAL).as_slice() {
            [] => {}
            [residual] => ids.push((factor.density, *residual)),
            residuals => {
                return Err(SkoroError::ShapeMismatch {
                    expected: 1,
                    found: residuals.len(),
                });
            }
        }
    }
    Ok(ids)
}

/// One term's whitened residual and its block Jacobian, the rows `[J | r]` of a least-squares
/// step.
pub struct ResidualBlock<B: Backend> {
    /// The term's log-likelihood node.
    pub factor: Id,
    /// The residual node it binds under [`RESIDUAL`].
    pub residual: Id,
    /// The free leaves the residual depends on, in the columns' order.
    pub keys: TangentLayout,
    /// `∂r/∂keys`: `[batch, dim(r), keys.total()]`.
    pub jacobian: Tensor<B, 3>,
    /// `r` itself: `[batch, dim(r)]`.
    pub value: Tensor<B, 2>,
}

/// Every scored term's residual and its Jacobian over the free leaves it reaches, in the row order
/// of [`residual_layout`].
///
/// This is what a Gauss–Newton or Levenberg–Marquardt step needs, block by block: the rows
/// `J_t δ ≈ −r_t` of each term, with `J_t` only over the term's keys. Keeping `r` and `J` in step
/// used to be the caller's job; here they come from the same list, so they cannot drift.
///
/// The Jacobians come from one compressed reverse pass
/// ([`jacobian_blocks`]) and the residuals from one evaluation.
/// A residual whose value is not a flat vector of its own tangent dimension — anything but a
/// Euclidean node — is [`SkoroError::ShapeMismatch`], since `r` is then no tangent vector.
///
/// Terms without a residual are skipped, as in [`residual_layout`];
/// [`require_role`](crate::coverage::require_role) is the guard against that.
pub fn residual_blocks<B: Backend, C: ModelReaderOps<B>>(
    client: &C,
    points: &ValueMap<B>,
    free: &TangentLayout,
) -> SkoroResult<Vec<ResidualBlock<B>>> {
    let terms = residual_terms(client)?;
    let residual_ids: Vec<Id> = terms.iter().map(|(_, r)| *r).collect();
    let rows: std::collections::HashMap<Id, _> =
        jacobian_blocks(client, points, free, &residual_ids)?
            .rows
            .into_iter()
            .map(|row| (row.row, row))
            .collect();
    let out = client.evaluate_ids(points, &residual_ids)?;
    terms
        .into_iter()
        .map(|(factor, residual)| {
            let row = rows
                .get(&residual)
                .expect("jacobian_blocks returns every distinct row");
            let value = value_of(&out, residual)?;
            let dim = row.jacobian.dims()[1];
            if value.dims()[1] != dim {
                return Err(SkoroError::ShapeMismatch {
                    expected: dim,
                    found: value.dims()[1],
                });
            }
            Ok(ResidualBlock {
                factor,
                residual,
                keys: row.keys.clone(),
                jacobian: row.jacobian.clone(),
                value,
            })
        })
        .collect()
}

/// The stacked residual `[batch, residual_layout(client).total()]`, in the row order of
/// [`residual_layout`] — the `r` that pairs with a dense `jacobian_vjp` over that layout.
pub fn residual_vector<B: Backend, C: ModelReaderOps<B>>(
    client: &C,
    points: &ValueMap<B>,
) -> SkoroResult<Tensor<B, 2>> {
    let layout = residual_layout(client)?;
    let ids: Vec<Id> = layout.ids().collect();
    let out = client.evaluate_ids(points, &ids)?;
    let mut parts = Vec::with_capacity(ids.len());
    for block in layout.blocks() {
        let value = value_of(&out, block.id)?;
        if value.dims()[1] != block.dimension {
            return Err(SkoroError::ShapeMismatch {
                expected: block.dimension,
                found: value.dims()[1],
            });
        }
        parts.push(value);
    }
    if parts.is_empty() {
        let (batch, device) = crate::jacobian::batch_and_device(points);
        return Ok(Tensor::zeros([batch, 0], &device));
    }
    Ok(Tensor::cat(parts, 1))
}

/// A node's value from an evaluation, wherever the compile recorded it.
fn value_of<B: Backend>(out: &ForwardOutput<B>, id: Id) -> SkoroResult<Tensor<B, 2>> {
    let (value, _) = out
        .evaluations
        .get(&id)
        .or_else(|| out.constants.get(&id))
        .or_else(|| out.variables.get(&id))
        .ok_or(SkoroError::NodeNotFound { id })?;
    Ok(value.unwrap_float_ref().clone())
}
