//! Reading the rows of a least-squares `J` off the model's bindings.

use skoro_backend::Backend;
use skoro_core::{
    error::{SkoroError, SkoroResult},
    id::Id,
};
use skoro_graph::tangent_layout::TangentLayout;

use crate::client_ops::ModelClientOps;
use crate::roles::RESIDUAL;

/// A layout over every log-density term's `RESIDUAL` binding, in term order. A term with no
/// residual is skipped, as is a proposal-only term — it is no part of the objective; a term with
/// more than one residual is [`SkoroError::ShapeMismatch`].
pub fn residual_layout<B: Backend, C: ModelClientOps<B>>(client: &C) -> SkoroResult<TangentLayout> {
    let mut ids: Vec<Id> = Vec::new();
    for factor in client.factors() {
        // Only the log-density's terms make up the sum of squares; a proposal-only term binds a
        // `RESIDUAL` too, but it is no part of the objective.
        if !factor.in_density {
            continue;
        }
        let residuals = client.bound(factor.density, RESIDUAL);
        match residuals.len() {
            0 => continue,
            1 => ids.push(residuals[0]),
            n => {
                return Err(SkoroError::ShapeMismatch {
                    expected: 1,
                    found: n,
                });
            }
        }
    }
    client.tangent_layout(&ids)
}
