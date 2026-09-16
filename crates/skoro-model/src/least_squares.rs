//! Reading a least-squares structure off the model's bindings.

use skoro_backend::Backend;
use skoro_core::{
    error::{SkoroError, SkoroResult},
    id::Id,
};
use skoro_graph::tangent_layout::TangentLayout;

use crate::client_ops::ModelClientOps;
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
pub fn residual_layout<B: Backend, C: ModelClientOps<B>>(client: &C) -> SkoroResult<TangentLayout> {
    let mut ids: Vec<Id> = Vec::new();
    for factor in client.factors() {
        // Only the log-density's terms make up the sum of squares; a proposal-only term binds a
        // `RESIDUAL` too, but it is no part of the objective.
        if !factor.in_density {
            continue;
        }
        match client.bound(factor.density, RESIDUAL).as_slice() {
            [] => {}
            [residual] => ids.push(*residual),
            residuals => {
                return Err(SkoroError::ShapeMismatch {
                    expected: 1,
                    found: residuals.len(),
                });
            }
        }
    }
    client.tangent_layout(&ids)
}
