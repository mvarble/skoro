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
/// A term with no residual is skipped. A term with more than one is
/// [`SkoroError::ShapeMismatch`].
pub fn residual_layout<B: Backend, C: ModelClientOps<B>>(client: &C) -> SkoroResult<TangentLayout> {
    let mut ids: Vec<Id> = Vec::new();
    for factor in client.factors() {
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
