//! Which terms carry a role, and which do not.
//!
//! The readers that assemble an objective from the terms' bindings are permissive: a term that
//! declared nothing under the role they want is skipped.
//! [`information_blocks`](crate::information::information_blocks) skips a term with no
//! [`FISHER_ROOT`](crate::roles::FISHER_ROOT), and
//! [`residual_layout`](crate::least_squares::residual_layout) a term with no
//! [`RESIDUAL`](crate::roles::RESIDUAL). That is deliberate — a model may legitimately mix terms
//! whose local structure a given reader understands with terms whose it does not — but it means
//! what those functions return covers a *sub*-model, and nothing in the returned value says so.
//!
//! This module is how a consumer finds out: [`terms_missing`] for the list, [`require_role`] for
//! the precondition. Neither changes what the readers do.

use skoro_backend::Backend;
use skoro_core::{
    error::{SkoroError, SkoroResult},
    id::Id,
    role::Role,
};

use crate::client_ops::ModelClientOps;

/// The scored terms that bind nothing under `role`: summands of the log-density `p` for which
/// [`bound`](ModelClientOps::bound) is empty, in term order.
///
/// Empty means every summand of `p` carries the role, so what the corresponding reader assembles
/// covers the whole model. A non-empty result means it covers a sub-model, while the gradient it
/// would be paired with covers every term — a Fisher step assembled from the two would solve a
/// mismatched system.
///
/// Proposal-only terms are never reported. They are no part of `p`, so they are expected to
/// contribute nothing, exactly as in the readers this checks.
pub fn terms_missing<B: Backend, C: ModelClientOps<B>>(client: &C, role: Role) -> Vec<Id> {
    client
        .factors()
        .into_iter()
        .filter(|factor| factor.in_density && client.bound(factor.density, role).is_empty())
        .map(|factor| factor.density)
        .collect()
}

/// [`SkoroError::MissingBinding`] naming the first scored term that binds nothing under `role`,
/// or `Ok(())` if every summand of the log-density carries it.
///
/// The guard a consumer that cannot proceed on a partial model calls before it assembles
/// anything. The readers are permissive by design, so this is opt-in strictness rather than a
/// change to them:
///
/// ```ignore
/// require_role(&model, FISHER_ROOT)?;
/// let a = information_root(&model, points, &free)?;
/// // `a` and the gradient it will be paired with now cover the same terms.
/// ```
///
/// Names the first offender and stops, as [`information_blocks`] does for a term with too many
/// roots. Call [`terms_missing`] to report every gap instead of the first.
///
/// [`information_blocks`]: crate::information::information_blocks
pub fn require_role<B: Backend, C: ModelClientOps<B>>(client: &C, role: Role) -> SkoroResult<()> {
    match terms_missing(client, role).first() {
        Some(&id) => Err(SkoroError::MissingBinding { id, role }),
        None => Ok(()),
    }
}
