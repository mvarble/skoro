//! The precondition that gates the whole algorithm.
//!
//! A model's negative log-density is `½Σ‖rᵢ(x)‖² + Σnᵢ(θ)`. Levenberg-Marquardt minimizes
//! the first term alone, so it is minimizing `−log p` **exactly when the normalizers do not
//! vary with the variables being optimized**. If they do, the optimizer converges — quietly,
//! confidently — to the minimum of a different function.
//!
//! This is not hypothetical. `Tril` is a manifold and therefore a continuous quantity, and
//! the Gaussian facade delegates `add_parameter` and `demote_to_parameter`, so a free
//! covariance factor is constructible through the Gaussian-only builder. "Every factor is
//! Gaussian" is *not* sufficient.

use skoro_backend::Backend;
use skoro_core::error::{SkoroError, SkoroResult};
use skoro_model::{NlsStructure, TangentLayout};

/// Check that the sum-of-squares objective really is this model's negative log-density.
///
/// The condition is **not** "covariances must be constant". A covariance that is a free
/// `Parameter` but is absent from `free` is fine, and that is an ordinary modelling case — a
/// noise level that is fixed but not known at compile time. The correct and weaker condition
/// is the one below: no normalizer parent may be a *column* of the layout being optimized.
///
/// `O(#factors × arity)`, needs no reflection, and inspects no operations: every fact it
/// reads was recorded by the builder when the factor was built.
///
/// # Errors
///
/// [`SkoroError::NormalizerNotConstant`], naming both the offending factor's density and
/// the parent that leaked into the free set — the two things needed to fix the model.
pub fn check_normalizers<B, M>(model: &M, free: &TangentLayout) -> SkoroResult<()>
where
    B: Backend,
    M: NlsStructure<B>,
{
    for factor in model.certificate().factors() {
        for &parent in &factor.normalizer_parents {
            if free.contains(parent) {
                return Err(SkoroError::NormalizerNotConstant {
                    density: factor.density,
                    parent,
                });
            }
        }
    }
    Ok(())
}
