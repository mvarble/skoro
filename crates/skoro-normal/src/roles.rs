//! The roles a normal's term binds, by which a reader in another crate finds its parts.
//!
//! The residual, the log-volume and the information root are named without the normal's prefix:
//! they mean the same of any distribution with a whitened residual or a closed-form information.

use skoro_core::{AtMostOne, ExactlyOne, Role};

use crate::Form;

/// The whitened residual `W (x ⊖ μ)` of a term, or `W (z ⊖ μ)` of a factor.
pub const RESIDUAL: Role<ExactlyOne> = Role::new("skoro.residual");
/// The mean `μ`.
pub const MEAN: Role<ExactlyOne> = Role::new("skoro.normal.mean");
/// The scale given as `σ`, with `Σ = σ²I`.
pub const ISOTROPIC: Role<AtMostOne> = Role::new("skoro.normal.scale.isotropic");
/// The scale given as `σ`, with `Σ = diag(σ)²`.
pub const DIAGONAL: Role<AtMostOne> = Role::new("skoro.normal.scale.diagonal");
/// The scale given as `L`, with `Σ = LLᵀ`.
pub const COVARIANCE_FACTOR: Role<AtMostOne> = Role::new("skoro.normal.scale.covariance_factor");
/// The scale given as `W`, with `Λ = WᵀW`.
pub const WHITENING: Role<AtMostOne> = Role::new("skoro.normal.scale.whitening");
/// The log-volume `λ_μ(x ⊖ μ)`, where it is not identically zero.
pub const LOG_VOLUME: Role<AtMostOne> = Role::new("skoro.log_volume");
/// The root `C` of the expected information, on a term that scores a quantity.
pub const INFORMATION_ROOT: Role<AtMostOne> = Role::new("skoro.information_root");

/// The role that binds a scale of `form`: exactly one of the four is bound.
pub fn scale(form: Form) -> &'static Role<AtMostOne> {
    match form {
        Form::Isotropic => &ISOTROPIC,
        Form::Diagonal => &DIAGONAL,
        Form::CovarianceFactor => &COVARIANCE_FACTOR,
        Form::Whitening => &WHITENING,
    }
}
