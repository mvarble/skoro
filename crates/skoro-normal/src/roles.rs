//! The roles a normal's term binds, by which a reader in another crate finds its parts.
//!
//! The residual, the log-volume and the information root are named without the normal's prefix:
//! they mean the same of any term with a whitened residual or a closed-form information, the
//! improper factor of [`Improper`](crate::Improper) among them.

use skoro_core::{AtMostOne, ExactlyOne, Role};

use crate::Form;

/// The whitened residual: `W (x ⊖ μ)` of a normal's term, `W (z ⊖ μ)` of its factor, and
/// `A δ - b` of an improper factor, of any number of components.
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
/// The varying part of a term's remainder: what is left of its log-density after the squares of
/// its residual and a normalizer constant in its free quantities, where that is not identically
/// zero. On a normal it is the chart's log-volume `λ_μ(x ⊖ μ)`, and on an improper factor its
/// linear part `ηᵀδ`. It is not the volume of a chart: a reader that needs a quantity's chart
/// volume reads it from the quantity's type.
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

/// The tangent `s = x ⊖ μ` of a normal's term, which scores a quantity: its residual is `W s`,
/// with `W` the whitening of the bound scale, and its log-volume, where it binds one, is
/// `λ_μ(s)`, the chart's log-volume of the scored quantity's space at `μ`.
pub const TANGENT: Role<AtMostOne> = Role::new("skoro.normal.tangent");
