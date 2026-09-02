//! The binding roles this crate defines. See `docs/superpowers/specs/2026-09-01-factor-bindings-and-fisher-root-spec.md` §2/§3.4 for the full contract table.

pub use skoro_graph::binding::Role;

/// The term's local views are expressed over these nodes, in this order. Cardinality ≥ 1 per
/// term; nodes must be distinct and continuous.
pub const ARGUMENT: Role = Role("argument");

/// An `EuclideanD` node `r` with `−log p = ½‖r‖² + n`. Cardinality 0 or 1 per term.
pub const RESIDUAL: Role = Role("residual");

/// Every node the normalizer `n` depends on. Any cardinality per term.
pub const NORMALIZER_PARENT: Role = Role("normalizer_parent");

/// `μ` in `r = L⁻¹(x ⊖ μ)`. Cardinality 0 or 1 per term.
pub const LOCATION: Role = Role("location");

/// `L` in `r = L⁻¹(x ⊖ μ)`, with `Σ = LLᵀ`. Cardinality 0 or 1 per term.
pub const WHITENING: Role = Role("whitening");

/// `EuclideanMatrixD(k, Σ dim(ARGUMENT))` with `CᵀC` equal to the expected information over the
/// arguments, in `ARGUMENT` order. Cardinality 0 or 1 per term.
pub const FISHER_ROOT: Role = Role("fisher_root");
