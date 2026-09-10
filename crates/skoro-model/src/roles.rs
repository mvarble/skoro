//! The binding roles a term may carry.
//!
//! A binding `(density, role, node)` says that `node` plays `role` for the log-density term
//! `density`. A distribution declares its bindings once, through
//! [`Factor`](crate::distribution::Factor), and any inference algorithm reads back the ones it
//! needs through [`ModelClientOps::bound`](crate::client_ops::ModelClientOps::bound).

pub use skoro_graph::binding::Role;

/// The nodes the term's local views are expressed over, in this order.
///
/// At least one per term. For a conditional `p(x | u₁..u_m)` they are `[x, u₁, .., u_m]`; for a
/// joint factor they are the quantities it scores. The nodes must be distinct and continuous.
pub const ARGUMENT: Role = Role("argument");

/// An `EuclideanD` node `r` with `−log p = ½‖r‖² + n`, `n` the normalizer.
///
/// At most one per term.
pub const RESIDUAL: Role = Role("residual");

/// A node the normalizer `n` in `−log p = ½‖r‖² + n` depends on.
///
/// Any number per term.
pub const NORMALIZER_PARENT: Role = Role("normalizer_parent");

/// The location `μ` in `r = L⁻¹(x ⊖ μ)`.
///
/// At most one per term.
pub const LOCATION: Role = Role("location");

/// The whitening `L` in `r = L⁻¹(x ⊖ μ)`, with `Σ = LLᵀ`.
///
/// At most one per term.
pub const WHITENING: Role = Role("whitening");

/// A Fisher information root: a node `C` of type `EuclideanMatrixD(k, D)`, `D` the summed
/// manifold dimension of the term's [`ARGUMENT`]s, such that `CᵀC` is the expected information
/// of the term over the arguments' tangent coordinates, in `ARGUMENT` order.
///
/// At most one per term. `k` need not equal `D`.
pub const FISHER_ROOT: Role = Role("fisher_root");
