//! # Operation-trait conventions
//!
//! Across Skoro, operations on quantities (`Q<B, M>`) are exposed through a few
//! recurring trait patterns. New operations should follow whichever of these
//! matches, for consistency:
//!
//! 1. **Operator traits** (e.g. [`QAdd`], [`QSub`], [`QLiteralMul`]). These
//!    back the standard library `std::ops` traits
//!    (`Add`/`Sub`/`Mul`/`Div`) so that `a + b` works on quantities. Define one
//!    per operator; the operator `impl` delegates to it. Because `std::ops`
//!    methods cannot return `Result`, these stay infallible (matching manifolds
//!    only).
//!
//! 2. **Extension `*Ops` traits** (e.g. `LinalgVectorOps`, `PointwiseOps`,
//!    `MatmulOps`, `TrilOps`, `SO3RotateOps`). These provide method-style
//!    operations on `Q<B, M>` (the `itertools`-style extension-trait pattern).
//!    Prefer a single blanket `impl<M: Marker> ...Ops for Q<B, M>` over a small
//!    marker trait when the operation is uniform across manifolds. Introduce a
//!    companion `*Type` / `*Builder` trait on the manifold *only* when the
//!    output quantity type or dimensions vary by operand (e.g. `TransposeType`,
//!    `MatmulQuantityBuilder`, `RotationVector`); fallible dimension checks
//!    there return [`skoro_core::error::SkoroResult`].
//!
//! 3. **`*Backend` dispatch traits** (e.g. `CrossProductBackend`,
//!    `SO3ExpBackend`). Use these when the underlying kernel differs per backend
//!    (CubeCL vs. ndarray). Always provide impls for **both** supported backends
//!    so the operation is usable everywhere.
//!
//! ## Two-track operations & mixed domains
//!
//! Operations come in two tracks (see `docs/TWO_TRACK_OPERATIONS.md`):
//! a differentiable **`DiffKernel`** (continuous output) and a forward-only
//! **`DiscreteKernel`** (discrete output). Both consume `Vec<KernelInput>`, where
//! a [`KernelInput`](skoro_diff::kernel_input::KernelInput) is `Float | Int`.
//!
//! When writing a kernel that reads a **discrete input** (e.g. a continuous map
//! that selects a component by a categorical index):
//! - extract the operand once with `.into_int()` / `.as_int()` (continuous
//!   operands use `.into_float()`); the transparent `Float` arithmetic operators
//!   panic on an `Int` operand — there is no implicit coercion;
//! - the framework already sets the discrete input's tangent to `None`
//!   (it is classified non-differentiable), so do not propagate a tangent for it;
//! - in `vjp`/`hvp`, return one cotangent slot **per input** and put `None` at
//!   every discrete input position.
//!
//! The `Q*` operator traits are unified over unbounded operand types: a
//! discrete output type implements e.g. `QAdd` with a `register` that calls
//! `register_map2` with a [`ForwardOnly`](skoro_graph::operation::ForwardOnly)-wrapped
//! kernel, so `Q<Integer> + Q<Integer>` uses the same single `std::ops::Add`
//! impl as continuous quantities.

use skoro_backend::Backend;
use skoro_core::error::SkoroResult;
use skoro_graph::quantity::ContinuousQuantity;

use crate::{map::register_map2, q::Q};

impl<B: Backend, T: ContinuousQuantity<B>> Q<B, T> {
    pub fn retract<V>(&self, tangent: &Q<B, V>) -> SkoroResult<Q<B, T>>
    where
        T: AssociatedTangentVector<V, B>,
        V: ContinuousQuantity<B>,
    {
        let quantity_type = self.get_quantity_type();
        let retraction = quantity_type.retraction();
        register_map2(quantity_type, retraction, (self, tangent))
    }

    pub fn inverse_retract<V>(&self, other: &Q<B, T>) -> SkoroResult<Q<B, V>>
    where
        T: AssociatedTangentVectorBuilder<V, B>,
        V: ContinuousQuantity<B>,
    {
        let quantity_type = self.get_quantity_type();
        let inverse_retraction = quantity_type.inverse_retraction();
        register_map2(
            quantity_type.build_tangent(),
            inverse_retraction,
            (self, other),
        )
    }
}

mod base;
mod euclidean;
mod matrix;
mod real;

pub use base::*;
