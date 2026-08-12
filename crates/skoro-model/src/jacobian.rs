//! Dense Jacobian assembly over flattened tangent spaces.
//!
//! Given an **input layout** (the columns) and an **output layout** (the rows), both
//! paths here produce the same `[batch, D_out, D_in]` tensor, where `D_in` and `D_out`
//! are the layouts' summed *manifold* dimensions. They differ only in cost:
//! [`jacobian_jvp`] pays `D_in` frames in one forward pass, [`jacobian_vjp`] pays
//! `D_out` frames in one reverse pass. A tall Jacobian (`D_out ≪ D_in` — a
//! low-dimensional residual over a large state) favours the VJP path, which is the case
//! a least-squares problem is usually in.
//!
//! These take a client rather than a graph, which is why they live here and not next to
//! [`TangentLayout`] in `skoro-graph`.
//!
//! # Structural constraint: columns are leaves, rows are mapped nodes
//!
//! This is a constraint on *how a model is written*, not just on how these functions are
//! called, and it is not enforced by the type system:
//!
//! > **Jacobian columns must be `Parameter` / `Stochastic` leaves. Jacobian rows must be
//! > `Mapped` / `LogDensity` nodes.**
//!
//! It follows from the compiler, and both passes agree on it. `vjp_forward` collects
//! cotangents only at `CompiledStep::Parameter` / `Sampler`, so nothing else can be a column;
//! `jvp_forward` emits tangents only in the `CompiledStep::Operation` branch, and `vjp_forward`
//! reads its seeds only there too, so nothing else can be a row.
//!
//! For a filter chain this is natural — `x_k` is a leaf and `f(x_k)`, `h(x_k)` are mapped
//! — but a state chain built entirely out of mapped nodes cannot be differentiated
//! block-to-block. A correlated joint over several quantities is no exception: it is
//! modelled as a likelihood factor over `Parameter` leaves, and those are ordinary
//! columns.
//!
//! A **row** that breaks the rule is reported as [`SkoroError::NotMapped`]. A **column**
//! that breaks it is not, and cannot be distinguished by these functions from the
//! legitimate case below: a mapped-node column yields a **silently zero column**. That is
//! the one failure mode here that no return value announces.
//!
//! # A leaf that does not reach the rows is not an error
//!
//! A column whose quantity has no path to any row gets a zero column, not a failure. The
//! derivative genuinely is zero, and this is an ordinary occurrence — any layout whose
//! column set is wider than the subgraph feeding its rows hits it. Both paths agree:
//! `vjp_forward` zero-defaults such a leaf, and
//! [`TangentLayout::flatten_frames`](skoro_graph::tangent_layout::TangentLayout::flatten_frames)
//! zero-fills a missing id.
//!
//! # Batch and device
//!
//! Both are taken from the first entry of `points`, exactly as `Graph::evaluate_ids_jvp`
//! and `Graph::evaluate_ids_vjp` take the batch size, so the seed frames cannot end up a
//! different width — or on a different device — from the pass that consumes them. An
//! empty `points` falls back to batch 1 on the default device.

use skoro_backend::{Backend, Tensor};
use skoro_core::{
    error::{SkoroError, SkoroResult},
    id::Id,
};
use skoro_graph::{compiler::ForwardOutput, tangent_layout::TangentLayout};
use skoro_operation::ValueMap;

use crate::client_ops::ModelClientOps;

/// Batch size and device, chosen the same way the evaluation pass chooses its batch size.
fn batch_and_device<B: Backend>(points: &ValueMap<B>) -> (usize, B::Device) {
    match points.values().next() {
        Some(point) => (point.dims()[0], point.device()),
        None => (1, B::Device::default()),
    }
}

/// Reject a row that is not a mapped node.
///
/// Every compile target lands in the sequence, and every `Mapped` / `LogDensity` node
/// lowers to a `CompiledStep::Operation`, whose value is recorded in `evaluations`. So a
/// row absent from `evaluations` is a `Parameter`, `Sampler`, `Constant` or `Observed`
/// node — none of which receives a seed in either pass.
///
/// Without this the result would be a zero row, which for a leaf is *wrong* rather than
/// merely useless: `∂x/∂x` is the identity. Both paths produce the same zero, so the
/// JVP/VJP agreement check cannot catch it either.
fn check_rows<B: Backend>(primal: &ForwardOutput<B>, outputs: &TangentLayout) -> SkoroResult<()> {
    let unmapped = outputs
        .ids()
        .find(|id| primal.evaluations.get(id).is_none());
    match unmapped {
        Some(id) => Err(SkoroError::NotMapped { id }),
        None => Ok(()),
    }
}

/// `J ∈ [batch, Out.total(), In.total()]` by one forward pass with `In.total()` frames.
///
/// Each input block `j` is seeded with rows `o_j .. o_j + d_j` of `I_{D_in}`, so output
/// block `o` comes back as the `[batch, d_o, D_in]` slab of `J` belonging to those rows —
/// concatenating the output blocks in layout order is the whole assembly. **No
/// transpose**, unlike [`jacobian_vjp`].
///
/// Costs `D_in` frames; prefer [`jacobian_vjp`] when the Jacobian is tall.
///
/// See the [module documentation](self) for the constraint that columns must be leaves
/// and rows must be mapped nodes, and for what a column that reaches no row does.
pub fn jacobian_jvp<B: Backend, C: ModelClientOps<B>>(
    client: &C,
    points: &ValueMap<B>,
    inputs: &TangentLayout,
    outputs: &TangentLayout,
) -> SkoroResult<Tensor<B, 3>> {
    let (batch, device) = batch_and_device(points);
    let seeds = inputs.identity_frames(batch, &device);
    let ids: Vec<Id> = outputs.ids().collect();
    let out = client.evaluate_ids_jvp(points, &seeds, &ids)?;
    check_rows(&out.primal, outputs)?;
    outputs.flatten_frames(&out.tangents, batch, inputs.total(), &device)
}

/// The same `J ∈ [batch, Out.total(), In.total()]`, by one reverse pass with
/// `Out.total()` frames.
///
/// Each output block `o` is seeded with rows `p_o .. p_o + d_o` of `I_{D_out}`, so input
/// block `j` comes back as a `[batch, d_j, D_out]` slab — that is `Jᵀ`, and concatenating
/// the *input* blocks gives `[batch, D_in, D_out]`, which the final `swap_dims(1, 2)`
/// turns into `J`.
///
/// **That transpose is where this pair fails silently.** [`jacobian_jvp`] needs none, and
/// getting it wrong here yields a well-shaped, wrong `J` whenever `D_in == D_out`. Both
/// seedings deliberately route through the single
/// [`TangentLayout::identity_frames`](skoro_graph::tangent_layout::TangentLayout::identity_frames)
/// helper so the two index conventions cannot drift apart.
///
/// Costs `D_out` frames; prefer this to [`jacobian_jvp`] when the Jacobian is tall, which
/// is the usual shape of a least-squares problem.
///
/// See the [module documentation](self) for the constraint that columns must be leaves
/// and rows must be mapped nodes, and for what a column that reaches no row does.
pub fn jacobian_vjp<B: Backend, C: ModelClientOps<B>>(
    client: &C,
    points: &ValueMap<B>,
    inputs: &TangentLayout,
    outputs: &TangentLayout,
) -> SkoroResult<Tensor<B, 3>> {
    let (batch, device) = batch_and_device(points);
    let seeds = outputs.identity_frames(batch, &device);
    let ids: Vec<Id> = outputs.ids().collect();
    let out = client.evaluate_ids_vjp(points, &seeds, &ids)?;
    check_rows(&out.primal, outputs)?;
    let transposed = inputs.flatten_frames(&out.cotangents, batch, outputs.total(), &device)?;
    Ok(transposed.swap_dims(1, 2))
}
