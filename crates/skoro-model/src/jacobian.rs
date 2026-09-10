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
//! # Structural constraint: columns are leaves
//!
//! This is a constraint on *how a model is written*, not just on how these functions are
//! called, and it is not enforced by the type system:
//!
//! > **Jacobian columns must be `Parameter` / `Stochastic` leaves.**
//!
//! It follows from the compiler, and both passes agree on it. `vjp_forward` collects
//! cotangents only at `CompiledStep::Parameter` / `Sampler`, so nothing else can be a column.
//! A column that breaks the rule cannot be distinguished by these functions from the
//! legitimate case below: a mapped-node column yields a **silently zero column**. That is
//! the one failure mode here that no return value announces.
//!
//! For a filter chain this is natural — `x_k` is a leaf and `f(x_k)`, `h(x_k)` are mapped
//! — but a state chain built entirely out of mapped nodes cannot be differentiated
//! block-to-block. A correlated joint over several quantities is no exception: it is
//! modelled as a likelihood factor over `Parameter` leaves, and those are ordinary
//! columns.
//!
//! # Rows may be any continuous node
//!
//! A mapped row is differentiated by the passes. A **leaf row** — a `Parameter`,
//! `Stochastic`, `Constant` or `Observed` node, which neither pass seeds — is filled in
//! here instead: the identity on its own columns when it is also a column, and zero
//! everywhere else. A constant or observed row is therefore all zero, which is correct: it
//! does not move with any column. This is what lets a term's arguments, which usually
//! include leaves, be the rows of a Jacobian.
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
use skoro_core::{error::SkoroResult, id::Id};
use skoro_graph::{compiler::ForwardOutput, tangent_layout::TangentLayout};
use skoro_operation::ValueMap;

use crate::client_ops::ModelClientOps;

/// Batch size and device, chosen the same way the evaluation pass chooses its batch size.
pub(crate) fn batch_and_device<B: Backend>(points: &ValueMap<B>) -> (usize, B::Device) {
    match points.values().next() {
        Some(point) => (point.dims()[0], point.device()),
        None => (1, B::Device::default()),
    }
}

/// Fill in the rows of `j` that neither pass seeds.
///
/// Every compile target lands in the sequence, and every `Mapped` / `LogLikelihood` node
/// lowers to a `CompiledStep::Operation`, whose value is recorded in `evaluations`. So a
/// row absent from `evaluations` is a leaf, and both passes left it zero. For a leaf that is
/// also a column that is *wrong* — `∂x/∂x` is the identity — and both paths would agree on
/// the wrong answer, so the identity block is written here.
fn fill_leaf_rows<B: Backend>(
    mut j: Tensor<B, 3>,
    primal: &ForwardOutput<B>,
    inputs: &TangentLayout,
    outputs: &TangentLayout,
    batch: usize,
    device: &B::Device,
) -> Tensor<B, 3> {
    for row in outputs.blocks() {
        if primal.evaluations.get(&row.id).is_some() {
            continue;
        }
        let Some(column) = inputs.block(row.id) else {
            continue;
        };
        let identity = Tensor::<B, 2>::eye(row.dimension, device)
            .unsqueeze_dim::<3>(0)
            .repeat_dim(0, batch);
        j = j.slice_assign(
            [
                0..batch,
                row.offset..row.offset + row.dimension,
                column.offset..column.offset + column.dimension,
            ],
            identity,
        );
    }
    j
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
/// See the [module documentation](self) for the constraint that columns must be leaves,
/// for how leaf rows are filled, and for what a column that reaches no row does.
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
    let j = outputs.flatten_frames(&out.tangents, batch, inputs.total(), &device)?;
    Ok(fill_leaf_rows(j, &out.primal, inputs, outputs, batch, &device))
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
/// See the [module documentation](self) for the constraint that columns must be leaves,
/// for how leaf rows are filled, and for what a column that reaches no row does.
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
    let transposed = inputs.flatten_frames(&out.cotangents, batch, outputs.total(), &device)?;
    let j = transposed.swap_dims(1, 2);
    Ok(fill_leaf_rows(j, &out.primal, inputs, outputs, batch, &device))
}
