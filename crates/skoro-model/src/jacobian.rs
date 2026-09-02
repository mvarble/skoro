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
//! # Rows may be any continuous node
//!
//! A Jacobian row may be a `Mapped` / `LogDensity` node — recorded in the primal's
//! `evaluations` — or a leaf (`Parameter`, `Stochastic`, `Constant`, `Observed`), which
//! never is. A leaf row is filled in after the fact by `fill_leaf_rows`: the identity on
//! its own columns if it also has an input block (`∂x/∂x = 1`), and zero otherwise — which
//! is already what the flattening produces, so there is nothing to do in that case.
//!
//! A **column** that is not a leaf is not detected here, and cannot be distinguished by
//! these functions from the legitimate case below: a mapped-node column yields a
//! **silently zero column**. That is the one failure mode here that no return value
//! announces.
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
fn batch_and_device<B: Backend>(points: &ValueMap<B>) -> (usize, B::Device) {
    match points.values().next() {
        Some(point) => (point.dims()[0], point.device()),
        None => (1, B::Device::default()),
    }
}

/// Overwrite the rows of `j` (`[batch, Out.total(), In.total()]`) belonging to any output
/// block that is a leaf (absent from `primal.evaluations` — a `Parameter`, `Stochastic`,
/// `Constant` or `Observed` node): the identity on its own columns if it also has an input
/// block, zero otherwise (already the case; nothing to do there).
fn fill_leaf_rows<B: Backend>(
    j: Tensor<B, 3>,
    primal: &ForwardOutput<B>,
    outputs: &TangentLayout,
    inputs: &TangentLayout,
    batch: usize,
    device: &B::Device,
) -> Tensor<B, 3> {
    let mut j = j;
    for r in outputs.blocks() {
        if primal.evaluations.get(&r.id).is_some() {
            continue;
        }
        let Some(c) = inputs.block(r.id) else {
            continue;
        };
        let identity = Tensor::<B, 2>::eye(r.dimension, device)
            .unsqueeze_dim::<3>(0)
            .repeat_dim(0, batch);
        j = j.slice_assign(
            [
                0..batch,
                r.offset..r.offset + r.dimension,
                c.offset..c.offset + c.dimension,
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
/// See the [module documentation](self) for how a leaf row is filled and for what a
/// column that reaches no row does.
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
    Ok(fill_leaf_rows(
        j,
        &out.primal,
        outputs,
        inputs,
        batch,
        &device,
    ))
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
/// See the [module documentation](self) for how a leaf row is filled and for what a
/// column that reaches no row does.
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
    Ok(fill_leaf_rows(
        j,
        &out.primal,
        outputs,
        inputs,
        batch,
        &device,
    ))
}
