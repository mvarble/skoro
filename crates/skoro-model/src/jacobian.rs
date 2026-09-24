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
//! # Columns are leaves, and the rule is enforced
//!
//! > **Jacobian columns must be `Parameter` / `Stochastic` leaves.**
//!
//! A column is something an estimator moves, and an estimator moves the free leaves. Both
//! functions check their `inputs` before any pass runs, and a `Mapped` or `LogLikelihood` column
//! is [`SkoroError::MappedColumn`](skoro_core::error::SkoroError::MappedColumn). Without the check
//! it would come back as a silently zero column, because `vjp_forward` collects cotangents only
//! at leaf steps, and that zero cannot be told apart from the legitimate one below. The check
//! runs up front rather than on the pass's output, so it also catches a mapped column that
//! reaches no row, and one whose constant fold lowered it to a constant step.
//!
//! `Constant` and `Observed` columns are accepted and come back zero (see the rows section).
//! `observe` turns a stochastic column into an observed one at the same id, and a filter's
//! layout must not start failing because it observed its state.
//!
//! Interior columns are left out on purpose, not deferred. Optimization never needs one: the
//! passes already apply the chain rule *through* mapped nodes to reach the free leaves. A
//! linearization at an interior node, such as `∂h/∂x₁` for `x₁ = f(x₀)` in an all-mapped chain,
//! is a different question, and
//! [`Graph::compile_between`](skoro_graph::graph::Graph::compile_between) answers it by cutting
//! at that node. A correlated joint needs no interior columns either: it is a likelihood factor
//! over `Parameter` leaves, and those are ordinary columns.
//!
//! # Rows may be any continuous node
//!
//! A mapped row is differentiated by the passes. A **leaf row** — a `Parameter`,
//! `Stochastic`, `Constant` or `Observed` node, which neither pass seeds — is filled in
//! here instead. A `Parameter` or unobserved `Stochastic` row is the identity on its own
//! columns when it is also a column, and zero everywhere else. A `Constant` or `Observed`
//! row is all zero, even when the caller also lists it as a column: it does not move with
//! any column, and neither pass seeds a constant, so its own column is zero too. This is
//! what lets a term's arguments, which usually include leaves, be the rows of a Jacobian.
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

use crate::client_ops::ModelReaderOps;

/// Batch size and device, chosen the same way the evaluation pass chooses its batch size.
pub(crate) fn batch_and_device<B: Backend>(points: &ValueMap<B>) -> (usize, B::Device) {
    match points.values().next() {
        Some(point) => (point.dims()[0], point.device()),
        None => (1, B::Device::default()),
    }
}

/// Fill in the rows of `j` that neither pass seeds.
///
/// Every compile target lands in the sequence. A `Mapped` / `LogLikelihood` node lowers to a
/// `CompiledStep::Operation`, recorded in `evaluations`; a `Parameter` or `Sampler` step is
/// recorded in `variables`; a `Constant` step, which is also how an observed node lowers, is
/// recorded in `constants`. Both passes leave every leaf row zero.
///
/// For a variable that is also a column that is *wrong* — `∂x/∂x` is the identity — and both
/// paths would agree on the wrong answer, so the identity block is written here. A constant
/// or observed row stays zero even when it is also listed as a column: neither pass seeds a
/// constant, so that column is zero as well, and the two stay consistent.
fn fill_leaf_rows<B: Backend>(
    mut j: Tensor<B, 3>,
    primal: &ForwardOutput<B>,
    inputs: &TangentLayout,
    outputs: &TangentLayout,
    batch: usize,
    device: &B::Device,
) -> Tensor<B, 3> {
    for row in outputs.blocks() {
        if primal.variables.get(&row.id).is_none() {
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
/// A `Mapped` or `LogLikelihood` column is
/// [`SkoroError::MappedColumn`](skoro_core::error::SkoroError::MappedColumn), checked before the
/// pass runs. See the [module documentation](self) for why columns must be leaves, for how leaf
/// rows are filled, and for what a column that reaches no row does.
pub fn jacobian_jvp<B: Backend, C: ModelReaderOps<B>>(
    client: &C,
    points: &ValueMap<B>,
    inputs: &TangentLayout,
    outputs: &TangentLayout,
) -> SkoroResult<Tensor<B, 3>> {
    client.ensure_leaf_columns(inputs)?;
    let (batch, device) = batch_and_device(points);
    let seeds = inputs.identity_frames(batch, &device);
    let ids: Vec<Id> = outputs.ids().collect();
    let out = client.evaluate_ids_jvp(points, &seeds, &ids)?;
    let j = outputs.flatten_frames(&out.tangents, batch, inputs.total(), &device)?;
    Ok(fill_leaf_rows(
        j,
        &out.primal,
        inputs,
        outputs,
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
/// A `Mapped` or `LogLikelihood` column is
/// [`SkoroError::MappedColumn`](skoro_core::error::SkoroError::MappedColumn), checked before the
/// pass runs. See the [module documentation](self) for why columns must be leaves, for how leaf
/// rows are filled, and for what a column that reaches no row does.
pub fn jacobian_vjp<B: Backend, C: ModelReaderOps<B>>(
    client: &C,
    points: &ValueMap<B>,
    inputs: &TangentLayout,
    outputs: &TangentLayout,
) -> SkoroResult<Tensor<B, 3>> {
    client.ensure_leaf_columns(inputs)?;
    let (batch, device) = batch_and_device(points);
    let seeds = outputs.identity_frames(batch, &device);
    let ids: Vec<Id> = outputs.ids().collect();
    let out = client.evaluate_ids_vjp(points, &seeds, &ids)?;
    let transposed = inputs.flatten_frames(&out.cotangents, batch, outputs.total(), &device)?;
    let j = transposed.swap_dims(1, 2);
    Ok(fill_leaf_rows(
        j,
        &out.primal,
        inputs,
        outputs,
        batch,
        &device,
    ))
}

/// The ids of `ids` that `layout` holds, in `layout`'s order.
pub(crate) fn in_layout_order(
    layout: &TangentLayout,
    ids: impl IntoIterator<Item = Id>,
) -> Vec<Id> {
    let mut blocks: Vec<_> = ids.into_iter().filter_map(|id| layout.block(id)).collect();
    blocks.sort_unstable_by_key(|block| block.offset);
    blocks.dedup_by_key(|block| block.id);
    blocks.into_iter().map(|block| block.id).collect()
}

/// One row node's Jacobian over the free leaves it can reach.
pub struct RowJacobian<B: Backend> {
    /// The row node.
    pub row: Id,
    /// The free leaves the row depends on — [`leaf_support`](ModelReaderOps::leaf_support)
    /// restricted to the columns — in the columns' order. Every other column of the row is zero.
    pub keys: TangentLayout,
    /// `∂row/∂keys`: `[batch, dim(row), keys.total()]`.
    pub jacobian: Tensor<B, 3>,
}

/// What [`jacobian_blocks`] returns.
pub struct JacobianBlocks<B: Backend> {
    /// One entry per distinct row, in first-appearance order.
    pub rows: Vec<RowJacobian<B>>,
    /// Frames the reverse pass carried: the sum over colours of the widest row in each. Dense
    /// [`jacobian_vjp`] would carry the summed dimension of every row.
    pub frames: usize,
}

/// The block-sparse Jacobian: each row node's derivative over only the free leaves it reaches, by
/// one compressed reverse pass.
///
/// Dense [`jacobian_vjp`] seeds every row in its own frames and returns every row over every
/// column, which is `O(D_out · D_in)` in both frames and memory. Here rows whose free-leaf supports
/// are disjoint are **coloured** alike and seeded into the same frames: a leaf's cotangent in those
/// frames is then the contribution of the one row of that colour that reaches it, so each block is
/// read back unmixed. A pose chain needs about three colours however long it is.
///
/// Rows are coloured as distinct nodes, not per term, since terms share rows — the argument `x_k`
/// of one factor is an argument of the next. A repeated id is computed once. One pass means one
/// compile, so the bounded program cache is not churned by a compile per term.
///
/// Columns follow the rule of the [module documentation](self): a `Mapped` or `LogLikelihood`
/// column is [`SkoroError::MappedColumn`](skoro_core::error::SkoroError::MappedColumn). A leaf row
/// that is itself a column gets its identity block, as in [`jacobian_vjp`].
pub fn jacobian_blocks<B: Backend, C: ModelReaderOps<B>>(
    client: &C,
    points: &ValueMap<B>,
    inputs: &TangentLayout,
    rows: &[Id],
) -> SkoroResult<JacobianBlocks<B>> {
    client.ensure_leaf_columns(inputs)?;
    let (batch, device) = batch_and_device(points);

    let mut seen = std::collections::HashSet::new();
    let rows: Vec<Id> = rows.iter().copied().filter(|id| seen.insert(*id)).collect();
    let row_layout = client.tangent_layout(&rows)?;

    // Each row's keys, in the columns' order: its support, looked up in the layout rather than the
    // layout scanned, so the cost follows the support and not the number of columns.
    let keys: Vec<Vec<Id>> = rows
        .iter()
        .map(|&row| in_layout_order(inputs, client.leaf_support(&[row])))
        .collect();

    // Greedy colouring: a row joins the first colour whose keys it does not touch.
    let mut colour_keys: Vec<std::collections::HashSet<Id>> = Vec::new();
    let mut colour_width: Vec<usize> = Vec::new();
    let mut colour_of: Vec<Option<usize>> = Vec::with_capacity(rows.len());
    for (i, row_keys) in keys.iter().enumerate() {
        if row_keys.is_empty() {
            colour_of.push(None);
            continue;
        }
        let width = row_layout.blocks()[i].dimension;
        let colour = colour_keys
            .iter()
            .position(|used| row_keys.iter().all(|k| !used.contains(k)))
            .unwrap_or_else(|| {
                colour_keys.push(Default::default());
                colour_width.push(0);
                colour_keys.len() - 1
            });
        colour_keys[colour].extend(row_keys.iter().copied());
        colour_width[colour] = colour_width[colour].max(width);
        colour_of.push(Some(colour));
    }
    let mut colour_offset = Vec::with_capacity(colour_width.len());
    let mut frames = 0;
    for width in &colour_width {
        colour_offset.push(frames);
        frames += width;
    }

    // Seed row `i` with the identity in its colour's frames.
    let mut seeds = std::collections::HashMap::new();
    let mut targets = Vec::new();
    for (i, &row) in rows.iter().enumerate() {
        let Some(colour) = colour_of[i] else {
            continue;
        };
        let d = row_layout.blocks()[i].dimension;
        let at = colour_offset[colour];
        let seed = Tensor::<B, 2>::eye(frames, &device)
            .slice_dim(0, at..at + d)
            .unsqueeze_dim::<3>(0)
            .repeat_dim(0, batch);
        seeds.insert(row, seed);
        targets.push(row);
    }
    let out = if targets.is_empty() {
        None
    } else {
        Some(client.evaluate_ids_vjp(points, &seeds, &targets)?)
    };

    let mut result = Vec::with_capacity(rows.len());
    for (i, &row) in rows.iter().enumerate() {
        let d = row_layout.blocks()[i].dimension;
        let layout = client.tangent_layout(&keys[i])?;
        let jacobian = match (colour_of[i], &out) {
            (Some(colour), Some(out)) => {
                let at = colour_offset[colour];
                let parts: Vec<Tensor<B, 3>> = layout
                    .blocks()
                    .iter()
                    .map(|key| {
                        if key.id == row && out.primal.variables.get(&row).is_some() {
                            // `∂x/∂x`, which neither pass seeds; see `fill_leaf_rows`.
                            return Tensor::<B, 2>::eye(d, &device)
                                .unsqueeze_dim::<3>(0)
                                .repeat_dim(0, batch);
                        }
                        match out.cotangents.get(&key.id) {
                            Some(cotangent) => {
                                cotangent.clone().slice_dim(2, at..at + d).swap_dims(1, 2)
                            }
                            None => Tensor::zeros([batch, d, key.dimension], &device),
                        }
                    })
                    .collect();
                Tensor::cat(parts, 2)
            }
            _ => Tensor::zeros([batch, d, 0], &device),
        };
        result.push(RowJacobian {
            row,
            keys: layout,
            jacobian,
        });
    }
    Ok(JacobianBlocks {
        rows: result,
        frames,
    })
}
