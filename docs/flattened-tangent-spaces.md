# Flattened Tangent Spaces

A design for ordering the tangent spaces of a set of quantities into contiguous blocks, so that
inference algorithms can work in dense block matrices while the model stays manifold-aware.

## 1. Motivation

Skoro's autodiff surface is keyed by `Id`. Every derivative-producing entry point takes and
returns `HashMap<Id, Tensor<..>>`:

- `skoro-graph/src/graph.rs:1533-1592` — `PointJvpEval`, `PointVjpEval`, `PointHvpEval`,
  `PointEvalOrder1/2`, and `EvalOrder1/2` all carry `HashMap<Id, TangentVector<B>>` or
  `HashMap<Id, CotangentFrame<B>>`.
- `skoro-model/src/model.rs` and `skoro-model/src/client_ops.rs` mirror those signatures.
  `skoro-model/src/hash_map_builders.rs` exists purely to make those maps ergonomic to build.

This is the right representation for a *graph*, and the wrong representation for the *linear
algebra*. An Extended Kalman Filter does not want per-quantity gradients; it wants a covariance
`P ∈ ℝ^{N×N}`, a measurement Jacobian `H ∈ ℝ^{M×N}`, and a correction `δ ∈ ℝ^N`. Gauss-Newton
wants a residual Jacobian and a normal-equation solve. Both need a stable assignment of each
quantity's tangent space to a contiguous range of indices.

Nothing in the workspace provides one. The closest thing is `id_to_index` inside
`Graph::compile_backward_from` (`skoro-graph/src/graph.rs:1372`), which is per-node-slot, local
to compilation, and never escapes.

This document proposes a `TangentLayout` that pins down such an ordering once, the flatten /
unflatten operations built on it, a Jacobian-assembly API, and the dense linear algebra the
consuming algorithms will need.

### Why Gauss-Newton rather than Newton

`hvp_forward` (`skoro-graph/src/compiler.rs:373`) takes a **rank-2** `TangentVector` per id, so
one pass yields exactly one Hessian-vector product. A dense Hessian therefore costs `N` passes.
A Gauss-Newton `JᵀJ` costs one Jacobian extraction (§4). That asymmetry is a large part of why
the flattened Jacobian API is worth building.

## 2. The invariant: dimension, not storage

> A flattened tangent vector is laid out by **manifold dimension**
> (`DynManifold::dimension()`), never by **storage size**
> (`NumComponents::num_components()`).

Every shape downstream depends on this. The two coincide for flat manifolds and diverge exactly
where it matters:

| quantity | `dimension()` | `num_components()` | source |
|---|---|---|---|
| `Real` | 1 | 1 | `skoro-riemann/src/real.rs:12` |
| `EuclideanD(n)` | n | n | `skoro-riemann/src/euclidean.rs:112` |
| `EuclideanMatrixD(m,n)` | m·n | m·n | `skoro-riemann/src/matrix.rs:167` |
| `Tril(n)` | n(n+1)/2 | n(n+1)/2 | `skoro-linalg/src/tril/manifold.rs:22` |
| `SO3` | **3** | **9** | `skoro-so3/src/quantity.rs:23` |
| `SE3` | **6** | **12** | `skoro-se3/src/quantity.rs:27` |

The two sizes are reachable from a kind-erased node through `DynQuantityType<B>`
(`skoro-graph/src/quantity.rs:174-188`): `qt.as_manifold()?.dimension()` for the tangent
dimension, `qt.num_components()` for storage. A layout uses only the former.

### Corollary: points never flatten

> The state stays on the manifold, in a `PointMap<B>`. Only tangents and cotangents flatten. A
> covariance `P` is an object on the tangent space *at the current point*, shape
> `[batch, N, N]`. A step is applied by `retract(unflatten(δ))`, never by vector addition.

This is what makes the design coherent. It is why no parallel storage-size layout is needed, and
why an EKF over `SE3` requires no special-casing: the filter's arithmetic is entirely in ℝ⁶ per
pose, and the only manifold-aware operation is the retraction at the end of the update.

## 3. `TangentLayout`

New file `crates/skoro-graph/src/tangent_layout.rs`, alongside `point_map.rs` — it needs
`DynQuantityType::as_manifold()` (`skoro-graph/src/quantity.rs:184`) and `Graph` for lookups.
Re-exported from `skoro-model` alongside `hash_map_builders`.

It lives in `skoro-graph`, below every algorithm crate, because EKF, Gauss-Newton, HMC, and MALA
all need the same object — HMC and MALA for their momentum and drift vectors, the other two for
Jacobians and covariances.

```rust
/// A contiguous ordering of the tangent spaces of a set of quantities.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TangentLayout {
    /// Ordered blocks, in the caller's order.
    blocks: Vec<TangentBlock>,
    /// Reverse index into `blocks`.
    index: HashMap<Id, usize>,
    /// Sum of all block dimensions.
    total: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TangentBlock {
    pub id: Id,
    pub offset: usize,
    pub dimension: usize,
}
```

```rust
impl TangentLayout {
    /// Primary constructor: preserves the caller's ordering.
    ///
    /// Errors with `NotContinuous` for a discrete id, `NodeNotFound` for an unknown
    /// one, and `DuplicateLayoutId` if an id appears twice.
    pub fn from_ids<B: Backend>(graph: &Graph<B>, ids: &[Id]) -> SkoroResult<Self>;

    /// Convenience: the same ids in ascending `Id` order.
    pub fn from_ids_sorted<B: Backend>(graph: &Graph<B>, ids: &[Id]) -> SkoroResult<Self>;

    pub fn total(&self) -> usize;
    pub fn blocks(&self) -> &[TangentBlock];
    pub fn block(&self, id: Id) -> Option<TangentBlock>;
    pub fn contains(&self, id: Id) -> bool;
    pub fn ids(&self) -> impl Iterator<Item = Id> + '_;

    /// Block dimensions equal, in order. Used to check that two layouts describe the
    /// same tangent space over different quantities — e.g. an EKF's `state` and
    /// `state_next`, which are paired positionally.
    pub fn same_shape(&self, other: &Self) -> bool;
}
```

Ordering is caller-specified by design: a filter's state ordering is a modelling decision, and a
block structure chosen by the caller is one they can reason about when reading a covariance
matrix. `from_ids_sorted` covers the case where the caller does not care.

Notes:

- `Id` (`skoro-core/src/id.rs:4`) derives only `PartialEq, Eq, Hash`. Add `PartialOrd, Ord` for
  `from_ids_sorted`. This is non-breaking, and `Id` already derefs to `u64`.
- A layout is built once and reused across every iteration of a filter or optimizer loop. It
  holds no tensors and carries no backend parameter, so it is cheap to clone and store.
- A layout is **not** revalidated against a graph after construction. Reusing one after
  `Graph::remove` on one of its ids is a caller error.

New `SkoroError` variants — the enum is `#[non_exhaustive]` (`skoro-core/src/error.rs:25`), so
this is additive:

```rust
DuplicateLayoutId { id: Id },
IdNotInLayout { id: Id },
LayoutDimensionMismatch { expected: usize, found: usize },
```

Prefer these over panics throughout the layout surface.

## 4. Flatten and unflatten

Methods on `TangentLayout`, over the existing aliases in `skoro-backend/src/lib.rs:90-105`:

```rust
// rank-2: [batch, d_i] blocks  <->  [batch, N]
pub fn flatten_vectors<B: Backend>(
    &self,
    map: &HashMap<Id, TangentVector<B>>,
    batch: usize,
    device: &B::Device,
) -> SkoroResult<Tensor<B, 2>>;

pub fn unflatten_vector<B: Backend>(
    &self,
    flat: Tensor<B, 2>,
) -> SkoroResult<HashMap<Id, TangentVector<B>>>;

// rank-3: [batch, d_i, f] blocks  <->  [batch, N, f]
pub fn flatten_frames<B: Backend>(
    &self,
    map: &HashMap<Id, CotangentFrame<B>>,
    batch: usize,
    frames: usize,
    device: &B::Device,
) -> SkoroResult<Tensor<B, 3>>;

pub fn unflatten_frame<B: Backend>(
    &self,
    flat: Tensor<B, 3>,
) -> SkoroResult<HashMap<Id, CotangentFrame<B>>>;
```

- Flattening is a **single** `Tensor::cat` over the ordered block list along dim 1, not a
  pairwise fold.
- Ids missing from the map are filled with `Tensor::zeros([batch, d_i, ..])` rather than
  erroring. This is deliberate: a VJP legitimately returns no cotangent for a leaf that does not
  reach the target.
- Unflattening is `narrow(1, offset, dimension)` per block.
- Every entry point validates `flat.dims()[1] == self.total()` and returns
  `LayoutDimensionMismatch` otherwise.

A companion helper, shared by both Jacobian paths in §5:

```rust
/// Per-id blocks of the `N × N` identity, where `N = self.total()`:
/// block `j` is rows `o_j .. o_j + d_j`, shape `[batch, d_j, N]`.
pub fn identity_frames<B: Backend>(
    &self,
    batch: usize,
    device: &B::Device,
) -> HashMap<Id, Tensor<B, 3>>;
```

On the builder side, `skoro-model/src/hash_map_builders.rs` currently takes a `&Q<B, T>` only in
`InputMapBuilder::insert_data`; the tangent builders (`TensorMapBuilder::with_tensor`,
`hash_map_builders.rs:53`) take a raw `Id`. Adding a `with_quantity(&Q<B, T>, tensor)` variant
keeps the layout API consistent with the rest of the model surface.

### `PointMap::retract_flat`

Added to `skoro-graph/src/point_map.rs`:

```rust
pub fn retract_flat(
    &self,
    layout: &TangentLayout,
    delta: Tensor<B, 2>,
) -> SkoroResult<Self>;

pub fn retract_flat_mut(
    &mut self,
    layout: &TangentLayout,
    delta: Tensor<B, 2>,
) -> SkoroResult<()>;
```

This deliberately differs from the existing `retract_mut` (`point_map.rs:142`), which requires a
tangent for **every** point in the map and errors `MissingPoint` otherwise. `retract_flat`
retracts only the ids in the layout and leaves the rest untouched — which is what an EKF update
needs, since the `PointMap` also holds constants and nuisance quantities.

### Retraction is not state advance

Easy to conflate, and the distinction decides which operation an algorithm reaches for:

> **Retraction applies a correction** to a point — an EKF update, a Gauss-Newton step. The new
> value is `retract(x, δ)` for a tangent `δ` in the same layout.
>
> **State advance replaces values** — `x_{k+1} = f(x_k)` is an evaluation of a *mapped* node
> whose result must be written into the *leaf* slot for the next iteration.

Nothing in the workspace does the second today. `ForwardOutput.evaluations`
(`skoro-graph/src/compiler.rs:98`) is a `PointMap` keyed by mapped ids, while `points` is a
`ValueMap` keyed by leaf ids, and there is no id-remapping write between them. The EKF sketch in
§7 does it by hand. If a second algorithm needs it, promote it to a helper alongside
`retract_flat`, using `TangentLayout::same_shape` to check the pairing.

## 5. Jacobian assembly

Given an **input layout** `In` (columns, `D_in = In.total()`) and an **output layout** `Out`
(rows, `D_out = Out.total()`), produce `J ∈ [batch, D_out, D_in]`.

```rust
pub fn jacobian_jvp<B: Backend, C: ModelClientOps<B>>(
    client: &C,
    points: &ValueMap<B>,
    inputs: &TangentLayout,
    outputs: &TangentLayout,
) -> SkoroResult<Tensor<B, 3>>;

pub fn jacobian_vjp<B: Backend, C: ModelClientOps<B>>(
    client: &C,
    points: &ValueMap<B>,
    inputs: &TangentLayout,
    outputs: &TangentLayout,
) -> SkoroResult<Tensor<B, 3>>;
```

**JVP path** — one pass, `D_in` frames.

1. Seed each input block `j` with rows `o_j .. o_j + d_j` of `I_{D_in}`: a `[batch, d_j, D_in]`
   frame, i.e. `inputs.identity_frames(batch, device)`. Pass as `tangents` to
   `evaluate_ids_jvp` (`skoro-model/src/model.rs:274`).
2. Read output block `o` from `out.tangents` at `[batch, d_o, D_in]`.
3. `Tensor::cat` along dim 1 in `Out` order → `[batch, D_out, D_in]`. **No transpose.**

**VJP path** — one pass, `D_out` frames.

1. Seed each output block `o` with rows `p_o .. p_o + d_o` of `I_{D_out}`: a
   `[batch, d_o, D_out]` frame, i.e. `outputs.identity_frames(batch, device)`. Pass as
   `cotangents` to `evaluate_ids_vjp` (`skoro-model/src/model.rs:286`).
2. Read input block `j` from `out.cotangents` at `[batch, d_j, D_out]`.
3. `Tensor::cat` along dim 1 in `In` order → `[batch, D_in, D_out]`, then `.swap_dims(1, 2)`.

**The transpose asymmetry between the two paths is where this design fails silently.** Validate
both against a finite-difference check on an `SO3` or `SE3` model before building anything on
top of them (§8). Routing both seedings through the single `identity_frames` helper keeps them
from drifting apart.

Cost is the choice: `D_in` frames for JVP against `D_out` frames for VJP. A tall Jacobian
(`D_out ≪ D_in`, e.g. a low-dimensional measurement of a large state) favours VJP; a wide one
favours JVP.

### Structural constraint: columns are leaves, rows are mapped nodes

This constrains how a user must model a filter chain, so it belongs in the API docs, not just
here.

- `vjp_forward` collects cotangents **only** at `CompiledStep::Parameter` / `Sampler` steps
  (`skoro-graph/src/compiler.rs:352-369`).
- `jvp_forward` emits tangents into `out.tangents` **only** in the `CompiledStep::Operation`
  branch (`compiler.rs:253-260`). Leaf seeds are written into `tangent_seq` by `non_op_forward`
  but never surface in the output map.

Both paths therefore agree:

> **Jacobian columns must be `Parameter` / `Stochastic` leaves. Jacobian rows must be `Mapped` /
> `LogDensity` nodes.**

For an EKF this is natural — `x_k` is a leaf, `f(x_k)` and `h(x_k)` are mapped — but a state
chain built entirely from mapped nodes cannot be differentiated block-to-block.

The rule admits no exceptions. A correlated joint over several quantities is represented as a
likelihood factor over `Parameter` leaves (`graph-marginalization.md` §5), and those are ordinary
columns — so `jacobian_vjp` is available for jointly-distributed quantities exactly as it is for any
other leaf.

Two further caveats:

- `evaluate_ids_ord1` / `evaluate_ids_ord2` seed `ones([batch, dim, 1])`
  (`skoro-graph/src/graph.rs:1107-1118`). That is a sum-of-rows gradient, **not** a Jacobian. The
  flattened API must go through `evaluate_ids_jvp` / `evaluate_ids_vjp` directly.
- `check_tensor_frames` (`compiler.rs:642`) requires a constant frame count across all supplied
  quantities. Identity seeding satisfies this by construction; noted so future seeding schemes do
  not break it.

### A defect to fix along the way

`compiler.rs:363` panics with `"Point should have a cotangent vector."` when a continuous
`Parameter` / `Sampler` in the compiled subgraph receives no tape write. That is reachable from
the Jacobian API whenever a layout column reaches the targets only through a `forward_only` or
`constant_inputs` path. Fix it by defaulting to zeros — consistent with the missing-id rule in
§4 — or by returning a `SkoroError`. It should not stay a panic on a public path.

## 6. The dense linear algebra gap

Both consuming algorithms need a symmetric-positive-definite solve, and **the workspace does not
have one.**

What exists:

- `skoro-linalg/src/tril/invert.rs:15` — `TrilInvert`, forward substitution on the *packed*
  lower triangle. It round-trips through host memory (`into_data()` / `as_slice()` at
  `invert.rs:89-90`, `Tensor::from_data` at `invert.rs:92`), so it is a CPU loop and a sync point
  on wgpu.
- `skoro-linalg/src/matrix/utils.rs` — `left_broadcast_matmul`, `right_broadcast_matmul`, and
  `conjugate(C, Ω)` for `C Ω Cᵀ`. Note that `conjugate` is **rank-4** (frame-carrying) while an
  EKF's `H P Hᵀ` is rank-3; either add a rank-3 overload or call `Tensor::matmul` directly.

What `burn` 0.21 provides (verified in `burn-tensor-0.21.0/src/tensor/linalg/`):

- `linalg::lu::<B, D, D1>` — batched LU with partial pivoting, device-side, blocked
  (`linalg/lu.rs:71`). Its own docs flag it as "not fully optimized".
- `det`, `diag`, `trace`, `matvec`, `outer`, and the `vector_norm` family; `Tensor::tril` /
  `triu` masks.
- **No Cholesky. No triangular solve. No general solve.** (`solve_for_u_blocks`, `lu.rs:498`, is
  private.)

Proposed: a `crates/skoro-linalg/src/dense/` module over dense `[batch, n, n]` tensors —
distinct from the packed `Tril` manifold, and eager, since these are not graph operations.

```rust
pub fn cholesky<B: Backend>(spd: Tensor<B, 3>) -> Tensor<B, 3>;   // lower factor
pub fn solve_lower<B: Backend>(l: Tensor<B, 3>, b: Tensor<B, 3>) -> Tensor<B, 3>;
pub fn solve_upper<B: Backend>(u: Tensor<B, 3>, b: Tensor<B, 3>) -> Tensor<B, 3>;
pub fn spd_solve<B: Backend>(a: Tensor<B, 3>, b: Tensor<B, 3>) -> Tensor<B, 3>;
```

Sequencing:

1. **Correctness first.** Host-side implementations mirroring
   `invert_packed_lower_triangular` (`tril/invert.rs:18`) — same numerics, same testing style as
   `skoro-linalg/tests/common/tril.rs`. Mark them as a known perf cliff.
2. **Then on-device**, via the recursive block identities, which are all matmuls and therefore
   backend-native:
   - for `L = [[L₁₁, 0], [L₂₁, L₂₂]]`, `L⁻¹ = [[L₁₁⁻¹, 0], [−L₂₂⁻¹ L₂₁ L₁₁⁻¹, L₂₂⁻¹]]`;
   - blocked Cholesky recurses the same way.

   Depth is `O(log n)`, and this is the structure burn's own LU already uses.
3. `linalg::lu` is a usable interim for a general solve, but it does not exploit symmetry and
   leaves no factor to reuse. Prefer Cholesky once step 2 lands.

## 7. Worked consumers

Sketches, not specifications — enough to show the API is sufficient. Each becomes its own crate
(`skoro-ekf`, `skoro-gauss-newton`, and later `skoro-hmc`, `skoro-mala`,
`skoro-metropolis-hastings`), all depending on `skoro-model` and `skoro-linalg`.

Scope note: **these algorithms consume eager tensors and are not differentiable model
operations.** Use burn's batched `Tensor::matmul` on rank-3 directly, not the graph `Matmul`
op (`skoro-linalg/src/matmul/mod.rs:7`), which expects `[b, m*n]` flattened matrices and builds
tape entries.

### EKF

Layouts built once: `state` over the state leaves, `state_next` over the process-model outputs,
`meas` over the measurement node.

```
// once, at setup
assert!(state.same_shape(&state_next));

predict:  F = jacobian_jvp(client, points, &state, &state_next)   // [b, N, N]
          P = F P Fᵀ + W                                          // W = process noise
          fwd = client.evaluate_ids(points, state_next.ids())     // advance, not retract
          for (leaf, mapped) in state.ids().zip(state_next.ids()):
              points[leaf] = fwd.evaluations[mapped]

update:   H = jacobian_vjp(client, points, &state, &meas)         // [b, M, N], M ≪ N ⇒ VJP
          S = H P Hᵀ + R
          K = spd_solve(S, H P)ᵀ                                  // no explicit inverse
          δ = K · innovation                                      // [b, N]
          points = points.retract_flat(&state, δ)                 // correct, not advance
          P = (I − K H) P (I − K H)ᵀ + K R Kᵀ                     // Joseph form
```

Two things this shows:

- The frame-count asymmetry decides the pass. `F` is square (`D_in = D_out = N`), so either
  works; `H` has `M ≪ N`, so VJP is the cheaper one.
- Predict is the concrete case of "state advance replaces values" from §4 — the leaf-slot write
  has no helper today and is done by hand.

### Gauss-Newton

The objective is `½ Σ ‖rᵢ‖²`, where `rᵢ` is the whitened residual of factor `i`.
`Normal::log_density` (`skoro-normal/src/distribution.rs:168-178`) **already builds that node**:

```rust
let colored_tangent = self.mean.inverse_retract(x);
let white_tangent   = self.std_dev.invert().vector_product(&colored_tangent)?;
```

`white_tangent: Q<B, EuclideanD>` is exactly `r`, but it is discarded — only the scalar
log-density is returned. Exposing it is the largest reuse available here:

- Add `Normal::whitened_residual(&self, x: &Q<B, M>) -> SkoroResult<Q<B, EuclideanD>>` returning
  that node, and have `log_density` call it, so there is one code path.
- Introduce a `WhitenedResidual<B>` trait in `skoro-model/src/distribution.rs` so
  `skoro-gauss-newton` is distribution-generic rather than `Normal`-specific.

The iteration is then:

```
residuals = TangentLayout over the whitened-residual nodes     // rows
free      = TangentLayout over the free leaves                 // columns

J  = jacobian_vjp(client, points, &free, &residuals)           // [b, M, N]
δ  = spd_solve(JᵀJ + λI, −Jᵀr)                                 // [b, N]
points = points.retract_flat(&free, δ)
```

`λ = 0` is Gauss-Newton; `λ > 0` is Levenberg-Marquardt, which the same code supports for free.

## 8. Verification

1. **Round-trip.** `unflatten(flatten(map)) == map` for a mixed layout containing `Real`,
   `EuclideanD`, `Tril`, `SO3`, and `SE3`. Confirms the dimension-not-storage rule of §2.
2. **JVP/VJP agreement.** `jacobian_jvp` and `jacobian_vjp` produce identical
   `[b, D_out, D_in]` tensors on the same model. This is the test that catches the transpose
   asymmetry.
3. **Finite differences.** Both paths against a central difference taken *through the
   retraction* (`retract_flat` with `±ε eᵢ`), on an `SE3` model so the 6-vs-12 split is
   exercised.
4. **Dense linalg.** `spd_solve(A, A) ≈ I`; `cholesky(L Lᵀ) ≈ L`; agreement with `TrilInvert` on
   a packed lower-triangular input converted to dense.
5. **EKF end-to-end.** A linear-Gaussian model, where the EKF must reproduce the exact Kalman
   filter, checked against the closed-form posterior.
6. **Gauss-Newton end-to-end.** A linear-Gaussian least-squares problem, which must converge in
   one iteration to the analytic solution.

Follow the existing test layout — per-backend entry points over a shared `common/` module, as in
`skoro-linalg/tests/{ndarray_f32,ndarray_f64,wgpu_f32}.rs` and `skoro-so3/tests/common/`. Run
with `cargo test --workspace`; the wgpu suites additionally exercise the device path and will
surface the `TrilInvert` host-roundtrip sync points called out in §6.
