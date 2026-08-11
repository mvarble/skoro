# Phase 1 — Interfaces

A header-file document. Every signature that phase 1 introduces or changes, with prose instead of
docstrings, so the shapes can be reviewed before a single `.rs` file is touched.

Bodies are `todo!()` throughout. There are no algorithms here, no loops, no numerics — only types,
generics and bounds. Where a signature comes from a design document, the document and section are
named immediately above it.

---

## Part 0 — What phase 1 is

**Deliverable: a working Levenberg-Marquardt optimizer**, run on a linear-Gaussian least-squares
problem and checked against the analytic solution.

That target is chosen because it is the shortest path that exercises every foundation the other
documents rest on — the tangent-space layout, Jacobian assembly, the host dense-linear-algebra
boundary, and the structural preconditions — while remaining **batch and stateless**. It needs no
covariance carried between steps, no per-tick model building, and no graph surgery.

Deliberately excluded, and why:

| excluded | why |
|---|---|
| graph teardown, `Graph::remove` fixes | only the sequential loop deletes nodes; LM builds once |
| the tick layer (`TimeSequence`, `Chain`, `Transition`) | LM has no notion of time |
| `advance`, `retract_flat` pairing across ticks | no state advance in a batch optimizer |
| marginalization, the graveyard | nothing retires |
| `skoro-ekf` | needs all of the above |

## Status: the groundwork is done

**Part A — the changes to structures that already existed — has been applied and removed from this
document.** The workspace builds and all tests pass (121 then; 140 after B.1). What landed:

| | | |
|---|---|---|
| `skoro-core/error.rs` | 6 new `SkoroError` variants + `Display` arms | 🟢 |
| `skoro-core/id.rs` | `Id: PartialOrd + Ord` | 🟢 |
| `skoro-graph/compiler.rs` | VJP cotangent panic → zeros; dead `ForwardOutput.observations` removed | 🔴 |
| `skoro-graph/point_map.rs` | `inverse_retract` takes `&PointMap<B>` | 🔴 |
| `skoro-graph/graph.rs` | 5 `pub` fields → read accessors; `add_mapped -> SkoroResult<Id>` | 🔴 |
| `skoro-model/hash_map_builders.rs` | `with_quantity` on both builders | 🟢 |
| `skoro-model/q.rs` | `client`/`id` private; `Q::mark_likelihood` and `get_server` gone | 🔴 |
| `skoro-model/client_ops.rs` | `ModelBuilderOps`; growth methods moved off `ModelClient` | 🔴 |
| `skoro-model` + `skoro-normal` | `Distribution::log_density -> SkoroResult`; `Normal::whitened_residual` | 🔴 |
| workspace-wide | **the whole node-building surface is fallible** — see below | 🔴 |

### Node building is fallible end to end

Beyond the original Part A: `register_map1..11`, `ModelClient::add_mapped`, every `std::ops`
operator on `Q`, and all 43 `*Ops` extension methods now return `SkoroResult<Q<..>>`. **No panic
remains anywhere on the path that builds a graph node.** The single surviving `expect` in
`skoro-model` is in `cleanup()` (`model.rs:215`), which removes only ids the model itself queued —
a genuine internal invariant rather than a user-reachable one.

Chaining is preserved by implementing the operators for `SkoroResult<Q>` on the left as well, so a
left-associative chain costs one `?` at the end rather than one per step:

```rust
let z = (a + b + c)?;            // (a + b) is a Result; `Result + Q` is implemented
let d = (x.norm_squared()? * -0.5)?;
```

Two shapes are **not** expressible, both blocked by the orphan rule — neither operand names a local
type:

| shape | example | write instead |
|---|---|---|
| `Result op Result` | `a * b + c * d` | `(a * b)? + (c * d)?` |
| `Result op literal` | `a * b + 1.0` | `(a * b)? + 1.0` |

`Q op Result` (`a + b * c`) is also absent, and that one is a deliberate trade rather than a hard
limit: it would overlap the blanket literal impl `Add<E: ToElement> for Q`, since nothing tells the
compiler a `SkoroResult` is not an element. Keeping the blanket impl means bare float literals still
infer; dropping it would have forced an `f32`/`f64` suffix on every literal in every model. Write
`a + (b * c)?`.

**One Part A item was not applied and moves to Part B**, because it turned out to depend on work this
document schedules later:

> **`PointMap::retract_flat` / `retract_flat_mut` need `TangentLayout`**, which is B.1. The
> signature is `retract_flat(&self, layout: &TangentLayout, delta: Tensor<B, 2>)`, so it cannot be
> written before the layout type exists. It belongs with B.1 and is listed there. This was an
> ordering mistake in the original Part A, not a change of plan.

The remaining parts are in dependency order.

## On breaking changes

The library has no external users, so a breaking change costs a migration and nothing else. That is
the cheapest this will ever be, and several interfaces were worse than they needed to be only
because fixing them would move a signature.

**New interfaces below are labelled — 🟢 additive, 🔴 breaking — with the migration cost stated.**
Where an interface would be meaningfully better if it were allowed to break, that is called out even
when the change is not taken, so the choice is visible rather than silent.

The bar applied: a breaking change is taken when it removes a panic from a public path, closes a
capability leak, or fixes an asymmetry that will otherwise be copied. It is *not* taken merely for
tidiness, because each one still costs review.

---

# Part B — New code, by crate

## B.1 `skoro-graph` — new module `tangent_layout.rs` — ✅ **implemented**

> **Done.** `crates/skoro-graph/src/tangent_layout.rs` plus `PointMap::retract_flat{,_mut}`, with
> 13 tests in `skoro-graph/tests/tangent_layout_tests.rs` and 2 backend-generic tests in
> `skoro-se3/tests/common/mod.rs`. Workspace: 140 tests passing, clippy clean.
>
> One supporting change: `Graph::get_manifold` was private and is now `pub`, which is what lets
> `from_ids` size a block. It already resolved [`Node::LogDensity`] to `Real`, so a layout can name
> density nodes as rows — exactly what a Jacobian's row layout needs.
>
> The interface below is as-built; the only deviation is that `flatten_vectors`/`flatten_frames`
> validate supplied block shapes (`BatchSizeMismatch` / `InconsistentFrameCount`) rather than
> trusting them, since a wrong-shaped block would otherwise fail inside `Tensor::cat` with no id
> to name.

**Existing crate.** The layout lives here, below every algorithm crate, because an EKF, a
least-squares optimizer, HMC and MALA all need the same object. It needs only
`DynQuantityType::as_manifold()` (`skoro-graph/src/quantity.rs:184`) and `Graph` for lookups, both
local.

*From `flattened-tangent-spaces.md` §3.*

```rust
// crates/skoro-graph/src/tangent_layout.rs

/// A contiguous ordering of the tangent spaces of a set of quantities.
///
/// Laid out by **manifold dimension**, never by storage size: `SO3` occupies 3 slots and
/// `SE3` occupies 6, not 9 and 12. Every shape downstream depends on this.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TangentLayout {
    blocks: Vec<TangentBlock>,
    index: HashMap<Id, usize>,
    total: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TangentBlock {
    pub id: Id,
    pub offset: usize,
    pub dimension: usize,
}
```

Ordering is caller-specified by design: a block structure the caller chose is one they can reason
about when reading a covariance or a Jacobian.

```rust
impl TangentLayout {
    /// Preserves the caller's ordering.
    ///
    /// Errors `NotContinuous` for a discrete id, `NodeNotFound` for an unknown one, and
    /// `DuplicateLayoutId` if an id appears twice.
    pub fn from_ids<B: Backend>(graph: &Graph<B>, ids: &[Id]) -> SkoroResult<Self> { todo!() }

    /// The same ids in ascending `Id` order, for callers who do not care.
    pub fn from_ids_sorted<B: Backend>(graph: &Graph<B>, ids: &[Id]) -> SkoroResult<Self> { todo!() }

    pub fn total(&self) -> usize { todo!() }
    pub fn blocks(&self) -> &[TangentBlock] { todo!() }
    pub fn block(&self, id: Id) -> Option<TangentBlock> { todo!() }
    pub fn contains(&self, id: Id) -> bool { todo!() }
    pub fn ids(&self) -> impl Iterator<Item = Id> + '_ { todo!() }

    /// Block dimensions equal, in order — two layouts describing the same tangent space
    /// over different quantities. Phase 1 does not use this; it is the rule a sequential
    /// estimator pairs `state` and `state_next` with.
    pub fn same_shape(&self, other: &Self) -> bool { todo!() }
}
```

A layout holds no tensors and carries no backend parameter, so it is cheap to clone and store. It is
**not** revalidated against a graph after construction — reusing one after `Graph::remove` is a
caller error.

*From `flattened-tangent-spaces.md` §4.*

```rust
impl TangentLayout {
    /// rank-2: `[batch, d_i]` blocks -> `[batch, N]`.
    ///
    /// Ids missing from `map` are filled with zeros rather than erroring: a VJP
    /// legitimately returns no cotangent for a leaf that does not reach the target.
    pub fn flatten_vectors<B: Backend>(
        &self,
        map: &HashMap<Id, TangentVector<B>>,
        batch: usize,
        device: &B::Device,
    ) -> SkoroResult<Tensor<B, 2>> { todo!() }

    pub fn unflatten_vector<B: Backend>(
        &self,
        flat: Tensor<B, 2>,
    ) -> SkoroResult<HashMap<Id, TangentVector<B>>> { todo!() }

    /// rank-3: `[batch, d_i, f]` blocks -> `[batch, N, f]`.
    pub fn flatten_frames<B: Backend>(
        &self,
        map: &HashMap<Id, CotangentFrame<B>>,
        batch: usize,
        frames: usize,
        device: &B::Device,
    ) -> SkoroResult<Tensor<B, 3>> { todo!() }

    pub fn unflatten_frame<B: Backend>(
        &self,
        flat: Tensor<B, 3>,
    ) -> SkoroResult<HashMap<Id, CotangentFrame<B>>> { todo!() }

    /// Per-id blocks of the `N x N` identity, where `N = self.total()`:
    /// block `j` is rows `o_j .. o_j + d_j`, shape `[batch, d_j, N]`.
    ///
    /// Both Jacobian paths seed through this one helper, which is what keeps their
    /// index conventions from drifting apart.
    pub fn identity_frames<B: Backend>(
        &self,
        batch: usize,
        device: &B::Device,
    ) -> HashMap<Id, Tensor<B, 3>> { todo!() }
}
```

Flattening is a **single** `Tensor::cat` over the ordered block list along dim 1, not a pairwise fold.
Unflattening is `narrow(1, offset, dimension)` per block. Every entry point validates
`flat.dims()[1] == self.total()` and returns `LayoutDimensionMismatch` otherwise.

### `PointMap::retract_flat` lands with this module

Carried over from the original Part A, where it was scheduled too early: it takes a `&TangentLayout`
and so cannot exist before the type above.

*From `flattened-tangent-spaces.md` §4.* 🟢 additive.

```rust
// crates/skoro-graph/src/point_map.rs — new methods on `impl<B: Backend> PointMap<B>`

impl<B: Backend> PointMap<B> {
    /// Retract only the quantities named by `layout`, leaving every other point untouched.
    ///
    /// This is deliberately unlike the existing `retract_mut`, which requires a tangent
    /// for *every* point in the map and errors `MissingPoint` otherwise. An optimizer's
    /// `PointMap` also holds constants and nuisance quantities that are not being
    /// stepped, so partial retraction is the operation it actually needs.
    ///
    /// `delta` is `[batch, layout.total()]`; each block is unflattened and applied
    /// through its own manifold's retraction. Errors `LayoutDimensionMismatch` if the
    /// widths disagree, `MissingPoint` for a layout id absent from the map.
    pub fn retract_flat(
        &self,
        layout: &TangentLayout,
        delta: Tensor<B, 2>,
    ) -> SkoroResult<Self> { todo!() }

    pub fn retract_flat_mut(
        &mut self,
        layout: &TangentLayout,
        delta: Tensor<B, 2>,
    ) -> SkoroResult<()> { todo!() }
}
```

The manifold for each block is reached as the existing private helper does (`manifold_point`,
`point_map.rs:126`), via `quantity_type.as_manifold()` to a `&dyn DynManifold<B>`. `DynManifold` is
sealed with a blanket impl (`skoro-riemann/src/base.rs:36-51`), so it is consumed, never implemented.

## B.2 `skoro-model` — new module `jacobian.rs`

**Existing crate.** These take a client, so they sit above `skoro-graph` where `TangentLayout` lives.
`skoro-model` re-exports `TangentLayout` alongside `hash_map_builders`.

*From `flattened-tangent-spaces.md` §5.*

```rust
// crates/skoro-model/src/jacobian.rs

/// `J ∈ [batch, Out.total(), In.total()]` by one forward pass with `In.total()` frames.
pub fn jacobian_jvp<B: Backend, C: ModelClientOps<B>>(
    client: &C,
    points: &ValueMap<B>,
    inputs: &TangentLayout,
    outputs: &TangentLayout,
) -> SkoroResult<Tensor<B, 3>> { todo!() }

/// The same `J`, by one reverse pass with `Out.total()` frames.
pub fn jacobian_vjp<B: Backend, C: ModelClientOps<B>>(
    client: &C,
    points: &ValueMap<B>,
    inputs: &TangentLayout,
    outputs: &TangentLayout,
) -> SkoroResult<Tensor<B, 3>> { todo!() }
```

Both return the same `[batch, D_out, D_in]` tensor; they differ only in cost. The JVP path costs
`D_in` frames, the VJP path `D_out`. A tall Jacobian (`D_out ≪ D_in` — a low-dimensional residual
over a large state) favours VJP, which is the case a least-squares problem is usually in.

> **The transpose asymmetry is where this pair fails silently.** The JVP path concatenates output
> blocks and is done. The VJP path concatenates *input* blocks, producing `[batch, D_in, D_out]`, and
> must then `.swap_dims(1, 2)`. Getting it wrong yields a well-shaped, wrong `J` whenever the matrix
> happens to be square. The two are checked against each other in Part D, item 2.

**Structural constraint on callers, which belongs in these functions' documentation:** `vjp_forward`
collects cotangents only at `CompiledStep::Parameter` / `Sampler` (`compiler.rs:352-369`), and
`jvp_forward` emits tangents only in the `Operation` branch (`compiler.rs:253-260`). So **columns must
be leaves and rows must be mapped nodes.** A model whose residuals are leaves, or whose free variables
are mapped, cannot be differentiated block-to-block.

## B.3 `skoro-model` — new module `structure.rs`

**Existing crate.** The factor-local bound and the certificate that accumulates it.

*From `model-structure-bounds.md` §4.*

```rust
// crates/skoro-model/src/structure.rs

/// A continuous distribution whose negative log-density is `½‖r(x)‖² + n(θ)`.
///
/// Bounded on `Distribution<B, Continuous>` because a discrete factor has no tangent
/// space and no residual vector exists for it at all.
pub trait WhitenedResidual<B: Backend>: Distribution<B, Continuous> {
    /// The residual node `r(x)`. The sum of squares of these is the objective.
    fn whitened_residual(&self, x: &Q<B, Self::Qty>) -> SkoroResult<Q<B, EuclideanD>>;

    /// The subset of `parent_ids()` on which the normalizer `n(θ)` depends.
    ///
    /// `Distribution::parent_ids` is positional and untyped, so there is no way to ask
    /// "which parent is the covariance". This asks the narrower question a least-squares
    /// consumer actually has. For `Normal` the answer is the covariance factor alone.
    fn normalizer_parents(&self) -> &[Id];
}
```

*From `model-structure-bounds.md` §6, with the amendment in [Part C](#part-c--what-writing-the-interfaces-changed).*

```rust
/// What a builder recorded about one factor, at the moment it built it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FactorRecord {
    pub density: Id,
    pub residual: Id,
    pub normalizer_parents: Vec<Id>,
}

/// The accumulated evidence that every factor in a model is least-squares shaped.
///
/// Unforgeable not because the type is sealed, but because entries can only be added
/// through `record`, which demands a `WhitenedResidual` distribution. A hand-built
/// certificate is empty, and there is no way to put a fabricated id into one.
#[derive(Debug, Clone, Default)]
pub struct ResidualCertificate {
    factors: Vec<FactorRecord>,
}

impl ResidualCertificate {
    pub fn new() -> Self { todo!() }

    /// Record a factor. The `D` bound is the whole guarantee.
    pub fn record<B, K, D>(
        &mut self,
        distr: &D,
        density: Id,
        residual: &Q<B, EuclideanD>,
    ) -> SkoroResult<()>
    where
        B: Backend,
        D: WhitenedResidual<B>,
    { todo!() }

    pub fn factors(&self) -> &[FactorRecord] { todo!() }
    pub fn residual_ids(&self) -> impl Iterator<Item = Id> + '_ { todo!() }
}
```

```rust
/// A model whose negative log-density is a sum of squared residuals.
///
/// The supertrait is deliberate: `ModelClientOps` supplies the whole evaluation surface
/// an optimizer needs — `evaluate_ids`, `evaluate_ids_vjp` and the rest — while
/// containing no way to grow the model. So an algorithm bounded on `NlsStructure` can
/// evaluate and differentiate without ever holding something that could invalidate the
/// certificate it was given.
pub trait NlsStructure<B: Backend>: ModelClientOps<B> {
    fn certificate(&self) -> &ResidualCertificate;

    /// The rows of `J`.
    fn residual_layout(&self) -> SkoroResult<TangentLayout> { todo!() }
}
```

```rust
/// The least-squares builder. Accepts any `WhitenedResidual` factor.
pub struct NlsModelClient<B: Backend> {
    client: ModelClient<B>,
    certificate: ResidualCertificate,
}

impl<B: Backend> NlsModelClient<B> {
    pub fn new(device: B::Device) -> Self { todo!() }

    /// Add a factor and record it. The bound is what a plain `ModelClient` does not enforce.
    pub fn add_residual_quantity<K, D>(&mut self, distr: D) -> SkoroResult<Q<B, D::Qty>>
    where
        K: Kind<B>,
        D: WhitenedResidual<B>,
    { todo!() }
}

impl<B: Backend> ModelClientOps<B> for NlsModelClient<B> { /* delegating; todo!() */ }
impl<B: Backend> NlsStructure<B> for NlsModelClient<B> { /* todo!() */ }
```

Note there is no `into_client()`. `GaussianModelClient` has one (`facade.rs:162`) and it dissolves
the guarantee; the new builder does not repeat that.

## B.4 `skoro-dense` — **new leaf crate**

Host-resident, `f64`, batch 1. Depends on `skoro-backend` and `skoro-core` and **nothing else in the
workspace** — in particular not `skoro-graph` or `skoro-model`, so nothing that merely uses `Normal`
or `Tril` pays for `faer`.

*From `estimator-linear-algebra.md` §7.*

```toml
# crates/skoro-dense/Cargo.toml
[package]
name = "skoro-dense"
description = "Host-side dense linear algebra for Skoro estimators."
authors.workspace = true
edition.workspace = true
version.workspace = true

[dependencies]
burn.workspace = true
faer.workspace = true

skoro-backend.workspace = true
skoro-core.workspace = true

[dev-dependencies]
approx.workspace = true
burn-ndarray.workspace = true
```

Two workspace-root edits are needed, since `members = ["crates/*"]` auto-discovers the directory but
nothing can depend on it until it is named:

```toml
# Cargo.toml, [workspace.dependencies]
faer = "0.23"
skoro-dense = { path = "./crates/skoro-dense" }
skoro-levenberg-marquardt = { path = "./crates/skoro-levenberg-marquardt" }
```

```rust
// crates/skoro-dense/src/lib.rs
//! # skoro-dense
//!
//! Host-side dense linear algebra at batch 1, in `f64`, for the deterministic
//! estimators. Factorizations ([`cholesky`], [`spd_solve`]) and least squares
//! ([`lstsq`]) over [`Dense`], with conversions to and from backend tensors.

pub mod dense;
pub mod factor;
```

```rust
// crates/skoro-dense/src/dense.rs

/// A single-batch dense matrix, host-resident, always `f64`.
#[derive(Debug, Clone, PartialEq)]
pub struct Dense {
    rows: usize,
    cols: usize,
    data: Vec<f64>,
}

impl Dense {
    pub fn zeros(rows: usize, cols: usize) -> Self { todo!() }
    pub fn identity(n: usize) -> Self { todo!() }
    pub fn rows(&self) -> usize { todo!() }
    pub fn cols(&self) -> usize { todo!() }

    /// From `[1, m, n]`. Errors `BatchNotSingleton` on any other leading dimension
    /// rather than silently taking slice 0.
    pub fn from_tensor<B: Backend>(t: Tensor<B, 3>) -> SkoroResult<Self> { todo!() }

    pub fn to_tensor<B: Backend>(&self, device: &B::Device) -> Tensor<B, 3> { todo!() }

    /// From `[1, n]` — a column.
    ///
    /// **Tangent and residual vectors only**: a `δ`, an `r`, an innovation. A point must
    /// never come through here. `Point<B>`, `TangentVector<B>` and `CotangentVector<B>`
    /// are all `Tensor<B, 2>` (`skoro-backend/src/lib.rs:77-95`), so the compiler cannot
    /// enforce this; `n` is a manifold dimension, not a storage size, and Part D item 1
    /// is what holds the convention up.
    pub fn from_vector<B: Backend>(t: Tensor<B, 2>) -> SkoroResult<Self> { todo!() }

    pub fn to_vector<B: Backend>(&self, device: &B::Device) -> Tensor<B, 2> { todo!() }

    /// Stack `[self; other]` — the row augmentation a damped least-squares step needs.
    pub fn vstack(&self, other: &Self) -> SkoroResult<Self> { todo!() }
}
```

Conversions go through `TensorData::convert::<f64>()` inbound and
`TensorData::convert::<B::FloatElem>()` outbound, the mechanism `TrilInvert` already uses
(`skoro-linalg/src/tril/invert.rs:89-92`).

```rust
// crates/skoro-dense/src/factor.rs

/// Lower Cholesky factor, or `NotPositiveDefinite`.
pub fn cholesky(spd: &Dense) -> SkoroResult<Dense> { todo!() }

pub fn solve_lower(l: &Dense, b: &Dense) -> SkoroResult<Dense> { todo!() }
pub fn solve_upper(u: &Dense, b: &Dense) -> SkoroResult<Dense> { todo!() }
pub fn spd_solve(a: &Dense, b: &Dense) -> SkoroResult<Dense> { todo!() }

/// Least squares by column-pivoted QR: `min ‖J δ − r‖`, without forming `JᵀJ`.
///
/// This gives the **undamped** step. Damping is the caller's job and costs nothing
/// extra: `(JᵀJ + λI) δ = −Jᵀr` is exactly `min ‖[J; √λ I] δ − [−r; 0]‖`, so an
/// iteration augments and calls this. The QR path is therefore available at every
/// damping value, at condition `κ` rather than `κ²`.
pub fn lstsq(j: &Dense, r: &Dense) -> SkoroResult<Dense> { todo!() }
```

All five return `SkoroResult`. That is the design argument made structural: a matrix that has stopped
being positive definite is a condition the caller must be able to observe, not a `NaN` that surfaces
several thousand iterations later.

> Four of these five could be reimplemented on `Tensor<B, 3>` behind the same names if a batched
> consumer ever appears. `lstsq` could not — `burn` 0.21 has no QR. An algorithm written against
> `lstsq` is committed to the host path, deliberately.

## B.5 `skoro-normal` — additions

**Existing crate.** The Gaussian refinement, and the certificate on the facade that already exists.

*From `model-structure-bounds.md` §4.*

```rust
// crates/skoro-normal/src/structure.rs

/// Additionally: `r(x) = L⁻¹ (x ⊖ μ)`, with the mean and whitening separately reachable.
///
/// This is the entire difference between "least-squares shaped" and "hierarchical
/// Gaussian". Separate access is what a filter needs, since `H = ∂μ/∂x` and `R = L Lᵀ`
/// are different objects. The *nonlinearity* of the mean needs no expression here:
/// `mean()` returns a graph node, which may be the output of arbitrarily many operations.
pub trait GaussianFactor<B: Backend>: WhitenedResidual<B> {
    fn mean(&self) -> &Q<B, Self::Qty>;
    fn whitening(&self) -> &Q<B, Tril>;
}
```

`Normal<B, M>` implements both nearly for free — it already has `mean()` (`distribution.rs:131`) and
`std_dev()` (`:135`):

```rust
impl<B: Backend, M: ContinuousQuantity<B>> WhitenedResidual<B> for Normal<B, M> { /* todo!() */ }
impl<B: Backend, M: ContinuousQuantity<B>> GaussianFactor<B> for Normal<B, M> { /* todo!() */ }
```

```rust
/// A model in which every conditional is Gaussian.
pub trait GaussianStructure<B: Backend>: NlsStructure<B> {
    fn blocks(&self) -> &[GaussianBlock];
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GaussianBlock {
    pub density: Id,
    pub residual: Id,
    pub mean: Id,
    pub whitening: Id,
}
```

`GaussianStructure` is declared in phase 1 because it is what makes the hierarchy meaningful — a
Gaussian model is accepted by a least-squares algorithm for free — but **phase 1 has no consumer for
`blocks()`**. Its consumer is the EKF.

*From `model-structure-bounds.md` §6.* The facade gains the cache:

```rust
// crates/skoro-normal/src/facade.rs:139
pub struct GaussianModelClient<B: Backend> {
    client: ModelClient<B>,
    certificate: ResidualCertificate,
    blocks: Vec<GaussianBlock>,
}

impl<B: Backend> NlsStructure<B> for GaussianModelClient<B> { /* todo!() */ }
impl<B: Backend> GaussianStructure<B> for GaussianModelClient<B> { /* todo!() */ }
```

`add_gaussian_quantity` (`facade.rs:82`) keeps its signature and gains the recording step. Two doc
changes ride along, and they are the point of the naming convention:

- `facade.rs:11-15` and `:196-199` currently claim `GaussianModelClient` is "the structural guarantee
  that every conditional in this model is Gaussian, used by inference algorithms". That is a
  *structure* claim written next to an *`Ops`* trait. It moves to `GaussianStructure`, and the prose
  is rewritten to claim capability only.
- `into_client()` (`facade.rs:162`) is documented as **dissolving the certificate**: a caller who
  takes the raw client can add factors the cache will never see.

> **`GaussianElem`** (`facade.rs:46-65`) bundles the six `B::FloatElem` operator bounds and is
> currently used nowhere, with every impl restating them longhand. The new traits need the same
> bounds — use it.

## B.6 `skoro-levenberg-marquardt` — **new crate**

**The consumer.** Everything above exists to make this small.

```toml
# crates/skoro-levenberg-marquardt/Cargo.toml
[package]
name = "skoro-levenberg-marquardt"
description = "Levenberg-Marquardt optimization over Skoro models."
authors.workspace = true
edition.workspace = true
version.workspace = true

[dependencies]
burn.workspace = true

skoro-backend.workspace = true
skoro-core.workspace = true
skoro-graph.workspace = true
skoro-model.workspace = true
skoro-dense.workspace = true

[dev-dependencies]
approx.workspace = true
skoro-normal.workspace = true
skoro-model = { workspace = true, features = ["testing"] }
burn-ndarray.workspace = true
```

*From `model-structure-bounds.md` §5.* The precondition, first, because it gates everything else:

```rust
// crates/skoro-levenberg-marquardt/src/precondition.rs

/// Check that the sum-of-squares objective really is this model's negative log-density.
///
/// `−log p = ½Σ‖rᵢ‖² + Σnᵢ(θ)`. Minimizing the first term alone is valid exactly when
/// the normalizers do not vary with the variables being optimized. The condition is
/// **not** "covariances must be constant" — a covariance that is a free `Parameter` but
/// is absent from `free` is fine, and that is an ordinary modelling case.
///
/// `O(#factors × arity)`, and it inspects no operations.
pub fn check_normalizers<B, M>(model: &M, free: &TangentLayout) -> SkoroResult<()>
where
    B: Backend,
    M: NlsStructure<B>,
{ todo!() }
```

It returns `NormalizerNotConstant { density, parent }`, naming both the offending factor and the
parent that leaked into the free set — the two things needed to fix the model.

*From `flattened-tangent-spaces.md` §7 and `estimator-linear-algebra.md` §3.*

```rust
// crates/skoro-levenberg-marquardt/src/lib.rs

#[derive(Debug, Clone, PartialEq)]
pub struct LmOptions {
    /// Initial damping. `0.0` is an undamped step, exact on a linear problem.
    pub lambda: f64,
    pub lambda_increase: f64,
    pub lambda_decrease: f64,
    pub max_iterations: usize,
    pub step_tolerance: f64,
    pub cost_tolerance: f64,
}

impl Default for LmOptions { /* todo!() */ }

#[derive(Debug, Clone, PartialEq)]
pub struct LmReport {
    pub iterations: usize,
    pub initial_cost: f64,
    pub final_cost: f64,
    pub final_lambda: f64,
    pub termination: Termination,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Termination {
    CostTolerance,
    StepTolerance,
    MaxIterations,
}
```

```rust
/// Minimize `½Σ‖rᵢ(x)‖²` over the quantities named by `free`.
///
/// `points` is updated in place through the manifold retraction, never by vector
/// addition — a step is `retract(x, δ)`, which is why `free` is a layout of manifold
/// dimensions and not a flat parameter count.
///
/// Calls `check_normalizers` before the first iteration and fails there rather than
/// silently optimizing the wrong objective.
pub fn solve<B, M>(
    model: &M,
    points: &mut PointMap<B>,
    free: &TangentLayout,
    options: &LmOptions,
) -> SkoroResult<LmReport>
where
    B: Backend,
    M: NlsStructure<B>,
{ todo!() }

/// One step, for callers driving their own loop.
pub fn step<B, M>(
    model: &M,
    points: &PointMap<B>,
    free: &TangentLayout,
    residuals: &TangentLayout,
    lambda: f64,
) -> SkoroResult<Tensor<B, 2>>
where
    B: Backend,
    M: NlsStructure<B>,
{ todo!() }
```

The body of `step` is four calls and no mathematics of its own, which is the measure of whether the
foundations are right: `jacobian_vjp` for `J`, `evaluate_ids` for `r`, `Dense::vstack` to augment
with `√λ I`, `lstsq` for `δ`.

> **`M: NlsStructure<B>` carries the evaluation surface with it**, because `NlsStructure` requires
> `ModelClientOps`. So `solve` needs no second client parameter, and it is holding nothing that could
> add a factor behind the certificate's back.

---

# Part C — What writing the interfaces changed

The exercise earned its keep here. One design in `model-structure-bounds.md` does not survive contact
with the crate graph.

### The sealed-trait design cannot work across crates — ✅ **folded into the docs**

`model-structure-bounds.md` used to specify `NlsStructure` as a **sealed** trait, on the argument
that a trait anyone can implement cannot carry an invariant. The argument was right and the mechanism
was not: sealing works by a private `Sealed` supertrait, so only the defining crate can implement the
trait — and `NlsStructure` must live in `skoro-model` while `GaussianModelClient` lives in
`skoro-normal`. Sealing would have forbidden exactly the implementation the trait exists to admit.

The fix moves the guarantee from the trait to `ResidualCertificate`'s constructor: `record` demands
`D: WhitenedResidual<B>`, so a hand-rolled implementor can only produce an empty or honest
certificate. The trait is forgeable; the evidence is not.

> **This is now recorded in `model-structure-bounds.md` §2**, with the naming convention in §3 and the
> verification in §9 rebuilt around it. That document is the durable record; this section is a
> pointer, kept only because the finding originated here.

### Two smaller notes

- **`GaussianStructure` ships without a consumer.** Phase 1 has no use for `blocks()`; only the EKF
  does. It is declared now because the refinement `GaussianStructure: NlsStructure` is the thing
  being validated, and declaring it later would mean revisiting the facade a second time.
- **Encapsulating `Graph` turned out to be the load-bearing change, not `Q.client`.** The interfaces
  in B.3 and B.5 assert that a certificate cannot be bypassed. Writing them made clear that `Graph`'s
  five `pub` fields plus a `pub` `ModelServer.graph` made every such assertion provisional — a
  `&mut Graph` reached through `get_server` could append to `log_likelihoods` directly. Privatizing
  `Q.client` closes the route from a *quantity*; encapsulating `Graph` closes the route from a
  *client*, and without it the rest is a convention. This only became visible once the certificate
  had a written signature to defend. Both are now applied.

---

# Part D — Verification

Per-backend entry points over a shared `common/`, following the existing layout exactly. Each entry
point is three lines and the bodies are generic over `B`:

```rust
// crates/skoro-levenberg-marquardt/tests/ndarray_f64.rs — complete file
mod common;
type B = skoro_backend::NdArrayBackend<f64, i32>;
tests!(crate::B, 1.0e-6, 1.0e-5);
```

Heed the workspace doctrine (`.claude/skills/skoro-diff-operation/references/testing.md`):
hand-computed constants generated from the implementation only lock in whatever the implementation
already did. Prefer finite differences and closed forms.

Items 2 and 5 are the gates — the two that fail silently otherwise. Write them first.

| # | test | from |
|---|---|---|
| 1 | **Layout round-trip.** `unflatten(flatten(map)) == map` for a mixed layout of `Real`, `EuclideanD`, `Tril`, `SO3`, `SE3`. Confirms dimension-not-storage: the `SO3` block is 3 wide, not 9. | `flattened-tangent-spaces.md` §8.1 |
| 2 | **JVP/VJP agreement.** `jacobian_jvp` and `jacobian_vjp` produce identical `[b, D_out, D_in]` tensors on the same model. **This is the test that catches the transpose asymmetry**, and it must include a non-square case or it proves nothing. | §8.2 |
| 3 | **Finite differences.** Both Jacobian paths against a central difference taken *through the retraction* (`retract_flat` with `±ε eᵢ`), on an `SE3` model so the 6-versus-12 split is exercised. | §8.3 |
| 4 | **Dense linalg.** `spd_solve(A, A) ≈ I`; `cholesky(L Lᵀ) ≈ L`; agreement with `TrilInvert` on a packed lower-triangular input converted to dense; `lstsq` on an ill-conditioned `J` beating `spd_solve(JᵀJ, ..)`, which is the QR argument made checkable. Plus the batch-1 guard: a `[2, m, n]` tensor returns `BatchNotSingleton`. | `estimator-linear-algebra.md` §9 |
| 5 | **The normalizer precondition, both directions.** A model with a free `Tril` covariance **in** the `free` layout is rejected with `NormalizerNotConstant`; the *same model* with that covariance outside `free` is accepted. A naive "covariances must be constant" implementation passes the first half and fails the second. | `model-structure-bounds.md` §9.2 |
| 6 | **Certificate honesty.** A `ResidualCertificate` built by hand is empty, and `record` cannot be called without a `WhitenedResidual` distribution — a `trybuild` compile-fail test. | Part C |
| 7 | **Residual and density agree.** `‖whitened_residual(x)‖²` reconstructs `Normal::log_density(x)` up to `constant + log_det`, on `Real`, `EuclideanD` and `SE3` means. | `model-structure-bounds.md` §9.1 |
| 8 | **LM end-to-end.** A linear-Gaussian least-squares problem. At `λ = 0` the analytic solution is reached in **one** iteration — the undamped step is exact on a linear problem, so anything else is a bug in `J` or in the solve. With `λ > 0` it converges to the **same** point in several, which checks that damping rescales the step without moving the optimum. | `flattened-tangent-spaces.md` §8.6 |
| 9 | **Cotangent defaulting.** A model with a free parameter that does not reach the residuals produces a zero column in `J` rather than panicking — the regression for the `vjp_forward` zero-defaulting already applied in `compiler.rs`. | `flattened-tangent-spaces.md` §5 |

Run with `cargo test --workspace`. The wgpu suites additionally exercise the device path, and will
surface the `TrilInvert` host round-trip as a sync point.

---

## Summary of files touched

Part A is applied; only the remaining work is listed.

| | file | change | |
|---|---|---|---|
| B.1 | `crates/skoro-graph/src/tangent_layout.rs` | new module | |
| B.1 | `crates/skoro-graph/src/point_map.rs` | `retract_flat`, `retract_flat_mut` (carried from Part A) | 🟢 |
| B.2 | `crates/skoro-model/src/jacobian.rs` | new module | |
| B.3 | `crates/skoro-model/src/structure.rs` | new module | |
| B.4 | `crates/skoro-dense/` | **new crate** | |
| B.5 | `crates/skoro-normal/src/structure.rs`, `facade.rs` | new module; cache on the facade | |
| B.6 | `crates/skoro-levenberg-marquardt/` | **new crate** | |
| — | `Cargo.toml` | `faer`, `skoro-dense`, `skoro-levenberg-marquardt` in `[workspace.dependencies]` | |

### Breaking changes not taken

Recorded so the choice is visible rather than forgotten.

- **`Distribution::parent_ids()` stays positional and untyped** (`skoro-normal/src/distribution.rs:164`
  returns `&[mean.id, std_dev.id]`). A richer parent description — names, or roles — would be a real
  ergonomic gain, and `model-structure-bounds.md` §8.4 flags the weakness. It is not taken because
  `normalizer_parents` (B.3) answers the only question phase 1 actually asks, and the right shape for
  the general fix is not yet known: designing it now, from one consumer, would likely produce
  something a second consumer breaks.
- **`IdGenerator: Clone` still copies the counter** (`skoro-core/src/id.rs:11-15`), so two clones of a
  graph issue colliding ids forever. Removing the impl would forbid `Graph: Clone` — which
  `sequential-estimation.md` §7 wants for snapshot/restore, where counter preservation is a *feature*.
  The right resolution belongs to the phase that builds retirement rollback, not to this one.
- **`Q<B, T>`'s type parameter stays unbounded.** Bounding `T: QuantityType<B, K>` would catch
  nonsense instantiations at construction rather than at first use, but `K` cannot be inferred at the
  struct definition without a second parameter, which is the `Q<B, T, Fam>` shape
  `model-structure-bounds.md` §6 rejects for unrelated reasons.
