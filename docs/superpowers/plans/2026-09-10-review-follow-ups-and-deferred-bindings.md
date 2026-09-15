# Review follow-ups: Fisher root cost, devices, leaf rows, and deferred bindings

Status: steps **3.1–3.3 are implemented**, and F4 is resolved (documentation only, §3.4).
Deferred bindings (§3.5) are **postponed** and not planned for some time. Of the lesser
observations (§2.5), items 1, 2 and 4 are implemented; 3, 5 and 6 remain open.

Context: the factor-bindings work
(`docs/superpowers/specs/2026-09-01-factor-bindings-and-fisher-root-spec.md`,
commits `91d4270..51ef621` on `sequential`) is complete. Every requirement R1–R30 is in, every §5
acceptance test passes, and the two bugs the review found have been fixed:

- a parameter bound as `FISHER_ROOT` (`c07d0b8`);
- a `Factor` from another model binding the wrong node (`51ef621`).

This document covers what the review left open, and proposes a plan for each item. Section 3.5 is a
full design for building bindings lazily ("deferred bindings"), which the spec (§7) put out of scope.

---

## 1. Summary

| # | Finding | Severity | Proposed step | Needs a decision? |
| --- | --- | --- | --- | --- |
| F1 | Every `Normal` builds a dense O(d⁵) matrix for its Fisher root, eagerly | High for d ≳ 10 | 3.1 sparse root op (**implemented**); 3.5 deferral (postponed) | **Decided: not for now** |
| F2 | Fixed matrices are created on the default device | Medium (breaks non-default GPU) | 3.2 (**implemented**; the Gaussian and joint roots were already fixed by 3.1) | No |
| F3 | A non-differentiable leaf listed as both row and column gets an identity block | Low (corner case) | 3.3 (**implemented**) | R22 amended |
| F4 | The Gaussian root's `(x, μ)` block is exact only on flat manifolds | Medium (semantics) | 3.4 | **Resolved: (a), docs only** |
| — | Lesser observations (§2.5) | Low | §2.5 | Some |

Steps 3.1–3.3 are done. **Deferred bindings** (3.5) are postponed: the design stays here as a record,
to revisit only if one of its triggers (§3.5.1) shows up. **F4** is resolved (§3.4).

---

## 2. Findings

### F1. Every Gaussian builds a dense O(d⁵) Fisher-root map, eagerly

**Where.** `Normal::factor` (`crates/skoro-normal/src/distribution.rs`) calls
`information_root(&w, d)` (`crates/skoro-normal/src/information.rs`). That rebuilds
`gaussian_root_matrix(d)` on the host and wraps it in a `FixedLinearMap`, whose tensor is stored in
the root node's operation for the node's lifetime. `JointNormal::factor` does the same with its
`D² × D(D+1)/2` scatter matrix.

**Why it is expensive.** `K` maps the packed whitening `W` (d(d+1)/2 entries) to the flattened root
`C` ((d + d²) × (2d + d(d+1)/2) entries), so it has O(d⁵) entries. Almost all of them are zero,
because each entry of `C` reads at most two entries of `W`:

| d | `K` dense entries | `K` as f64 | nonzeros | `C` entries |
| --- | ---: | ---: | ---: | ---: |
| 3 | 864 | 0.01 MB | 29 | 144 |
| 6 (SE3) | 29,106 | 0.23 MB | 148 | 1,386 |
| 9 | 255,150 | 2.0 MB | 411 | 5,670 |
| 12 | 1,241,136 | 9.9 MB | 872 | 15,912 |
| 15 | 4,320,000 | 35 MB | 1,585 | 36,000 |
| 20 | 22,050,000 | 176 MB | 3,480 | 105,000 |

**When the cost is paid.** On every `add_quantity(Normal::new(..))`, whether or not anything ever
reads the information:

- the host builds `K`;
- `K` is uploaded to the backend and kept alive by the root node;
- when `L` is constant, the root is constant-folded at add time, which evaluates `C` immediately.

A sliding window re-adds Gaussians every tick, so the cost recurs.

**Root cause.** A linear map with O(d³) structure is stored densely. This is a representation
problem, not a need for laziness. Step 3.1 removes nearly all of the cost; deferral (3.5) removes the
rest.

### F2. Fixed matrices are created on the default device

`information_root` (`crates/skoro-normal/src/information.rs`) and `JointNormal::factor`
(`crates/skoro-normal/src/joint.rs`) call `FixedLinearMap::from_row_major(.., &Default::default())`,
as R24 and R26 prescribe. The quantity a distribution is built over may live on another device (a
second GPU, say). `FixedLinearMap::forward` then multiplies tensors from two devices, which fails
or silently copies, depending on the backend.

### F3. A non-differentiable leaf listed as both row and column gets an identity block

R22 (`fill_leaf_rows` in `crates/skoro-model/src/jacobian.rs`) writes the identity for any output
block that is absent from the primal evaluations and also present in the input layout. An observed
or constant node is absent from the evaluations too. So if a caller lists one in both layouts, its
row is the identity even though its column is zero: neither pass seeds a constant.

This contradicts spec §8 ("Observed and constant nodes are never differentiated. Their Jacobian
rows and columns are zero"). It only arises when a caller puts a constant in the column layout,
which is a caller mistake, but the result should still be consistent.

### F4. The Gaussian root's `(x, μ)` block is exact only on flat manifolds

§4.1 sets `C = [W, −W, 0; 0, 0, C_L]`. The `[W, −W]` block assumes `∂(x ⊖ μ)/∂x = I` and
`∂(x ⊖ μ)/∂μ = −I`. On `EuclideanD` that is exact. On `SE3` or `SO3` those derivatives are the
inverse right and left Jacobians of `Log`, which equal the identity only at `x = μ`.

The review first read this as "the expected information at the mean", a first-order approximation.
The analysis in §3.4 shows it is better than that: the corrections linear in `x ⊖ μ` have mean zero
under the model, so the root is exact on flat spaces and correct to second order in the dispersion on
groups. It is still not the Gauss–Newton matrix at the current `x`.

**Resolved (2026-09-11): option (a)**, documentation only; see §3.4.

### 2.5 Lesser observations

1. **Per-term compilation in `information_blocks`.** (**Implemented**; R23 amended.) Each term with a root compiles and runs its own
   `jacobian_vjp` plus a separate `evaluate_ids` for the root: O(terms) compilations per call. One
   layout over the union of all terms' arguments, one reverse pass, and one evaluation of every root
   would be O(1) compilations. Each term's block is then sliced out. This needs no compiled-program
   cache, which §7 excludes.
2. **An orphan stochastic node when a factor fails.** (**Implemented**; R16 amended.) When `distr.factor(&x)` or the new model check
   fails inside `ModelClient::add_quantity`, the stochastic node `x` is already in the graph. It has
   no density and no handle, so it is swept at the next `cleanup()`, but it is visible until then.
   Removing `x` on the error path would make the call atomic.
3. **`ModelServer::add_factor` cannot check model identity**, because the server does not know its
   own id; its docs point to the client method. Storing the id on `ModelServer` would let the server
   check too.
4. **Spec text has drifted from the code** (**implemented**: amended in place, with an amendments
   list at the top of the spec and §1 relabelled as the pre-change record) in three places, all
   deliberate:
   - R1's message wording;
   - R3's `hvp` (zeros shaped like the input, not `ω.zeros_like()`);
   - §5.8's residual refusal, which is `NodeStillReferenced`, since a `Normal` residual feeds its own
     density.

   §1.3's "facts about the current code" are also stale now. The spec should be amended, or given a
   changelog section, so it stays a reliable reference.
5. **`skoro_backend::ndarray_simd_workaround`** must be retired with the ndarray backend. There are
   two call sites (`PointwiseAbs`, `LiteralScalarDivisionLeft`), and the backend regression test
   `unary_kernels_on_non_standard_layouts` can keep running with the plain methods as a
   backend-agnostic layout check.
6. **Four `skoro-so3` `wgpu_f32` tests fail**, and did before this work (`91d4270`). They are f32 GPU
   tolerance misses, for example a scaled error of 2.3e-6 against a 1e-6 bound. Unrelated, but worth
   tracking.

---

## 3. Plan

### 3.1 Sparse, differentiable Gaussian roots (fixes F1's cost, and F2 for Gaussians)

**Status: implemented** (`SparseLinearMap` in `crates/skoro-linalg/src/matrix/sparse_linear_map.rs`;
the roots in `crates/skoro-normal/src/information.rs` and `joint.rs`). `gaussian_root_map` is exposed
alongside `gaussian_root_triplets` so the nonzero count can be tested directly.

The dense `K` is replaced by a general sparse linear-map operation in `skoro-linalg`, and both
Gaussian roots are built from it.

**Operation.** `skoro_linalg::matrix::SparseLinearMap` is a `DiffOperation` for `y = K x` that stores
`K` as merged nonzero triplets:

```rust
#[derive(Debug, Clone)]
pub struct SparseLinearMap {
    rows: usize,
    cols: usize,
    entries: Arc<SparseEntries>, // shared, cheap to clone
}
struct SparseEntries { targets: Vec<i64>, sources: Vec<i64>, coefficients: Vec<f64> }

impl SparseLinearMap {
    /// Duplicate (row, col) pairs are summed; zero coefficients are dropped.
    pub fn from_triplets(
        rows: usize,
        cols: usize,
        triplets: impl IntoIterator<Item = (usize, usize, f64)>,
    ) -> Self;
    pub fn dims(&self) -> [usize; 2];
    pub fn nnz(&self) -> usize;
}

pub fn sparse_linear_map<B, I, O>(x: &Q<B, I>, out: O, map: SparseLinearMap) -> SkoroResult<Q<B, O>>;
```

The sugar checks the rows and then the columns, exactly as `linear_map` does.

Every mode gathers and scatter-adds, with index and coefficient tensors built on the input tensor's
device inside the call (following `TrilDiag`):

- **Forward and JVP** compute `zeros.select_assign(1, targets, x.select(1, sources) · coefficients)`,
  over vectors and frames respectively.
- **VJP** is the adjoint: gather at `targets`, scale, and scatter-add into `sources`, gated on
  `diffs[0]`.
- **HVP** returns zeros shaped like the input's cotangent.

burn's `select_assign` accumulates repeated indices on every backend; merging duplicates at
construction keeps the kernels correct regardless. No `abs` or `recip` is involved, so the ndarray
SIMD workaround is not needed.

**Roots.**

- `skoro_normal::information::gaussian_root_triplets(d)` emits the nonzeros of `K` from the §4.1
  loops.
- `gaussian_root_matrix(d)` becomes the densification of those triplets, kept as the host reference
  for tests.
- `information_root(w, d)` keeps its signature and becomes
  `sparse_linear_map(w, EuclideanMatrixD(d + d², 2d + d(d+1)/2), SparseLinearMap::from_triplets(..))`.
- `JointNormal::factor` uses the triplets `(i·D + j, i(i+1)/2 + j, 1)` in place of its dense scatter
  matrix.

Nothing is cached per dimension, since building O(d³) triplets per factor is negligible.

**Cost afterwards, per Gaussian:** one graph node; O(d³) index data (148 entries at d = 6, 1,585 at
d = 15); and, when `L` is constant, a folded `C` of (d + d²)(2d + d(d+1)/2) values.

**Tests.**

- `sparse_linear_map` on ndarray f32/f64 and wgpu: forward against a host dense product with
  duplicate triplets, numeric JVP/VJP/HVP checks, and rows-then-columns shape errors.
- `information_root` against `gaussian_root_matrix(d) · W` for d = 1…6.
- `nnz` of the Gaussian map at d = 15 is 1,585.
- All §5.7 and §5.9 tests pass unchanged.

Spec R24 and R26 are amended to describe `sparse_linear_map`.

**Rejected alternative: assembling the information on the host.** The only consumers of the
information in an optimizer are host solves (`skoro-dense`), so building the root and `AᵀA` outside
the graph was considered; it would have removed the root node entirely. It was rejected because
Riemannian manifold HMC uses the information as a metric whose derivatives enter its dynamics, so the
root must remain a differentiable graph node. The assembly in `information_blocks` (`C · J` from the
Jacobian passes) is itself not yet tape-differentiable; making it so is future RMHMC work.

### 3.2 Device placement for fixed matrices (F2)

**Status: implemented** (`crates/skoro-linalg/src/matrix/linear_map.rs`; the ignored cross-device
test is `fixed_linear_map_follows_its_input_device` in `crates/skoro-linalg/tests/wgpu_f32.rs`).

`FixedLinearMap` keeps its tensor, but `forward`, `vjp` and `hvp` use
`self.matrix.clone().to_device(&input.device())`. That is a no-op when the devices already match.
Document on `from_row_major` that the device argument is only where the matrix starts. After 3.1
removes the Gaussian and joint uses, this protects any other caller.

Test: the existing `linear_map` tests. A real cross-device test needs two devices; add it to the
wgpu suite behind an `#[ignore]` with a note, since CI has one adapter.

### 3.3 Leaf rows (F3)

**Status: implemented** (`fill_leaf_rows` in `crates/skoro-model/src/jacobian.rs`; tests in
`crates/skoro-model/tests/jacobian.rs`; R22 amended).

In `fill_leaf_rows`, write the identity only when the row id appears in `primal.variables`, which is
where a `Parameter` or `Sampler` step records its value. A constant or observed row is recorded in
`primal.constants` and stays zero, matching its zero column. No graph access is needed.

- Test: an observed node listed as both a row and a column produces zero on both paths.
- Amend R22's wording.

### 3.4 What the Gaussian root means on curved manifolds (F4) — decided

**Decision (2026-09-11): (a), keep `[W, −W]` and document its accuracy.** No code changes.

The options considered were:

- **(a) Keep and document.** `FISHER_ROOT` on a `Normal` is the expected information: exact on flat
  manifolds, and correct to second order in the dispersion on groups.
- **(b) Gauss–Newton `(x, μ)` block.** Take that block from the residual's own Jacobian at the
  current point (`jacobian_vjp` over the `RESIDUAL` row). It is exact at that point on any manifold,
  but it depends on `x`, so it is not an expected information, and it would turn the root from a node
  into an assembly rule.
- **(c) Exact expected information on Lie groups.** `E[J_r⁻ᵀ WᵀW J_r⁻¹]` over `x ~ N(μ, Σ)` has no
  closed form in general, and would need a series or sampled estimate.

Why (a):

1. **Accuracy.** The inverse retraction's Jacobians are `±I` plus terms linear in `v = x ⊖ μ`. For
   `SO3`, `J_r⁻¹(ω) = I + ½[ω]× + O(‖ω‖²)`; `SE3` adds linear translation terms. Under the model `v` is
   a zero-mean Gaussian, so the linear terms vanish in expectation, and the error of `CᵀC` is second
   order in the dispersion. That is the same order as the `Normal`-on-a-group density itself, which
   omits the retraction's volume factor `1 − ‖ω‖²/12 + …`. With an isotropic `Σ` the corrections
   vanish identically.
2. **Verified numerically.** Take `Σ = σ² L₀L₀ᵀ` for a skewed `L₀`, with 100,000 draws. The relative
   error of `CᵀC` against the covariance of the model's own score, with sampling noise cancelled by
   sharing the random draws with a flat control, is:

   | σ | SO3 | SE3 |
   | --- | ---: | ---: |
   | 0.1 | 0.012 | 0.016 |
   | 0.2 | 0.048 | 0.064 |
   | 0.4 | 0.189 | 0.255 |

   Each doubling of σ multiplies the error by 3.9–4.0.
3. **The right object for RMHMC.** A metric whose root depends only on `L` has vanishing derivatives
   with respect to `x` and `μ`. Option (b) would require derivatives of the logarithm's Jacobians.
4. **Gauss–Newton remains available.** An optimizer far from its solution can use the exact curvature
   from the `RESIDUAL` Jacobians without redefining `FISHER_ROOT`.

Documented in the "Accuracy on curved manifolds" section of the `skoro_normal::information` module,
and in `FISHER_ROOT`, `Normal::factor`, spec §2 and §4.1, and `docs/fisher-scoring.md` §8.

### 3.5 Deferred bindings (the lazy design)

**Status: postponed.** Not planned for some time. The design is kept as a record, to revisit if one
of the triggers in §3.5.1 appears.

#### 3.5.1 Is it still needed after 3.1?

Mostly for three kinds of case, not the one that motivated it. After 3.1 an eager Gaussian root costs
one node, and a folded `C` only when `L` is constant. Deferral becomes worth building when any of
these holds:

1. **A family whose root is expensive to build:** a sampled or series-estimated information,
   option 3.4(c), a non-Gaussian family, or a root that needs its own factorization.
2. **A very large model** in which most terms are never read for information, so their root nodes and
   folded values are pure overhead.
3. **A future role that is costly to materialize,** such as sparsity structure or a precomputed
   Cholesky factor.

The design below is general for that reason: nothing in it is specific to `FISHER_ROOT`.

#### 3.5.2 Goals and non-goals

Goals:

- **Nothing until read:** declaring a deferred binding adds no nodes and holds no data beyond the
  builder object.
- **Same reference semantics as a bound node:** `remove` refuses a dependency, `remove_closure`
  takes the term, and the collector marks dependencies.
- **No side table** that has to be pruned; that is what the bindings work removed.
- **No reference cycle** between the server and anything it stores.
- **Respects the model's lock discipline:** builders build through the client, so no lock is held
  while they run.
- **Idempotent and race-safe:** concurrent readers settle on one node.
- **No compiler change** (R10 still holds).

Non-goals:

- Caching evaluated values. Deferral delays *building an expression*, and the built node is evaluated
  at points like any other. Nothing needs invalidating when values change.
- Automatically discarding a built root (a "forget" call). Possible later; see §3.5.9.

#### 3.5.3 Graph data model (`crates/skoro-graph/src/binding.rs`)

A binding's target becomes one of two things:

```rust
pub enum Target {
    /// Bound now.
    Node(Id),
    /// Not built yet. `deps` are the nodes the builder will read; they are references exactly as
    /// a bound node is. `builder` is opaque to the graph (skoro-model downcasts it).
    Deferred { deps: Vec<Id>, builder: Arc<dyn Any + Send + Sync> },
}

pub struct Bindings {
    by_source: HashMap<Id, Vec<(Role, Target)>>,
    /// to -> {from}: over every `Node` target *and* every `Deferred` dependency.
    referrers: HashMap<Id, HashSet<Id>>,
}
```

Operations:

- `bind(from, role, to)` is unchanged.
- `bind_deferred(from, role, deps, builder)`.
- `bound(from, role) -> Vec<Id>` keeps returning realized nodes only.
- `deferred(from, role) -> Option<(Vec<Id>, Arc<dyn Any + Send + Sync>)>` returns the pending entry.
- `realize(from, role, &builder, to) -> Realized` replaces the `Deferred` entry whose builder is
  `Arc::ptr_eq` to `builder` with `Node(to)`, and rebuilds `from`'s referrer entries. It returns
  `Realized::Stale` if the entry is gone (the term was removed) or already realized.
- `remove_source` and `remove_target` are unchanged in spirit. A deferred entry is dropped with its
  source; a dependency cannot be removed while the source exists (below).
- `validate()` also checks that `referrers` covers deferred dependencies.

Because dependencies are ordinary referrer entries, `Graph::remove`, `remove_closure`,
`validate_indices` and `ModelServer::cleanup` need **no new logic**:

- `remove(&[dep])` fails with `NodeStillBound { id: dep, by: term }`.
- `remove_closure(&[dep])` takes the term, and with it the deferred entry.
- `cleanup` marks dependencies of marked terms (it walks `bindings_of`, which yields dependency ids
  for deferred entries).

`Graph` stays `Send + Sync` and `Clone`: builders are `Arc`s.

#### 3.5.4 Model API (`crates/skoro-model`)

```rust
/// Builds a deferred binding's node on first read.
pub trait DeferredBinding<B: Backend>: Send + Sync + 'static {
    /// Build the node from `deps` (in declaration order), through `client`.
    ///
    /// May run more than once if readers race; only one result is kept. Must not capture handles
    /// or clients — take everything graph-shaped from `deps`.
    fn build(&self, client: &ModelClient<B>, deps: &[Handle<B>]) -> SkoroResult<Handle<B>>;
}

impl<B: Backend> Factor<B> {
    /// Declare that `role` will be bound to whatever `builder` builds from `deps`.
    pub fn bind_deferred(
        self,
        role: Role,
        deps: Vec<Handle<B>>,
        builder: impl DeferredBinding<B>,
    ) -> Self;
}

impl<B: Backend> Handle<B> {
    /// The typed quantity this handle names; `QuantityTypeMismatch` otherwise.
    pub fn typed<K: Kind<B>, T: QuantityType<B, K>>(&self) -> SkoroResult<Q<B, T>>;
}

pub trait ModelClientOps<B: Backend> {
    /// Build `id`'s deferred binding under `role`, if it has one, then return what it binds.
    fn resolve(&self, id: Id, role: Role) -> SkoroResult<Vec<Id>>;
    // `bound` is unchanged: realized nodes only.
}
```

`Factor` keeps the dependency `Handle`s alive until `add_quantity` or `add_factor` records the entry,
which is the same guarantee eager bindings rely on. The model check added in `51ef621` extends to
dependency handles.

**Why builders must not capture handles.** A `Handle` holds a `ModelClient`, which holds
`Arc<RwLock<ModelServer>>`. The server holds the graph, which would hold the builder, so the server
could never be dropped. Builders receive handles as arguments instead, freshly minted at resolve
time. If a builder ever needs the client outside `build`, it can store the `server_id` and call
`ModelClient::from_id`, which goes through the existing weak registry.

#### 3.5.5 Resolution protocol

`resolve(id, role)` on `ModelClient`:

1. **Read lock.** If `graph.deferred(id, role)` is `None`, return `graph.bound(id, role)`.
   Otherwise clone the dependency ids and the builder `Arc`, mint a `Handle` for each dependency
   (a `NodeDropGuard` only needs the handle registry), and **release the lock**.
2. **No lock held.** Downcast the builder and call `build(self, &deps)`. It adds nodes through the
   client, taking and releasing the write lock per node, exactly as `Distribution::factor` does
   inside `add_quantity`.
3. **Write lock.** `graph.realize(id, role, &builder, built.id)`:
   - `Done`: return `vec![built.id]`. The built node's handle drops afterwards, and the binding keeps
     the node alive.
   - `Stale`: another reader won, or the term was removed meanwhile. Return
     `graph.bound(id, role)` (possibly empty) and drop our handle. Our nodes are unreachable and go at
     the next `cleanup()`; mark the handle registry dirty so they do.

Correctness under concurrency:

- **Two readers:** both may build, one realizes, and the other is `Stale` and discards its nodes. The
  builder is deterministic up to node identity, so which one wins is irrelevant.
- **Reader against `remove_closure` of the term:** the entry is dropped with the source, so
  `realize` is `Stale` and the built nodes are discarded.
- **Reader against removal of a dependency:** impossible without also removing the term (the
  dependency is still bound), which reduces to the previous case.
- **Reader against demotion:** the retired density keeps its entry until swept. A reader who
  realizes in between builds a node no term uses, and it goes with the retired density at the next
  sweep.

**Constant folding** happens when the node is built, as it would eagerly, so the folded value is paid
only by terms that are read.

#### 3.5.6 Using it

`Normal::factor`, on top of 3.1:

```rust
Factor::new(density)
    .bind(RESIDUAL, &r)
    .bind(LOCATION, &self.mean)
    .bind(WHITENING, &self.std_dev)
    .bind(NORMALIZER_PARENT, &self.std_dev)
    .bind_deferred(FISHER_ROOT, vec![Handle::from(&w)], GaussianRootBuilder { d })
```

Here `w = L⁻¹` is already part of the density's expression, so depending on it costs nothing.
`GaussianRootBuilder::build` does `deps[0].typed::<Continuous, Tril>()?` and registers
`GaussianFisherRoot { d }` over it. `JointNormal` defers `TrilUnpack` over its whitening the same way.

`information_blocks` calls `client.resolve(f.density, FISHER_ROOT)` instead of `client.bound(..)`.
`least_squares::residual_layout` and `GaussianBlock` keep using `bound`, since those roles stay eager.

#### 3.5.7 Decisions for review

1. **Tier.** `resolve` grows the graph, but it cannot change the log-density or the set of terms, and
   it is idempotent, much like `cleanup()`. Proposed: `ModelClientOps`, so `information` keeps its
   estimator-tier bound. The alternative is `ModelBuilderOps`, which moves `information` to the
   builder tier.
2. **Explicit or transparent.** Proposed: an explicit `resolve`, with `bound` staying a pure read.
   A transparent `bound` would take a write lock on a read call and hide node creation.
3. **Default for `FISHER_ROOT`.** Proposed: deferred for `Normal` and `JointNormal` once this exists,
   with `RESIDUAL`, `LOCATION`, `WHITENING` and `NORMALIZER_PARENT` staying eager (they are shared
   with the density and free).
4. **Cardinality.** A term should not carry both a `Node` and a `Deferred` entry for a
   single-valued role. Enforce this in `Factor` construction (an error), or leave it as a
   documented contract.

#### 3.5.8 Phases and tests

| Phase | Scope | Tests |
| --- | --- | --- |
| D1 | `Target`, `bind_deferred`, `deferred`, `realize`, `validate` in `skoro-graph` | dependencies refuse `remove` (`NodeStillBound`); `remove_closure(dep)` takes the term and the entry; `validate_indices` holds after every mutation; `realize` returns `Stale` after the source is removed or already realized |
| D2 | `DeferredBinding`, `Factor::bind_deferred`, `Handle::typed`, `ModelClientOps::resolve`, collector marking of dependencies | nothing is built before `resolve` (node count and a build counter); a second `resolve` does not build; two threads resolving concurrently settle on one id and the loser is collected after `cleanup()`; resolving a term removed mid-build returns empty and leaks nothing; a dependency survives `cleanup()` with no handle |
| D3 | `Normal`/`JointNormal` defer their roots; `information_blocks` resolves | all §5.7 and §5.9 tests pass (tests that inspect the root via `bound` switch to `resolve`); adding N Gaussians adds exactly N fewer nodes than eager; the folded-root test observes folding at resolve time |
| D4 | Spec and docs: `FISHER_ROOT` contract ("may be deferred; read with `resolve`"), §7 updated, readme | `cargo doc` without new warnings |

Each phase is additive until D3, which is the only behavioural change: a root is absent from
`bound()` until it is resolved.

#### 3.5.9 Alternatives considered

- **A per-term side table on `ModelServer`** (density → builder). Rejected: it recreates the
  registry's pruning problem, since every removal and demotion path must remember to prune it.
  Storing entries inside `Bindings` means they die with their source for free.
- **No root nodes at all: "family kernels" in `information_blocks`.** A kernel recognizes a term by
  its bindings (for example `WHITENING` + `LOCATION` + three arguments) and computes `C · J`
  directly, never materializing `C`. It is the most efficient option: O(d²·F) with zero nodes.
  Rejected as the primary design because:
  - recognition by binding shape is implicit (a term could bind those roles for another reason);
  - callers would have to pass the right kernels to `information`;
  - every consumer of a root would need its own kernel.

  It remains a good later optimization *inside* `information_blocks` for known root operations.
- **Transparent resolution in `bound`.** Rejected; see decision 2.
- **Forgetting realized nodes** (turning a `Node` back into `Deferred` to reclaim memory). Deferred
  until a workload needs it. A sliding window already retires whole terms, which is the dominant
  lifetime.

---

## 4. Decisions requested

1. Steps 3.1–3.3: done, with the R22/R24/R26 spec amendments.
2. F4: decided, option (a) (§3.4).
3. Deferred bindings: decided, wait. Not for some time; §3.5.1 lists what would reopen it, and
   §3.5.7 the questions that would then need answers.
4. Lesser observations (§2.5): items 1, 2 and 4 are done; 3 is unscheduled; 5 waits for the flex
   backend; 6 is separate `skoro-so3` work.
