# Specification: factor bindings and Fisher information roots

This document is a complete, self-contained specification for a change to the `skoro`
workspace. It is written so that an implementer with no prior context can carry it out. Every
requirement is numbered (`R1`, `R2`, ...) and every acceptance test is named. Do not add
features that are not listed here; §7 lists what is deliberately out of scope.

Workspace root: `/home/mvarble/dev/skoro`. Rust 2024 edition. Run all commands from the root.

---

## 0. Goal in one paragraph

Today, when a distribution is added to a model, everything it knew is discarded except a sampler
and an expression tree for its log-density. A side table (`FactorRecord` / `ResidualRegistry`
in `crates/skoro-model/src/residual.rs`) re-attaches one fact, the whitened residual, and needs
its own pinning and pruning machinery. This change replaces that side table with **bindings**:
labelled reference edges stored in the graph, from a log-density term to other nodes. A
distribution declares its bindings once when it is added. The garbage collector treats a binding
as a reference. Any inference algorithm reads the bindings it needs. The first new binding is a
**Fisher information root**, a matrix-valued node `C` with `CᵀC` equal to the expected
information of the term over its arguments, which is what Fisher scoring needs.

---

## 1. Facts about the current code (verified; rely on these)

### 1.1 Graph (`crates/skoro-graph`)

- `Graph<B>` (`src/graph.rs:55-72`) has private fields `id_generator`, `nodes: HashMap<Id, Node<B>>`,
  `children: HashMap<Id, HashSet<Id>>` (reverse of `parent_ids`), `log_likelihoods: Vec<Id>`
  (the scoring terms, in marking order), `stochastic_to_log_density: HashMap<Id, Id>`,
  `pinnable: HashSet<Id>`, `private_nodes: HashSet<Id>`.
- `Node<B>` (`src/node.rs:10-25`) has six variants: `Constant`, `Parameter`, `Stochastic`,
  `Observed`, `Mapped`, `LogLikelihood`. `LogLikelihoodNode` has `stochastic_id: Option<Id>`:
  `Some(x)` means "this term is the log-density of x"; `None` means a free-standing term.
- `Graph::remove(&mut self, ids: &[Id])` (`src/graph.rs:1269`) refuses the batch if any node in
  it has a child outside the batch (`SkoroError::NodeStillReferenced { id, child_id }`), then
  removes each node, calling `unregister_children` and fixing the per-kind indices.
- `Graph::remove_closure(&mut self, roots)` (`src/graph.rs:221`) collects the downward closure
  through `children`, orders it, calls `remove`, then `reclaim_private_nodes`.
- `Graph::validate_indices(&self) -> Result<(), String>` (`src/graph.rs:295`) recomputes every
  derived index from `nodes` and reports the first disagreement. `debug_validate_indices` runs
  it after every mutation in debug builds.
- `Graph::demote_to_parameter` (`src/graph.rs:1232`) turns a `Stochastic` node into a
  `Parameter`, severs its parent edges, and calls `retire_density_of` (`src/graph.rs:1888`),
  which turns the density back into an ordinary `Mapped` node and removes it from
  `log_likelihoods`. The node is **not** deleted.
- `Graph::log_density(points)` (`src/graph.rs:1373`) compiles `log_likelihoods`, runs a
  forward pass, and sums the evaluated terms into a `[batch, 1]` tensor.
- `Graph::compile_backward_from(targets)` (`src/graph.rs:2038`) lowers `Constant` and
  `Observed` to `CompiledStep::Constant`, `Parameter` to `CompiledStep::Parameter`,
  `Stochastic` to `CompiledStep::Sampler`, `Mapped`/`LogLikelihood` to `CompiledStep::Operation`.
- The VJP pass (`src/compiler.rs:298-391`) reads cotangent seeds only at `Operation` steps and
  collects cotangents only at `Parameter` and `Sampler` steps. A continuous leaf that received
  nothing gets an explicit zero cotangent. So: a leaf can be a Jacobian **column** but not a
  Jacobian **row** today.
- `TangentLayout` (`src/tangent_layout.rs:43`): `from_ids(graph, ids)` sizes each block by
  **manifold dimension** (`SO3` is 3, `SE3` is 6, `Tril(n)` is `n(n+1)/2`,
  `EuclideanMatrixD(m, n)` is `m·n`). Errors: `NotContinuous`, `NodeNotFound`,
  `DuplicateLayoutId`. `identity_frames(batch, device)` gives per-block identity slabs.
- `PointMap::get(&id) -> Option<(&Input<B>, &Arc<dyn DynQuantityType<B>>)>`
  (`src/point_map.rs:43`); `DynQuantityType::as_any()` allows downcasting the quantity type.

### 1.2 Model (`crates/skoro-model`)

- `ModelServer<B>` (`src/model.rs:45-59`) owns `graph`, `device`, `handles: Arc<HandleRegistry>`,
  `registry: ResidualRegistry`. `ModelClient<B>` is `Arc<RwLock<ModelServer<B>>>` plus an id.
- `Q<B, T>` (`src/q.rs:20-27`) has private fields `id`, `client: ModelClient<B>`,
  `guard: Arc<NodeDropGuard>`. `Q::new(id, client, guard)` is `pub(crate)`. `Q::id()` is public.
  The guard registers the id as a collector root while any clone of the `Q` lives.
- `ModelServer::cleanup` (`src/model.rs:434-492`) is a mark-and-sweep. Roots: live handles,
  `graph.log_likelihoods()`, and the nodes every `FactorRecord` names (`model.rs:444-449`). The
  mark walks `parent_ids()` and `graph.stochastic_of(id)`. It runs at the end of `remove`,
  `remove_closure`, `demote_to_parameter`, and on explicit `cleanup()`. Never on `add_*`.
- `ModelClient::add_quantity` (`src/model.rs:832-851`): takes the write lock, adds the
  stochastic node, builds the `Q`, **drops the lock**, calls `distr.log_density(&x)`, re-takes
  the lock, calls `mark_density`. The lock drop is mandatory (the distribution builds nodes
  through the client).
- `Distribution<B, K>` (`src/distribution.rs:19-34`): `type Qty`, `type Sampler`,
  `quantity_type()`, `sampler()`, `parent_ids() -> &[Id]`,
  `log_density(&self, x: &Q<B, Qty>) -> SkoroResult<Q<B, Real>>`.
- `ModelClientOps` / `ModelBuilderOps` (`src/client_ops.rs:51`, `:268`) are the two capability
  tiers. `remove` is on the client tier and refuses to break references; `remove_closure` is on
  the builder tier and takes referents too.
- `jacobian_jvp` / `jacobian_vjp` (`src/jacobian.rs:103`, `:136`) return
  `[batch, outputs.total(), inputs.total()]`. `check_rows` (`jacobian.rs:82-90`) rejects a row
  that is not a mapped node with `SkoroError::NotMapped`.
- `register_map1(quantity_type, op, &q)` (`src/map.rs:27`) registers a unary `DiffOperation` as
  a mapped node.
- Value maps are built with `skoro_model::hash_map_builders::points()` / `point_map()` and
  `.with_quantity(&q, tensor)` / `.with_tensor(id, tensor)` / `.build()`.

### 1.3 Normal (`crates/skoro-normal`)

- `Normal<B, M>` (`src/distribution.rs:96`) holds `mean: Q<B, M>`, `std_dev: Q<B, Tril>`,
  `quantity_type: M`, `constant: f64`, `ids: [Id; 2]` (`[mean, std_dev]`).
- `Normal::whitened_residual(&self, x)` (`src/distribution.rs:154`) builds
  `r = L⁻¹ (x ⊖ μ)` via `self.mean.inverse_retract(x)` then `self.std_dev.invert()?.vector_product(..)`.
- `Normal::log_density` (`src/distribution.rs:189-197`) builds
  `−½‖r‖² + constant + Σ log|diag(L⁻¹)|`, calling `invert()` a second time.
- `facade::add_gaussian_quantity` (`src/facade.rs:38`) adds the quantity, builds a **second**
  residual, and records a `FactorRecord`. `facade::gaussian_blocks` reads records.
- `structure::GaussianBlock` (`src/structure.rs:20`) is a view of a record with `μ`/`L`.
- `joint::JointNormal` (`src/joint.rs:46`) is a factor over several existing quantities with a
  constant whitening `W: Q<B, Tril>`; `add_joint_factor` (`src/joint.rs:173`) marks a likelihood
  and records a factor.
- `Tril(n)` packs the lower triangle row-major: entry `(i, j)`, `j ≤ i`, at index
  `i(i+1)/2 + j` (`crates/skoro-linalg/src/tril/manifold.rs:47-57`).
- `EuclideanMatrixD(rows, cols)` stores row-major (`crates/skoro-riemann/src/matrix.rs:197-217`).
- `skoro-linalg` matmul sugar supports `Q<EuclideanMatrixD> · Q<EuclideanD> → Q<EuclideanD>`
  (`src/matrix/matmul/sugar.rs:70`).

### 1.4 Operations (`crates/skoro-operation`)

- `DiffOperation<B>` (`src/diff/base.rs:123`): `type VjpState`, `type HvpState`,
  `forward(inputs: Vec<Input<B>>, diff_state: DiffModeState<B>) -> DiffOperationForward<..>`,
  `vjp(state, omega: CotangentFrame<B>) -> Vec<Option<CotangentFrame<B>>>`,
  `hvp(vjp_state, hvp_state, omega) -> Vec<Option<CotangentFrame<B>>>`.
- `DiffModeState`: `ForwardOnly`, `Jvp(Vec<Option<TangentFrame>>)`, `Vjp(Vec<bool>)`,
  `Hvp { diffs: Vec<bool>, tangents: Vec<Option<TangentVector>> }`. The returned
  `DiffOperationForward` variant **must** match (`forward_only`, `jvp`, `vjp`, `hvp` constructors).
- Shapes: `Point` is `[batch, storage]`; `TangentVector`/`CotangentVector` `[batch, dim]`;
  `TangentFrame`/`CotangentFrame` `[batch, dim, frames]`.
- Reference implementation of a linear op: `LiteralScalarMultiplication`
  (`src/ops.rs:644-706`). Note its `hvp` returns `Some(omega.zeros_like())` when the input is
  differentiable and `None` otherwise. Copy that gating exactly.
- Read `.claude/skills/skoro-diff-operation/SKILL.md` before writing the new op (§3.2).

---

## 2. Definitions

- **Term**: a `Node::LogLikelihood` that is in `log_likelihoods`. Its **scored** quantity is
  `stochastic_id`.
- **Binding**: a triple `(from: Id, role: Role, to: Id)` meaning "`to` plays role `role` for
  `from`". Stored in the graph, not in any node. A binding is a **reference**: it keeps `to`
  alive and forbids removing `to` while `from` exists.
- **Arguments** of a term: the nodes bound under `ARGUMENT`, in binding order. For a conditional
  `p(x | u₁..u_m)` they are `[x, u₁, .., u_m]`. For a joint factor they are the quantities it scores.
- **Fisher root** of a term: a node of type `EuclideanMatrixD(k, D)` with
  `D = Σ dim(arguments)` such that `CᵀC = E[−∂² log p]` over the arguments' tangent
  coordinates (in `ARGUMENT` order), the expectation being over the scored quantity given the
  parents. `k` may differ from `D`.
- **Free layout**: a `TangentLayout` over the leaves an algorithm optimises.

---

## 3. Requirements

### 3.1 skoro-core

**R1.** In `crates/skoro-core/src/error.rs`, remove `SkoroError::NodeIsRecorded { id }` and add
`SkoroError::NodeStillBound { id: Id, by: Id }` with the message
"node {id} is still bound by {by}". Fix every use site (`grep -rn NodeIsRecorded crates`).

### 3.2 skoro-linalg: a fixed linear map operation

**R2.** Add `crates/skoro-linalg/src/matrix/linear_map.rs` with

```rust
#[derive(Debug, Clone)]
pub struct FixedLinearMap<B: Backend> { matrix: Tensor<B, 2> }   // shape [out, in]
impl<B: Backend> FixedLinearMap<B> {
    pub fn new(matrix: Tensor<B, 2>) -> Self;
    pub fn from_row_major(rows: usize, cols: usize, data: &[f64], device: &B::Device) -> Self;
}
```

`from_row_major` builds the tensor through `TensorData::new(data.to_vec(), [rows, cols])`
(never `Tensor::from_floats`, which rounds through `f32`).

**R3.** `impl<B: Backend> DiffOperation<B> for FixedLinearMap<B>` with `VjpState = bool`,
`HvpState = ()`, one input `x: [batch, in]`:
- forward output `y = x · Kᵀ`, shape `[batch, out]` (`x.matmul(K.transpose())` with `K`
  unsqueezed to `[1, out, in]` and broadcast, or an equivalent).
- `Jvp`: tangent frame `dx: [batch, in, f]` maps to `K · dx`, shape `[batch, out, f]`.
- `Vjp`: state is `diffs[0]`; `vjp` returns `[Some(Kᵀ · ω)]` with `ω: [batch, out, f]` giving
  `[batch, in, f]` when the state is `true`, else `[None]`.
- `Hvp`: forward tangent as in `Jvp` but on a rank-2 vector; `hvp` returns
  `[Some(ω.zeros_like())]` when the vjp state is `true`, else `[None]` (linear map).
- Backend-generic: no ndarray/cube split.

**R4.** Sugar in the same module, exported from `skoro_linalg::matrix`:

```rust
pub fn linear_map<B, I, O>(x: &Q<B, I>, out: O, map: FixedLinearMap<B>) -> SkoroResult<Q<B, O>>
where I: ContinuousQuantity<B>, O: ContinuousQuantity<B>
```

It must check `map.matrix.dims() == [out.dimension(), x.get_quantity_type().dimension()]` and
return `SkoroError::ShapeMismatch { expected, found }` otherwise, then call `register_map1`.

### 3.3 skoro-graph: bindings and terms

**R5.** Add `crates/skoro-graph/src/binding.rs`, declare `pub mod binding;` in `src/lib.rs`:

```rust
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct Role(pub &'static str);
impl std::fmt::Display for Role { /* prints the str */ }

#[derive(Default, Debug, Clone)]
pub struct Bindings {
    by_source: HashMap<Id, Vec<(Role, Id)>>,   // insertion order kept; a role may repeat
    referrers: HashMap<Id, HashSet<Id>>,       // to -> set of from
}
impl Bindings {
    pub fn bind(&mut self, from: Id, role: Role, to: Id);
    pub fn of(&self, from: Id) -> &[(Role, Id)];                 // empty slice if none
    pub fn bound(&self, from: Id, role: Role) -> Vec<Id>;         // in binding order
    pub fn referrers_of(&self, to: Id) -> impl Iterator<Item = Id> + '_;
    pub fn remove_source(&mut self, from: Id);   // drops from's bindings; fixes referrers
    pub fn remove_target(&mut self, to: Id);     // drops every (_, to) pair; fixes referrers
    pub fn validate(&self) -> Result<(), String>; // referrers == reverse of by_source, no empty sets
}
```

**R6.** `Graph<B>` gains a private field `bindings: Bindings` and public methods:

```rust
pub fn bind(&mut self, from: Id, role: Role, to: Id) -> SkoroResult<()>;
pub fn bindings_of(&self, from: Id) -> &[(Role, Id)];
pub fn bound(&self, from: Id, role: Role) -> Vec<Id>;
pub fn referrers_of(&self, to: Id) -> Vec<Id>;
```

`bind` returns `NodeNotFound { id }` if either endpoint is not in `nodes` (check `from` first),
otherwise records the binding and calls `debug_validate_indices`. Duplicate identical triples are
allowed and stored twice. No cycle check.

**R7.** `Graph::remove`: before removing anything, for every `id` in the batch, if some
`referrer` in `referrers_of(id)` is **not** in the batch, return
`NodeStillBound { id, by: referrer }`. (Keep the existing `NodeStillReferenced` check first.)
When removing a node: call `self.bindings.remove_source(id)` and
`self.bindings.remove_target(id)`.

**R8.** `Graph::remove_closure`: the closure walk extends the queue with
`referrers_of(id)` as well as `children.get(&id)`.

**R9.** `Graph::validate_indices`: additionally return an error if any binding endpoint is not
in `nodes`, or if `self.bindings.validate()` fails.

**R10.** The compiler (`topological_sort`, `compile_backward_from`) must not read bindings.

**R11.** Terms. In `src/graph.rs` (or a new `src/term.rs` re-exported from `graph`):

```rust
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum TermKind { Proposal, Weight }
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct Term { pub density: Id, pub scored: Option<Id>, pub kind: TermKind }

impl<B: Backend> Graph<B> {
    pub fn terms(&self) -> Vec<Term>;   // one per id in log_likelihoods, same order
    pub fn log_density_terms(&self, points: &ValueMap<B>, terms: &[Id]) -> SkoroResult<Point<B>>;
}
```

`kind` is `Proposal` iff `scored == Some(s)` and `self.nodes[s]` is `Node::Stochastic`;
otherwise `Weight`. `log_density_terms` returns `NotLogLikelihood { id }` for any id not in
`log_likelihoods`, and otherwise behaves exactly like `log_density` restricted to `terms`
(an empty slice yields zeros of shape `[batch, 1]`, batch taken from `points` as `log_density` does).

### 3.4 skoro-model: roles, handles, factors

**R12.** Add `crates/skoro-model/src/roles.rs`, `pub mod roles;` in `lib.rs`:

```rust
pub use skoro_graph::binding::Role;
pub const ARGUMENT: Role = Role("argument");
pub const RESIDUAL: Role = Role("residual");
pub const NORMALIZER_PARENT: Role = Role("normalizer_parent");
pub const LOCATION: Role = Role("location");
pub const WHITENING: Role = Role("whitening");
pub const FISHER_ROOT: Role = Role("fisher_root");
```

Each constant carries a doc comment stating its contract (from §2 and this table):

| Role | Cardinality per term | Contract |
|---|---|---|
| `ARGUMENT` | ≥ 1, ordered | the term's local views are expressed over these, in this order; nodes must be distinct and continuous |
| `RESIDUAL` | 0 or 1 | an `EuclideanD` node `r` with `−log p = ½‖r‖² + n` |
| `NORMALIZER_PARENT` | any | every node `n` depends on |
| `LOCATION` | 0 or 1 | `μ` in `r = L⁻¹(x ⊖ μ)` |
| `WHITENING` | 0 or 1 | `L` in `r = L⁻¹(x ⊖ μ)`, with `Σ = LLᵀ` |
| `FISHER_ROOT` | 0 or 1 | `EuclideanMatrixD(k, Σ dim(ARGUMENT))`, `CᵀC` = expected information over the arguments in `ARGUMENT` order |

**R13.** In `src/q.rs` add

```rust
pub struct Handle<B: Backend> { pub(crate) id: Id, pub(crate) client: ModelClient<B>, pub(crate) guard: Arc<NodeDropGuard> }
impl<B: Backend> Handle<B> { pub fn id(&self) -> Id }
impl<B: Backend, T> From<&Q<B, T>> for Handle<B>   // clones the three fields
impl Clone, Debug for Handle
```

**R14.** In `src/distribution.rs`:

```rust
pub struct Factor<B: Backend> { pub log_density: Q<B, Real>, pub bindings: Vec<(Role, Handle<B>)> }
impl<B: Backend> Factor<B> {
    pub fn new(log_density: Q<B, Real>) -> Self;
    pub fn bind<T>(mut self, role: Role, q: &Q<B, T>) -> Self;   // pushes (role, Handle::from(q))
}
```

and the trait method `log_density` is **replaced** by

```rust
fn factor(&self, x: &Q<B, Self::Qty>) -> SkoroResult<Factor<B>>;
```

**R15.** `ModelServer` gains `bind(&mut self, from: Id, role: Role, to: Id) -> SkoroResult<()>`
delegating to `Graph::bind`, and

```rust
pub fn add_factor(&mut self, factor: &Factor<B>, arguments: &[Id]) -> SkoroResult<()>
```

which calls `mark_likelihood(&factor.log_density)`, then binds `ARGUMENT` to each of
`arguments` in order, then binds every `(role, handle)` in `factor.bindings`. If `arguments` is
empty return `ShapeMismatch { expected: 1, found: 0 }` before mutating anything.

**R16.** `ModelClient::add_quantity` becomes:

1. write lock; `add_stochastic`; build `x`; **drop the lock**;
2. `let factor = distr.factor(&x)?;`
3. write lock; `mark_density(&factor.log_density, &x)?`;
   bind `ARGUMENT` to `x.id()` then to each id of `distr.parent_ids()` in order;
   bind every `(role, handle)` of `factor.bindings`; return `x`.

`ModelClient::add_factor(&self, factor: Factor<B>, arguments: &[Id]) -> SkoroResult<Q<B, Real>>`
takes the write lock, calls the server, returns `factor.log_density`.
`ModelClient::bind(&self, from: Id, role: Role, to: Id) -> SkoroResult<()>` likewise.

**R17.** `ModelServer::cleanup`: the root set is live handles plus `graph.log_likelihoods()`
only (delete the record roots). In the mark loop, after pushing `parent_ids()` and
`stochastic_of(id)`, also push every `to` in `graph.bindings_of(id)`.

**R18.** Delete: `src/residual.rs` and its `pub mod`/re-exports; the `registry` field;
`ModelServer::{record_factor, registry, residual_layout, prune_registry, debug_validate_state}`;
the `registry.names` check in `ModelServer::remove`; `record_factor`, `factors`,
`residual_layout`, `mark_likelihood` from `ModelBuilderOps`; the `crossbeam-queue` dependency if
it is now unused.

**R19.** `ModelClientOps` gains:

```rust
fn factors(&self) -> Vec<FactorInfo>;                 // one per term, term order
fn bindings_of(&self, id: Id) -> Vec<(Role, Id)>;
fn bound(&self, id: Id, role: Role) -> Vec<Id>;
fn terms(&self) -> Vec<Term>;
fn log_proposal(&self, points: &ValueMap<B>) -> SkoroResult<Point<B>>;  // sum of Proposal terms
fn log_weight(&self, points: &ValueMap<B>) -> SkoroResult<Point<B>>;    // sum of Weight terms
```

with, in `src/client_ops.rs`,

```rust
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct FactorInfo { pub density: Id, pub scored: Option<Id>, pub kind: TermKind, pub arguments: Vec<Id> }
```

`arguments` is `bound(density, ARGUMENT)`.

**R20.** `ModelBuilderOps` after the change contains exactly: `add_quantity`, `add_factor`,
`bind`, `stacked_tangent`, `remove_closure`.

**R21.** Add `crates/skoro-model/src/least_squares.rs` with

```rust
pub fn residual_layout<B: Backend, C: ModelClientOps<B>>(client: &C) -> SkoroResult<TangentLayout>
```

collecting, for each `FactorInfo` in order, `bound(density, RESIDUAL)`; a term with no residual
is skipped; a term with more than one returns `ShapeMismatch { expected: 1, found: n }`; the
layout is `client.tangent_layout(&ids)`.

### 3.5 skoro-model: Jacobian rows may be leaves

**R22.** In `src/jacobian.rs` delete `check_rows` and the `NotMapped` error path. After
assembling `J` (both `jacobian_jvp` and `jacobian_vjp`), for every block `r` of `outputs` whose
id is **absent from** `primal.evaluations` (i.e. it is a `Parameter`, `Stochastic`, `Constant`
or `Observed` node):

- if `inputs.block(r.id)` is `Some(c)`: set `J[:, r.offset..r.offset+r.dimension, c.offset..c.offset+c.dimension]`
  to the identity (`r.dimension == c.dimension` necessarily);
- otherwise leave those rows zero (they already are).

Update the module docs: rows may be any continuous node; a leaf row is the identity on its own
columns and zero elsewhere.

### 3.6 skoro-model: Fisher assembly

**R23.** Add `crates/skoro-model/src/information.rs`, `pub mod information;`:

```rust
pub struct InformationBlock<B: Backend> { pub factor: Id, pub root: Tensor<B, 3> } // [batch, k, free.total()]

pub fn information_blocks<B, C: ModelClientOps<B>>(client: &C, points: &ValueMap<B>, free: &TangentLayout)
    -> SkoroResult<Vec<InformationBlock<B>>>;
pub fn information_root<B, C>(client: &C, points: &ValueMap<B>, free: &TangentLayout) -> SkoroResult<Tensor<B, 3>>; // [batch, Σk, D]
pub fn information<B, C>(client: &C, points: &ValueMap<B>, free: &TangentLayout) -> SkoroResult<Tensor<B, 3>>;      // [batch, D, D]
```

`information_blocks`, for each `f` in `client.factors()`:

1. `roots = client.bound(f.density, FISHER_ROOT)`. Empty: skip this term. More than one:
   `ShapeMismatch { expected: 1, found: n }`.
2. `f.arguments` empty: `ShapeMismatch { expected: 1, found: 0 }`.
   `args = client.tangent_layout(&f.arguments)?`.
3. `j = jacobian_vjp(client, points, free, &args)?` — shape `[batch, args.total(), free.total()]`.
4. Evaluate the root: `out = client.evaluate_ids(points, &[root])?`; take
   `out.evaluations.get(&root)` (if `None`, the root is a constant-folded or leaf node; then
   look in `out.constants`); downcast its quantity type to `EuclideanMatrixD(k, cols)`
   (error `QuantityTypeMismatch { id: root }` if it is not one); require `cols == args.total()`
   (else `LayoutDimensionMismatch { expected: args.total(), found: cols }`); reshape the float
   tensor `[batch, k·cols]` to `[batch, k, cols]`. If the evaluated tensor has batch 1 and
   `points` has batch `b > 1`, repeat it along dim 0 to `b`.
5. `block = root.matmul(j)` → `[batch, k, free.total()]`. Push `InformationBlock { factor: f.density, root: block }`.

`information_root` concatenates the blocks along dim 1 in order; with no blocks it returns
`ShapeMismatch { expected: 1, found: 0 }`. `information` returns `Aᵀ A` via
`a.clone().swap_dims(1, 2).matmul(a)`.

Batch and device are taken from the first entry of `points` exactly as `jacobian.rs:47-52` does.

### 3.7 skoro-normal

**R24.** Add `crates/skoro-normal/src/information.rs`:

```rust
/// K with vec(C) = K · vec(W), W = L⁻¹ packed (Tril), C the Gaussian Fisher root over [x, μ, L].
pub fn gaussian_root_matrix(d: usize) -> (usize, usize, Vec<f64>);   // (rows, cols, row-major data)
pub fn information_root<B, M>(w: &Q<B, Tril>, d: usize) -> SkoroResult<Q<B, EuclideanMatrixD>>;
```

`gaussian_root_matrix(d)` implements §4.1 exactly. `information_root` builds
`FixedLinearMap::from_row_major(rows, cols, &data, &Default::default())` and returns
`linear_map(w, EuclideanMatrixD(d + d*d, 2*d + d*(d+1)/2), map)`.

**R25.** `Normal::factor(&self, x)` replaces `Normal::log_density`:

```
d  = self.quantity_type.dimension()
n  = self.std_dev.get_quantity_type::<Continuous>().0;  if n != d → ShapeMismatch { expected: d, found: n }
w  = self.std_dev.invert()?                                      // built ONCE
r  = w.vector_product(&self.mean.inverse_retract(x)?)?
log_det = w.diag()?.abs()?.log()?.sum()?
density = ((r.norm_squared()? * -0.5)? + self.constant)? + log_det   // a SkoroResult<Q<Real>>
root = information_root(&w, d)?
Factor::new(density?)
    .bind(RESIDUAL, &r).bind(LOCATION, &self.mean).bind(WHITENING, &self.std_dev)
    .bind(NORMALIZER_PARENT, &self.std_dev).bind(FISHER_ROOT, &root)
```

Keep `whitened_residual` as a public helper (it may build its own nodes).

**R26.** `JointNormal` gains `pub fn factor<C: ModelBuilderOps<B>>(&self, client: &C) -> SkoroResult<Factor<B>>`
binding `RESIDUAL` (the residual node), `NORMALIZER_PARENT` (the whitening), and `FISHER_ROOT`
= the dense whitening as `EuclideanMatrixD(D, D)` built with `linear_map(&self.whitening, .., P)`
where `D = Σ dim(ids)` and `P` is the `[D·D, D(D+1)/2]` scatter matrix with
`P[i·D + j, i(i+1)/2 + j] = 1` for `j ≤ i`, zero elsewhere. `add_joint_factor(client, &joint)`
becomes `client.add_factor(joint.factor(client)?, joint.ids())`.

**R27.** Delete `facade::add_gaussian_quantity` and `facade::gaussian_blocks`. Keep the three
`constant_covariance_*` helpers. A Gaussian enters a model through
`client.add_quantity(Normal::new(mean, std_dev))`.

**R28.** `structure::GaussianBlock` keeps its fields and gains
`pub fn of<B, C: ModelClientOps<B>>(client: &C, factor: &FactorInfo) -> Option<Self>` (all of
`RESIDUAL`, `LOCATION`, `WHITENING` present, exactly one each) and
`pub fn all<B, C>(client: &C) -> Vec<Self>` over `client.factors()`. Delete `of(&FactorRecord)`.

### 3.8 Umbrella crate and docs

**R29.** `crates/skoro/src/lib.rs` prelude: remove `FactorRecord`, `ResidualRegistry`; add
`Role`, `roles`, `Factor`, `FactorInfo`, `Handle`, `Term`, `TermKind`,
`information::{information, information_root, information_blocks, InformationBlock}`,
`least_squares::residual_layout`, `FixedLinearMap`, `linear_map`.

**R30.** `readme.md`: update the `skoro-graph`, `skoro-model`, `skoro-linalg`, and
`skoro-normal` bullets to describe bindings, roles, `Factor`, and the Fisher root. Remove every
mention of `FactorRecord`, `ResidualRegistry`, `add_gaussian_quantity`.

---

## 4. Mathematics (implement exactly)

### 4.1 The Gaussian Fisher root over `[x, μ, L]`

Let `d` be the manifold dimension of `x` and `μ`, `d_L = d(d+1)/2` the dimension of `L`,
`W = L⁻¹` (lower triangular), `packed(i, j) = i(i+1)/2 + j` for `j ≤ i`.

`C` has `d + d²` rows and `2d + d_L` columns:

```
C = [ W   −W   0   ]   rows 0 .. d          (block over x and μ)
    [ 0    0   C_L ]   rows d .. d + d²     (block over L)
```

`C_L` is `d² × d_L`. Its column `q = packed(a, b)` is `vec(S_q)` (row-major) with
`S_q = (G_q + G_qᵀ) / √2` and `G_q = W · E_q`, `E_q` the matrix with a single 1 at `(a, b)`.
Entrywise: `G_q[r][c] = W[r][a]` if `c == b` else `0`.

Why: over `(x, μ)` the expected information of `−½‖W(x ⊖ μ)‖²` is
`[I, −I]ᵀ WᵀW [I, −I]`, which the first block reproduces. Over `L`, with `Σ = LLᵀ`,
`F_LL[q][q'] = ½ tr(Σ⁻¹ ∂_qΣ Σ⁻¹ ∂_{q'}Σ)`, and since `Σ⁻¹ ∂_qΣ = L⁻ᵀ (G_q + G_qᵀ) Lᵀ`, the trace
equals `tr((G_q + G_qᵀ)(G_{q'} + G_{q'}ᵀ))`, so `C_Lᵀ C_L = F_LL`. Cross blocks between
`(x, μ)` and `L` vanish in expectation (they are linear in the whitened residual).

**Construction of `K` (`gaussian_root_matrix(d)`)**, with `c_cols = 2d + d_L`,
`c_rows = d + d²`, `rows = c_rows · c_cols`, `cols = d_L`, `K = zeros(rows × cols)` row-major,
`cidx(r, c) = r · c_cols + c`:

```
for i in 0..d:
  for j in 0..=i:
    K[cidx(i, j),     packed(i, j)] += 1.0
    K[cidx(i, d + j), packed(i, j)] += -1.0
s = 1 / sqrt(2)
for a in 0..d:
  for b in 0..=a:
    q = packed(a, b)
    for r in 0..d:
      for c in 0..d:
        row = cidx(d + r*d + c, 2*d + q)
        if c == b && a <= r: K[row, packed(r, a)] += s     // G_q[r][c]  = W[r][a]
        if r == b && a <= c: K[row, packed(c, a)] += s     // G_qᵀ[r][c] = W[c][a]
```

Return `(rows, cols, K)`.

### 4.2 Host references used by the tests

Let `L` be the dense lower-triangular `d × d` matrix unpacked from a `Tril` value.

- `Σ = L Lᵀ`; `Σ⁻¹` by any small dense inverse (write a 2×2 or 3×3 inverse in the test, or
  use `skoro_dense::factor::spd_solve` against the identity).
- `F_LL[q][q']` for `q = packed(a,b)`, `q' = packed(a',b')`:
  `dΣ_q = E_q Lᵀ + L E_qᵀ`; `F_LL[q][q'] = ½ · trace(Σ⁻¹ · dΣ_q · Σ⁻¹ · dΣ_{q'})`.
- Score covariance (second Bartlett identity): with `x ~ N(μ, Σ)` sampled `N` times, the
  gradient `g_i` of `log p` with respect to `L`'s tangent coordinates satisfies
  `mean_i(g_i g_iᵀ) → F_LL`.

---

## 5. Acceptance tests

Every test below must exist with the given name and assertion. Use `NdArrayBackend<f64, i32>`
in one binary per file and never mix `f32` and `f64` backends in one test executable. Build
tensors from `TensorData`, not `Tensor::from_floats`. Tolerances are absolute unless stated.

### 5.1 `crates/skoro-graph/tests/graph_mutation.rs` (add a `mod bindings`)

| Test | Setup | Assert |
|---|---|---|
| `bind_refuses_a_missing_endpoint` | one constant `a` | `bind(a, Role("r"), fresh_id)` is `Err(NodeNotFound)`; `bind(fresh_id, .., a)` likewise |
| `bind_is_readable_in_order` | constants `a, b, c` | after `bind(a,R1,b)`, `bind(a,R2,c)`, `bind(a,R1,c)`: `bindings_of(a) == [(R1,b),(R2,c),(R1,c)]`, `bound(a,R1) == [b,c]`, `referrers_of(b) == [a]` |
| `remove_refuses_a_bound_target` | `bind(a, R, b)` | `remove(&[b])` is `Err(NodeStillBound { id: b, by: a })`; `remove(&[a, b])` is `Ok` |
| `remove_source_clears_the_reference` | `bind(a, R, b)` | after `remove(&[a])`, `referrers_of(b)` is empty and `remove(&[b])` is `Ok` |
| `remove_closure_takes_referrers` | `bind(a, R, b)` | `remove_closure(&[b])` removes both `a` and `b` |
| `validate_indices_holds_after_binding_mutations` | the sequences above | `validate_indices()` is `Ok` after every step |
| `terms_classify_proposal_and_weight` | stochastic `x` with a density; observed `y` with a density; a free-standing likelihood | `terms()` has kinds `[Proposal, Weight, Weight]` in marking order; after `observe(x)` the first becomes `Weight` |
| `log_density_terms_sums_only_the_named_terms` | two terms | `log_density_terms(&[t1]) + log_density_terms(&[t2]) == log_density()` within `1e-12`; `log_density_terms(&[mapped_id])` is `Err(NotLogLikelihood)` |

### 5.2 `crates/skoro-linalg/tests/linear_map.rs`

| Test | Assert |
|---|---|
| `forward_matches_the_host_product` | for a random `K: 3×2` and `x: [batch 2, 2]`, output equals `x Kᵀ` within `1e-12` |
| `jacobian_is_the_matrix` | `jacobian_vjp(inputs=[x], outputs=[y])` equals `K` for every batch row, and equals `jacobian_jvp` |
| `hvp_is_zero` | `evaluate_hvp` along any tangent returns second-order cotangents all zero |
| `shape_mismatch_is_reported` | `linear_map` with `out` of the wrong dimension is `Err(ShapeMismatch)` |

### 5.3 `crates/skoro-model/tests/bindings.rs` (replaces `tests/residual.rs`)

Use a local test distribution whose `factor` returns
`Factor::new(register_map1(Real, LiteralScalarMultiplication::new(2.0), x)?).bind(RESIDUAL, &some_mapped_node)`.

| Test | Assert |
|---|---|
| `add_quantity_binds_the_arguments` | `bound(density, ARGUMENT) == [x, parents...]` in that order |
| `add_quantity_binds_what_the_factor_declared` | `bound(density, RESIDUAL) == [residual]` |
| `a_bound_node_survives_cleanup` | drop the residual's `Q`; `cleanup()`; the residual is still in the graph |
| `remove_refuses_a_bound_node` | `remove(&[residual])` is `Err(NodeStillBound { id: residual, by: density })` |
| `remove_closure_takes_the_term_with_its_binding` | `remove_closure(&[residual])`: residual and density gone; `log_density == 0` |
| `demotion_sweeps_the_bindings_with_the_density` | `demote_to_parameter(&x)`; residual (handle dropped) is gone; `factors()` is empty |
| `add_factor_marks_and_binds` | `add_factor(Factor::new(term).bind(RESIDUAL,&r), &[a, b])`: `factors()` has one entry with `scored == None`, `kind == Weight`, `arguments == [a, b]`; `bound(term, RESIDUAL) == [r]` |
| `add_factor_needs_an_argument` | `add_factor(.., &[])` is `Err(ShapeMismatch { expected: 1, found: 0 })` |
| `factors_are_in_term_order` | three terms added in order appear in that order |
| `residual_layout_follows_term_order_and_dimension` | two terms with residuals of dims 2 and 3: `residual_layout(&client).total() == 5`, ids in order |

Keep the existing compile-fail case `tests/compile_fail/remove_closure_is_not_a_client_op.rs`.

### 5.4 `crates/skoro-model/tests/terms.rs`

| Test | Assert |
|---|---|
| `proposal_plus_weight_is_the_log_density` | model with a stochastic `x`, an observed `y`, and one `add_factor` term: `log_proposal + log_weight == log_density` within `1e-12` |
| `observing_moves_a_term_to_the_weight` | before `observe(x)` the `x` term is `Proposal`; after, `Weight` |

### 5.5 `crates/skoro-model/tests/jacobian.rs` (add)

| Test | Assert |
|---|---|
| `a_leaf_row_that_is_a_column_is_the_identity` | outputs `[u, x]` with `x` a parameter in `inputs`: rows of `x` equal the identity on `x`'s columns and zero elsewhere, on both paths |
| `a_leaf_row_that_is_not_a_column_is_zero` | outputs `[u, c]` with `c` a constant not in `inputs`: rows of `c` are zero on both paths, no error |

### 5.6 `crates/skoro-model/tests/teardown.rs`

Update `LazyDistr` to implement `factor` (returning `Factor::new(density)` with no bindings).
Every existing test in the file must still pass unchanged in intent. Remove the mention of
`NlsStructure` at line 11.

### 5.7 `crates/skoro-normal/tests/information.rs` (new)

| Test | Setup | Assert |
|---|---|---|
| `observed_gaussian_information_is_h_transpose_precision_h` | `θ: Parameter EuclideanD(2)`; `μ = log(A·θ + c)` with `A` a constant `EuclideanMatrixD(3,2)` and `c` a constant `EuclideanD(3)` chosen so `A θ + c > 0`; `L` a constant non-diagonal `Tril(3)`; `y = add_quantity(Normal::new(μ, L))`, then `observe(y)`; `free = [θ]` | `information(free)` equals `Hᵀ Σ⁻¹ H` within `1e-9`, where `H = jacobian_vjp(inputs=free, outputs=[μ])` and `Σ⁻¹` from `L` on the host; `information_root` has shape `[1, 3 + 9, 2]` |
| `gaussian_prior_information_is_the_precision` | constants `μ`, `L` (non-diagonal `Tril(2)`); `x = add_quantity(Normal::new(μ, L))` unobserved; `free = [x]` | `information(free) == Σ⁻¹` within `1e-9` |
| `covariance_information_matches_the_trace_formula` | `L: Parameter Tril(2)` at `L0 = [[1.5, 0], [-0.4, 0.8]]`; constant `μ`; `x` observed at any value; `free = [L]` | `information(free)` equals the host `F_LL` of §4.2 within `1e-9` (a `3×3` matrix) |
| `covariance_information_matches_the_score_covariance` | as above but `x` unobserved; seed the backend if it offers `Backend::seed` (optional; the tolerance is wide enough without it); `N = 40_000`; `generate_from(points{L: L0 repeated N}, N)`; read the sampled `x` from the returned `variables`; feed `x` and `L0` (batch `N`) into `log_density_ord1`; average the outer products of the `L` cotangents over the batch | every entry within `0.05 · max(1, |F_LL|)` of the host `F_LL` |
| `information_is_the_gram_of_the_root` | any model above | `information == information_rootᵀ · information_root` within `1e-12` |
| `se3_root_is_sized_by_dimension` | `Normal` over `SE3` with `Tril(6)` | the `FISHER_ROOT` node's quantity type is `EuclideanMatrixD(42, 33)`; `bound(density, ARGUMENT)` has layout total `6 + 6 + 21 = 33` |
| `a_constant_covariance_folds_the_root` | constant `L` | the root node is `Node::Mapped` with `value.is_some()`; with `L` a parameter, `value.is_none()` |
| `a_mismatched_covariance_is_refused` | `Normal::new(μ: EuclideanD(3), L: Tril(2))` then `add_quantity` | `Err(ShapeMismatch { expected: 3, found: 2 })` |

### 5.8 `crates/skoro-normal/tests/{structure,retire,facade,window,demote}.rs`

Rewrite every use of `add_gaussian_quantity`, `FactorRecord`, `factors()`, `residual_layout()`,
`gaussian_blocks()` to the new API (`add_quantity(Normal::new(..))`, `factors()` → `FactorInfo`,
`bound(density, RESIDUAL)`, `least_squares::residual_layout(&client)`, `GaussianBlock::all(&client)`).
The following behaviours must remain asserted:

- `structure.rs`: `whitened_residual` is the `r` in `−log p = ½‖r‖² + n` for `Real`,
  `EuclideanD` (non-diagonal `L`), and `SE3` (residual width 6); the builder binds the residual,
  location, whitening, and normalizer parent it built; the bound density is the model's own term.
- `retire.rs`: demoting a Gaussian quantity removes its term from `factors()`, its rows from
  `residual_layout`, and (after cleanup) its residual from the graph; `remove_closure(&[x])`
  takes the factor and its bindings; client-tier `remove` of a bound residual is
  `Err(NodeStillBound)`.
- `window.rs`: the sliding window grows and retires without leaking (`graph_node_count`
  bounded), using `add_quantity` and `add_joint_factor`.
- `facade.rs` / `demote.rs`: adapt mechanically.

### 5.9 `crates/skoro/tests/stacked_tangent.rs`

Adapt the `records` and `retirement` modules to bindings (`FactorInfo.arguments == ids`,
`bound(density, RESIDUAL)` is the residual). Add:

| Test | Assert |
|---|---|
| `joint_information_is_the_precision` | for the existing `coupled_precision()` fixture over `SE3 + EuclideanD(3)`, `information(free = the two ids)` equals `Λ` (the 9×9 precision) within `1e-8` |

---

## 6. Phases and checkpoints

Do the phases in order. Each checkpoint must pass before the next phase starts.

**Phase 1 (graph, additive).** R1, R5–R11 and tests 5.1.
Checkpoint: `cargo test -p skoro-graph -p skoro-core` passes; `cargo build --workspace` passes
(nothing else references the removed error yet except `skoro-model`, which you patch by
replacing `NodeIsRecorded` with `NodeStillBound { id, by: id }` temporarily).

**Phase 2 (linalg, additive).** R2–R4 and tests 5.2.
Checkpoint: `cargo test -p skoro-linalg`.

**Phase 3 (model, additive).** R12, R13, R14's `Factor` struct (keep the old `log_density`
trait method for now), R15, R16's `add_factor`/`bind` on the client (not yet the new
`add_quantity`), R17, R19, R21, R22, R23, and tests 5.4, 5.5, 5.3's `add_factor_*` tests.
Checkpoint: `cargo test --workspace` passes with the record machinery still present.

**Phase 4a (the breaking step).** R14's trait change, R16's new `add_quantity`, R24–R26,
5.6, and adapt every `impl Distribution` in tests. `facade::add_gaussian_quantity` may
temporarily call `add_quantity` and still record.
Checkpoint: `cargo test --workspace`.

**Phase 4b (delete records).** R18, R20, R27, R28, R29, 5.3 (rest), 5.8, 5.9.
Checkpoint: `cargo test --workspace`; `grep -rn "FactorRecord\|ResidualRegistry\|record_factor\|residual_layout()\|add_gaussian_quantity\|NodeIsRecorded" crates readme.md` returns nothing.

**Phase 5 (Fisher tests and docs).** 5.7, R30.
Final checkpoint:

```
cargo build --workspace
cargo test --workspace
cargo clippy --workspace --all-targets -- -D warnings
cargo doc --workspace --no-deps
```

---

## 7. Out of scope (do not implement)

- A Fisher-scoring or Levenberg-Marquardt solver loop.
- Compiling a sub-program that stops at a set of intermediate nodes ("cut" compilation), a
  `support(id)` query, block-sparse Jacobian storage, sparse solvers, or a compiled-program cache.
- Lazily built bindings from a stored distribution object.
- Any change to sampling, `observe`, `stacked_tangent`, or the discrete kind.

---

## 8. Traps and conventions

- **One backend per test binary.** Mixing `NdArrayBackend<f32,_>` and `<f64,_>` in one test
  executable silently demotes the dtype.
- **`Tensor::from_floats` is `f32`-lossy** even on an `f64` backend. Use `TensorData`.
- **HVP state gating** (`.claude/skills/skoro-diff-operation/SKILL.md`): `VjpState` gates on
  `diffs`, `HvpState` on `tangents`; an `hvp` slot must be `None` whenever the `vjp` slot is.
- **`Q.client` is `pub(crate)`.** `Handle` must live in `skoro-model`. Distributions cannot add
  constants; that is why the Gaussian root is a `FixedLinearMap` holding its matrix.
- **Lock discipline in `add_quantity`.** The write lock must be released before calling
  `distr.factor(&x)` and re-taken afterwards, or the call deadlocks.
- **A `Factor` must stay alive until its bindings are recorded.** Its `Handle`s are the roots
  that keep the residual and root nodes from being swept in between.
- **Observed and constant nodes are never differentiated.** Their Jacobian rows and columns are
  zero; that is correct, not an error.
- **Layouts are by manifold dimension**, never storage. `SE3` is 6 wide.
- **`cleanup` runs only on removal, demotion, or an explicit call.** A test that drops a handle
  and expects the node gone must call `cleanup()`.
- **Bindings are not data edges.** The compiler must never read them; the collector and `remove`
  must always treat them as references.
