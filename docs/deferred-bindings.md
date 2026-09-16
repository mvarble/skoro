# Deferred bindings — design, postponed

A complete design for building a factor's bindings lazily, on first read, instead of eagerly at
`add_quantity` time. **Nothing here is implemented, and none of it is scheduled.** It is kept
because the design is finished and the decision was "wait", not "no": §3.5.1 lists the triggers
that would reopen it, and §3.5.7 the four questions that would then need answers.

The section numbering is the original `3.5.x`, kept so the internal cross-references still
resolve. `R10`, `§5.7`, `§5.9` and `§7` point at the factor-bindings spec, which described
shipped code and was removed; recover it with
`git show 2948f6d:docs/superpowers/specs/2026-09-01-factor-bindings-and-fisher-root-spec.md`.

---

## 3.5.1 Is it still needed, now that the Gaussian roots are sparse?

Mostly for three kinds of case, not the one that motivated it. Now that `SparseLinearMap` backs the
Gaussian and joint roots, an eager Gaussian root costs one node, and a folded `C` only when `L` is
constant. Deferral becomes worth building when any of these holds:

1. **A family whose root is expensive to build:** a sampled or series-estimated information,
   an exact expected information on a Lie group (no closed form in general), a non-Gaussian family,
   or a root that needs its own factorization.
2. **A very large model** in which most terms are never read for information, so their root nodes and
   folded values are pure overhead.
3. **A future role that is costly to materialize,** such as sparsity structure or a precomputed
   Cholesky factor.

The design below is general for that reason: nothing in it is specific to `FISHER_ROOT`.

## 3.5.2 Goals and non-goals

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
- **No compiler change** (R10 still holds). Re-checked after the compiler work in
  [graph-compiler.md](graph-compiler.md), which added a compiled-program cache, a cut-aware
  traversal and a `Given` step kind: the compiler still never reads a binding, so this design
  is untouched by all three. The cache is the only part worth a second thought, and it keys on
  `(epoch, cuts, targets)` with every graph mutator bumping the epoch — so a binding realized
  lazily would invalidate like any other mutation, without this design having to know the cache
  exists.

Non-goals:

- Caching evaluated values. Deferral delays *building an expression*, and the built node is evaluated
  at points like any other. Nothing needs invalidating when values change.
- Automatically discarding a built root (a "forget" call). Possible later; see §3.5.9.

## 3.5.3 Graph data model (`crates/skoro-graph/src/binding.rs`)

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

## 3.5.4 Model API (`crates/skoro-model`)

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

## 3.5.5 Resolution protocol

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

## 3.5.6 Using it

`Normal::factor`, as it stands today:

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

## 3.5.7 Decisions for review

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

## 3.5.8 Phases and tests

| Phase | Scope | Tests |
| --- | --- | --- |
| D1 | `Target`, `bind_deferred`, `deferred`, `realize`, `validate` in `skoro-graph` | dependencies refuse `remove` (`NodeStillBound`); `remove_closure(dep)` takes the term and the entry; `validate_indices` holds after every mutation; `realize` returns `Stale` after the source is removed or already realized |
| D2 | `DeferredBinding`, `Factor::bind_deferred`, `Handle::typed`, `ModelClientOps::resolve`, collector marking of dependencies | nothing is built before `resolve` (node count and a build counter); a second `resolve` does not build; two threads resolving concurrently settle on one id and the loser is collected after `cleanup()`; resolving a term removed mid-build returns empty and leaks nothing; a dependency survives `cleanup()` with no handle |
| D3 | `Normal`/`JointNormal` defer their roots; `information_blocks` resolves | all §5.7 and §5.9 tests pass (tests that inspect the root via `bound` switch to `resolve`); adding N Gaussians adds exactly N fewer nodes than eager; the folded-root test observes folding at resolve time |
| D4 | Spec and docs: `FISHER_ROOT` contract ("may be deferred; read with `resolve`"), §7 updated, readme | `cargo doc` without new warnings |

Each phase is additive until D3, which is the only behavioural change: a root is absent from
`bound()` until it is resolved.

## 3.5.9 Alternatives considered

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

