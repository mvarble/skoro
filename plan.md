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

The parts below are in dependency order. Part A must land first: everything in Part B compiles
against it.

> **One item in Part A is a true blocker rather than a nicety.** `vjp_forward` panics when a
> continuous leaf receives no tape write, and `jacobian_vjp` reaches that case routinely. See A.3.

## On breaking changes

The library has no external users, so a breaking change costs a migration and nothing else. That is
the cheapest this will ever be, and several interfaces are worse than they need to be only because
fixing them would move a signature.

**Every item below is labelled — 🟢 additive, 🔴 breaking — with the migration cost stated.** Where an
interface would be meaningfully better if it were allowed to break, that is called out even when the
change is not taken, so the choice is visible rather than silent.

The bar applied: a breaking change is taken when it removes a panic from a public path, closes a
capability leak, or fixes an asymmetry that will otherwise be copied. It is *not* taken merely for
tidiness, because each one still costs review.

Six of the nine Part A sections carry breaking changes. Most are zero- or one-call-site; two — the
`Graph` field encapsulation (A.5) and `Q.client` (A.7) — are real work, and they are the two the
design actually rests on. A short list of breaking changes **not** taken, with reasons, closes the
document.

---

# Part A — Changes to structures that already exist

Bottom-up by crate. Each entry says what changes, which document asks for it, and whether it is
additive or breaking.

## A.1 `skoro-core/src/error.rs` — 🟢 additive

Six new variants. The enum is `#[non_exhaustive]` (`error.rs:24`), so this is additive.

Three constraints govern what may be written here, all discovered by reading the file rather than
assumed:

- It does **not** use `thiserror`. `Display` is a hand-written match at `error.rs:84`, and **every new
  variant needs its own arm there** or the build fails.
- The enum derives `Debug, Clone, PartialEq, Eq`. **No variant may carry an `f32` or `f64`** — they
  are not `Eq`. This is why `NotPositiveDefinite` below carries no pivot value and
  `NormalizerNotConstant` carries no magnitude.
- House style: lowercase message, no trailing period, ids rendered `{id:?}`.

*From `flattened-tangent-spaces.md` §3, `estimator-linear-algebra.md` §7, `model-structure-bounds.md` §5.*

```rust
// crates/skoro-core/src/error.rs — added to `pub enum SkoroError`

    /// An id appeared twice while building a `TangentLayout`.
    DuplicateLayoutId { id: Id },

    /// A layout was asked about an id it does not contain.
    IdNotInLayout { id: Id },

    /// A flattened tensor's dimension disagrees with the layout that must interpret it.
    LayoutDimensionMismatch { expected: usize, found: usize },

    /// A Cholesky factorization was attempted on a matrix that is not positive definite.
    NotPositiveDefinite { rows: usize },

    /// A host dense operation received a batch dimension other than 1.
    BatchNotSingleton { found: usize },

    /// A factor's normalizer depends on a quantity being optimized, so a
    /// sum-of-squares objective is not the model's negative log-density.
    NormalizerNotConstant { density: Id, parent: Id },
```

```rust
// the matching arms in `impl fmt::Display for SkoroError` (error.rs:84)

Self::DuplicateLayoutId { id } =>
    write!(f, "id {id:?} appears more than once in the tangent layout"),
Self::IdNotInLayout { id } =>
    write!(f, "id {id:?} is not present in the tangent layout"),
Self::LayoutDimensionMismatch { expected, found } =>
    write!(f, "flattened tangent has the wrong dimension; expected {expected}, found {found}"),
Self::NotPositiveDefinite { rows } =>
    write!(f, "a {rows}x{rows} matrix is not positive definite"),
Self::BatchNotSingleton { found } =>
    write!(f, "host dense linear algebra requires batch size 1, found {found}"),
Self::NormalizerNotConstant { density, parent } =>
    write!(f, "log-density {density:?} has a normalizer depending on {parent:?}, which is being \
               optimized; the objective is not a sum of squares"),
```

### Why `LayoutDimensionMismatch` is a third variant and not a reuse

Two near-neighbours already exist and neither fits:

- `ShapeMismatch { expected, found }` (`error.rs:81`) counts **elements of supplied data** — it is
  raised when a user packs a host value of the wrong length.
- `InconsistentFrameCount { expected, found }` (`error.rs:65`) counts **frames** across quantities in
  one differentiation pass.

`LayoutDimensionMismatch` counts **summed manifold dimensions** and is raised when a flat tensor and
a layout disagree. Collapsing it into `ShapeMismatch` would make the three indistinguishable in a
`match`, which is the thing an estimator most needs to tell apart when a layout is wrong.

## A.2 `skoro-core/src/id.rs` — 🟢 additive

`Id` derives `Debug, Clone, Copy, PartialEq, Eq, Hash` (`id.rs:4-5`) and **has no ordering**.
`TangentLayout::from_ids_sorted` needs one.

*From `flattened-tangent-spaces.md` §3.*

```rust
// crates/skoro-core/src/id.rs:4
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct Id(u64);
```

Sorting by `*id` through the existing `Deref<Target = u64>` (`id.rs:49-54`) would also work and needs
no change at all. The derive is preferred because `from_ids_sorted` is a public convenience and
`ids.sort()` reads better than `ids.sort_by_key(|i| **i)` at every call site.

## A.3 `skoro-graph/src/compiler.rs` — 🔴 breaking (0 call sites)

**The blocker.** Two changes, one of which will otherwise crash phase 1 on contact.

*From `flattened-tangent-spaces.md` §5, "A defect to fix along the way".*

`vjp_forward` harvests cotangents from `Parameter` and `Sampler` steps and panics if a continuous leaf
in the compiled subgraph received no tape write:

```rust
// compiler.rs:334 — today
panic!("Point should have a cotangent vector.")
```

`jacobian_vjp` reaches this whenever a layout column touches the targets only through a
`forward_only` or `constant_inputs` path — an ordinary occurrence, not a corner case: any free
parameter that happens not to influence the particular residuals being differentiated. The fix is to
default to zeros, which is the same rule `TangentLayout::flatten_*` already uses for ids missing from
a map (§4 there), so the two agree:

```rust
// compiler.rs — replacing the panic
//
// A leaf that does not reach the targets has zero derivative with respect to them.
// Returning zeros rather than panicking makes that an answer instead of a crash.
let cotangent = tape_cotangent
    .unwrap_or_else(|| Tensor::zeros([batch_size, dimension, frames], device));
```

*From `sequential-estimation.md` §3.*

The second change is a deletion. `ForwardOutput.observations` is declared at `compiler.rs:93`, is
**never written and never read** anywhere in the workspace, and is actively misleading: observed
nodes lower to `CompiledStep::Constant` (`graph.rs:1342-1352`), so their values arrive in
`constants`. Anyone reaching for `observations` to find an innovation input gets an empty map.

```rust
// crates/skoro-graph/src/compiler.rs:89
#[derive(Debug, Clone, Default)]
pub struct ForwardOutput<B: Backend> {
    pub variables: PointMap<B>,
    pub constants: PointMap<B>,
    pub evaluations: PointMap<B>,
    // `observations` removed — observed nodes land in `constants`.
}
```

> **Breaking, and worth it at zero cost.** Removing a `pub` field from a `pub` struct is a breaking
> change in principle; in practice the field has no readers anywhere, so nothing migrates. Left in
> place it is a trap that costs someone an afternoon: the name promises exactly what an innovation
> computation wants, and it silently yields an empty map. Deleting it now is free; deleting it once
> two estimator crates have grown around the confusion is not.

## A.4 `skoro-graph/src/point_map.rs` — 🟢 additive, plus 🔴 one asymmetry fix (1 call site)

`PointMap<B>` (`point_map.rs:21`) is a struct holding parallel `points` and `quantity_types` maps.
Its existing `retract_mut` (`:142`) requires a tangent for **every** point in the map and returns
`MissingPoint` otherwise. An optimizer needs the opposite: retract the layout's ids and leave
constants and nuisance quantities alone.

*From `flattened-tangent-spaces.md` §4.*

```rust
// crates/skoro-graph/src/point_map.rs — new methods on `impl<B: Backend> PointMap<B>`

impl<B: Backend> PointMap<B> {
    /// Retract only the quantities named by `layout`, leaving every other point untouched.
    ///
    /// `delta` is `[batch, layout.total()]`; each block is unflattened and applied through
    /// its own manifold's retraction. Errors `LayoutDimensionMismatch` if the widths
    /// disagree, `IdNotInLayout` never — an id in the layout but absent from the map is
    /// `MissingPoint`, matching `retract_mut`.
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

The manifold for each block is reached exactly as the existing private helper does
(`manifold_point`, `point_map.rs:126`), via `quantity_type.as_manifold()` to a
`&dyn DynManifold<B>`. `DynManifold` is sealed with a blanket impl
(`skoro-riemann/src/base.rs:36-51`), so it is consumed, never implemented.

### Take the asymmetry fix while here

```rust
// crates/skoro-graph/src/point_map.rs:169 — was `others: PointMap<B>`
pub fn inverse_retract(&self, others: &PointMap<B>) -> SkoroResult<HashMap<Id, TangentVector<B>>>
```

> **Breaking, one call site, and the kind of thing that spreads.** `retract` takes `&self` and
> borrows; `inverse_retract` consumes `others` by value for no reason — it only reads. Callers must
> clone a whole `PointMap` (two `HashMap`s plus an `Arc` bump per entry) to keep their own copy. An
> optimizer computing residuals against a reference point does exactly that, every iteration. Fixing
> it now costs one signature; leaving it means every future estimator either clones or restructures
> around the borrow.

## A.5 `skoro-graph/src/graph.rs` — 🔴 breaking, and the one that makes the rest mean anything

Two changes. Neither is required to make phase 1 *compile*; both are required for its guarantees to
be worth stating.

### Every field of `Graph` is `pub`, and so is `ModelServer.graph`

```rust
// crates/skoro-graph/src/graph.rs:34 — today
pub struct Graph<B: Backend> {
    pub nodes: HashMap<Id, Node<B>>,
    pub children: HashMap<Id, HashSet<Id>>,
    pub log_likelihoods: Vec<Id>,
    pub log_density_to_stochastic: HashMap<Id, Id>,
    pub stochastic_to_log_density: HashMap<Id, Id>,
    id_generator: IdGenerator,
}
```

Four mutable indices with a hand-maintained invariant between them — `mark_likelihood_impl`
(`:254-335`) writes all four in concert — exposed for arbitrary outside mutation. Anyone holding a
`&mut Graph` can push onto `log_likelihoods` without a corresponding `LogDensity` node, or remove a
node while leaving its id in two maps.

This is the leak that makes every other guarantee provisional. `ModelServer.graph` is `pub`
(`model.rs:45`) and `ModelClient::get_server` (`model.rs:481`) hands out the `Arc<RwLock<..>>`, so the
route from *any* client to arbitrary graph mutation is two field accesses. A certificate asserting
"every factor is least-squares shaped" is only as good as the impossibility of appending a fifth
factor behind its back.

```rust
// crates/skoro-graph/src/graph.rs
pub struct Graph<B: Backend> {
    nodes: HashMap<Id, Node<B>>,
    children: HashMap<Id, HashSet<Id>>,
    log_likelihoods: Vec<Id>,
    log_density_to_stochastic: HashMap<Id, Id>,
    stochastic_to_log_density: HashMap<Id, Id>,
    id_generator: IdGenerator,
}

impl<B: Backend> Graph<B> {
    // Read-only accessors replacing the public fields.
    pub fn node(&self, id: Id) -> Option<&Node<B>> { todo!() }
    pub fn nodes(&self) -> impl Iterator<Item = (&Id, &Node<B>)> { todo!() }
    pub fn node_count(&self) -> usize { todo!() }
    pub fn children_of(&self, id: Id) -> Option<&HashSet<Id>> { todo!() }
    pub fn log_likelihoods(&self) -> &[Id] { todo!() }
    pub fn density_of(&self, stochastic: Id) -> Option<Id> { todo!() }
    pub fn stochastic_of(&self, density: Id) -> Option<Id> { todo!() }
}
```

> **This is the largest migration in phase 1 and the one most worth doing now.** Mutation already
> goes through methods — `add_*`, `observe`, `mark_*`, `update_stochastic`, `demote_to_parameter`,
> `remove` — so the fields are read directly far more than they are written. `ModelServer.graph`
> becomes private with `graph(&self) -> &Graph<B>`, and `get_server` keeps working for readers while
> ceasing to be a mutation vector. Every later document depends on this: the certificate (B.3), the
> facade's invariant (B.5), and the bounded-growth assertions in `sequential-estimation.md` §12 all
> assume the four indices agree, and today nothing enforces that.

### `Graph::add_mapped` returns a bare `Id` and panics

```rust
// crates/skoro-graph/src/graph.rs:205-231 — today
pub fn add_mapped<K, T>(&mut self, ..) -> Id {
    // ...
    self.register_children(id, &parent_ids)
        .expect("internal: mapped node added with dangling parent");
}
```

Every sibling constructor returns `SkoroResult`. This one panics, under a comment asserting that a
dangling parent "indicates an internal bug rather than user error".

```rust
pub fn add_mapped<K, T>(&mut self, ..) -> SkoroResult<Id> { todo!() }
```

> **Breaking, and it removes a panic from a public path** — the bar this document set for taking one.
> The comment's premise is already shaky and becomes false once anything removes nodes: a `Q` held
> across a removal makes a dangling parent ordinary user error. Phase 1 does not remove nodes, so
> this is pre-emptive; it is included because the migration is mechanical now (`add_mapped` is
> reached through `map.rs`'s `register_mapN` helpers, which already return `SkoroResult`) and
> becomes a rewrite of every operator's plumbing later.

*Both changes relate to `graph-marginalization.md` §8, items 3 and 5.*

## A.6 `skoro-model/src/hash_map_builders.rs` — 🟢 additive

The tangent builders take a raw `Id` (`TensorMapBuilder::with_tensor`, `hash_map_builders.rs:53`)
while `InputMapBuilder::insert_data` takes a `&Q<B, T>`. The inconsistency shows up immediately when
seeding a Jacobian pass.

*From `flattened-tangent-spaces.md` §4.*

```rust
// crates/skoro-model/src/hash_map_builders.rs

impl<B: Backend, T> TensorMapBuilder<B, T> {
    /// Insert by quantity rather than by raw id, matching `InputMapBuilder::insert_data`.
    pub fn with_quantity<Qty>(self, q: &Q<B, Qty>, tensor: T) -> Self { todo!() }
}
```

## A.7 `skoro-model/src/q.rs` — 🔴 breaking (13 sites)

All three tiers of the unforgeability work. Every builder interface in Part B is shaped by whether
this holds.

*From `model-structure-bounds.md` §6.*

**Tier 1.** Delete `Q<B, Real>::mark_likelihood` (`q.rs:175-181`). It has **zero call sites** in the
workspace, so nothing migrates. It is deleted rather than deprecated because its existence lets any
holder of a real-valued quantity attach an arbitrary factor to a restricted model.

```rust
// crates/skoro-model/src/q.rs:175-181 — deleted entirely
//
// impl<B: Backend> Q<B, Real> {
//     pub fn mark_likelihood(&self) -> SkoroResult<()> { ... }
// }
```

**Tier 2.** The `client` field becomes private, replaced by a narrow accessor. This is where the
guarantee actually lives: today `gaussian.add_parameter(..).client.add_quantity(..)` reaches the
unrestricted surface through any quantity a facade returned, and `client.server_id` reaches
`ModelClient::from_id`.

```rust
// crates/skoro-model/src/q.rs:21
pub struct Q<B: Backend, T> {
    id: Id,                          // was `pub`
    client: ModelClient<B>,          // was `pub`
    pub(crate) guard: Arc<NodeDropGuard>,
    _phantom: PhantomData<T>,
}

impl<B: Backend, T> Q<B, T> {
    pub fn id(&self) -> Id { todo!() }
}

impl<B: Backend, T> Q<B, T> {
    /// The mapping surface, and nothing else.
    ///
    /// Operator sugar and the `*Ops` extension traits build mapped nodes through this.
    /// It deliberately does not expose `add_quantity`, `mark_likelihood`, `get_server`
    /// or `server_id`, so a quantity handed out by a restricted facade cannot be used
    /// to widen that facade.
    pub fn mapping(&self) -> &impl MappingOps<B> { todo!() }
}

/// The subset of the client needed to build mapped nodes.
pub trait MappingOps<B: Backend> {
    fn add_mapped<K, T>(
        &self,
        quantity_type: T,
        op: Arc<dyn GraphOperation<B>>,
        parent_ids: Vec<Id>,
    ) -> SkoroResult<Q<B, T>>
    where
        K: Kind<B>,
        T: QuantityType<B, K>;
}
```

**Migration.** Thirteen call sites across four crates, all of them internal operator plumbing:

| file | sites |
|---|---|
| `skoro-model/src/map.rs` | `:16`, `:32`, `:35`, `:73`, `:78` |
| `skoro-so3/src/ops/exp/cube.rs` | `:93`, `:136`, `:182`, `:236` |
| `skoro-so3/src/ops/log/cube.rs` | `:98`, `:141`, `:187`, `:241` |
| `skoro-linalg/src/vector/cross_product/cube.rs` | `:215`, `:259` |
| `skoro-normal/src/facade.rs` | the delegating body |

`Q`'s `Debug`, `PartialEq` and `Hash` impls (`q.rs:34`, `:52`, `:59`) read `client.server_id` and
move inside the same module, so they are unaffected.

`Q.id` goes private in the same pass. It is read widely, but only ever to pass into a graph call, so
the migration is `q.id` → `q.id()` and nothing more. Making it private now is what lets `Q` change
representation later — a tick-scoped generation counter for the graveyard, say — without another
breaking release.

**Tier 3, now taken.** `ModelClient::get_server` (`model.rs:481`) is narrowed and `from_id`
(`model.rs:457`) kept:

```rust
impl<B: Backend> ModelClient<B> {
    /// Read-only access. `ModelServer::graph()` returns `&Graph<B>` (A.5), so this is no
    /// longer a mutation vector.
    pub fn server_call<O, F: Fn(&ModelServer<B>) -> O>(&self, f: F) -> O { todo!() }

    // `get_server(&self) -> Arc<RwLock<ModelServer<B>>>` — removed.
}
```

> **Deferring Tier 3 was the wrong call, and A.5 is why.** I had planned to keep `get_server` on the
> grounds that a private `Q.client` removes the route *from a quantity*. But `ModelClient` is
> constructible directly, and with `ModelServer.graph` public the pair is a complete bypass of every
> facade. Once A.5 makes `graph` private, `get_server` is merely redundant with `server_call` — so
> removing it costs nothing and closes the last hole. `from_id` stays: it recovers a client by
> server id, which after A.5 confers only what `ModelClientOps` already allows, and
> `skoro-model/tests/plumbing.rs:81-90` exercises it deliberately.

## A.8 `skoro-model/src/client_ops.rs` — 🔴 breaking (2 call sites)

`ModelClientOps` (`client_ops.rs:46`) has 22 methods and deliberately omits the stochastic growth
path, which is what makes a restricting facade possible at all. That omission now needs a home, so a
builder can be generic over "can grow this model".

*From `sequential-estimation.md` §5, `model-structure-bounds.md` §6.*

```rust
// crates/skoro-model/src/client_ops.rs

/// Capability: this client can add stochastic quantities and free-standing factors.
///
/// Separate from `ModelClientOps` rather than folded into it, because
/// `GaussianModelClient` implements that trait precisely in order to *withhold*
/// this surface (`skoro-normal/src/facade.rs:196-199`).
pub trait ModelBuilderOps<B: Backend>: ModelClientOps<B> {
    fn add_quantity<K, D>(&self, distr: D) -> SkoroResult<Q<B, D::Qty>>
    where
        K: Kind<B>,
        D: Distribution<B, K>;

    fn update_quantity<K, D>(&self, tag: &Q<B, D::Qty>, distr: D) -> SkoroResult<()>
    where
        K: Kind<B>,
        D: Distribution<B, K>;

    fn mark_likelihood(&self, likelihood: &Q<B, Real>) -> SkoroResult<()>;
}
```

Implemented by `ModelClient<B>` only. The methods are generic over `D: Distribution`, so the trait is
not dyn-compatible — which is fine, since builders are monomorphized.

> **Take the inherent methods away, don't leave both paths.** `add_quantity` (`model.rs:533`),
> `update_quantity` (`:615`) and `mark_likelihood` (`:592`) are inherent methods on `ModelClient`
> today. If they stay inherent *and* appear on the trait, inherent-method resolution wins at every
> call site and the trait is decoration — a reader cannot tell which surface a builder was written
> against. Moving them costs two call sites (`skoro-normal/src/facade.rs:103` and
> `skoro-normal/tests/demote.rs:208`) plus a `use skoro_model::ModelBuilderOps` where they are
> called. Doing this is what makes "a facade withholds growth" a checkable statement instead of a
> convention.

## A.9 `skoro-normal/src/distribution.rs` and `skoro-model/src/distribution.rs` — 🔴 breaking (3 impls)

Expose the residual that `log_density` already builds and throws away — and, now that breaking is
permitted, remove the panic that sits beside it.

*From `flattened-tangent-spaces.md` §7, `model-structure-bounds.md` §4.*

Today (`distribution.rs:168-178`) `white_tangent` is a local binding, consumed immediately by
`.norm_squared()`, its `Id` never returned to any caller. The change is to name it:

```rust
// crates/skoro-normal/src/distribution.rs

impl<B: Backend, M: ContinuousQuantity<B>> Normal<B, M> {
    /// The whitened residual `r(x) = L⁻¹ (x ⊖ μ)`.
    ///
    /// `−log p(x) = ½‖r(x)‖² + n(θ)`, where the normalizer `n` is the `log_det` term
    /// and the constant. Least-squares consumers minimize the first term only, which is
    /// valid exactly when `n` does not vary with the optimization variables — see the
    /// precondition in Part B, `skoro-levenberg-marquardt`.
    pub fn whitened_residual(&self, x: &Q<B, M>) -> SkoroResult<Q<B, EuclideanD>> { todo!() }
}
```

The return type is `Q<B, EuclideanD>` and needs **no bound on `M` beyond the one `Normal` already
carries**: for a generic `M`, `inverse_retract` resolves only through the blanket
`AssociatedTangentVectorBuilder<EuclideanD, B>` impl (`skoro-model/src/ops/base.rs:18`), so the
tangent type is forced rather than chosen.

### Make `Distribution::log_density` fallible

The reason `whitened_residual` returning `SkoroResult` looked awkward is that its only caller cannot
propagate: `log_density` returns a bare `Q<B, Real>` and therefore ends in

```rust
.expect("internal: Normal covariance factor and tangent dimensions should match")
```

(`distribution.rs:173-175`). That is a panic on a public path, reachable whenever a covariance
factor's dimension disagrees with its mean's tangent dimension — a modelling error, not an internal
one.

```rust
// crates/skoro-model/src/distribution.rs:18
pub trait Distribution<B: Backend, K: Kind<B>> {
    type Qty: QuantityType<B, K>;
    type Sampler: Sampler<B, K>;

    fn quantity_type(&self) -> Self::Qty;
    fn sampler(&self) -> Self::Sampler;
    fn parent_ids(&self) -> &[Id];
    fn log_density(&self, x: &Q<B, Self::Qty>) -> SkoroResult<Q<B, Real>>;   // was `Q<B, Real>`
}
```

> **Three impls and two callers.** The impls are `Normal` (`skoro-normal/src/distribution.rs:144`)
> and two test doubles (`skoro-model/tests/plumbing.rs:61`,
> `skoro-model/tests/discrete_ops.rs:230`). The callers are `ModelClient::add_quantity`
> (`model.rs:553`) and `update_quantity` (`:627`), both of which already return `SkoroResult` and
> currently discard the possibility of failure. This is the whole blast radius, and it converts a
> panic into a `?`. Left alone, every future distribution inherits the same forced `expect`, and the
> asymmetry — a fallible `whitened_residual` next to an infallible `log_density` computing the same
> thing — is exactly the kind of thing that gets copied.

With this taken, `log_density` becomes `Ok(self.constant + -0.5 * self.whitened_residual(x)?.norm_squared() + log_det)`
and there is one code path rather than two.

---

# Part B — New code, by crate

## B.1 `skoro-graph` — new module `tangent_layout.rs`

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

### The sealed-trait design cannot work across crates

`model-structure-bounds.md` §2 and §6 specify `NlsStructure` as a **sealed** trait, on the argument
that a trait anyone can implement cannot carry an invariant. The argument is right and the mechanism
is not: sealing works by a private `Sealed` supertrait, so **only the defining crate can implement
the sealed trait**. But `NlsStructure` must live in `skoro-model` (it bounds `Distribution` and
`ModelClientOps`), while `GaussianModelClient` lives in `skoro-normal` — which could then never
implement it. The very variance the trait exists to provide is what sealing forbids.

**The fix moves the guarantee from the trait to the certificate's constructor.** `NlsStructure`
becomes an ordinary trait. `ResidualCertificate` is what cannot be forged, because entries enter it
only through

```rust
pub fn record<B, K, D>(&mut self, distr: &D, density: Id, residual: &Q<B, EuclideanD>)
    -> SkoroResult<()>
where D: WhitenedResidual<B>;
```

A hand-rolled implementor of `NlsStructure` must produce a `ResidualCertificate`, and the only
certificate it can produce is either empty or populated through the bounded `record`. The invariant
holds, and downstream crates can still implement the trait.

This is arguably the better design regardless: it puts the guarantee on the **data** the algorithm
consumes rather than on a marker beside it, which is the same move as "the certificate and the
algorithm's input are the same object".

**`docs/model-structure-bounds.md` §2 and §6 need a follow-up amendment** recording this. It is not
made in phase 1 — this document is the record until then.

### Two smaller notes

- **`GaussianStructure` ships without a consumer.** Phase 1 has no use for `blocks()`; only the EKF
  does. It is declared now because the refinement `GaussianStructure: NlsStructure` is the thing
  being validated, and declaring it later would mean revisiting the facade a second time.
- **Encapsulating `Graph` (A.5) turned out to be the load-bearing change, not `Q.client`.** The
  interfaces in B.3 and B.5 assert that a certificate cannot be bypassed. Writing them made clear
  that `Graph`'s five `pub` fields plus a `pub` `ModelServer.graph` make every such assertion
  provisional — a `&mut Graph` reached through `get_server` can append to `log_likelihoods`
  directly. `Q.client` closes the route from a *quantity*; A.5 closes the route from a *client*, and
  without it the rest is a convention. This only became visible once the certificate had a written
  signature to defend.

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
| 9 | **Cotangent defaulting.** A model with a free parameter that does not reach the residuals produces a zero column in `J` rather than panicking — the regression for [A.3](#a3-skoro-graphsrccompilerrs). | `flattened-tangent-spaces.md` §5 |

Run with `cargo test --workspace`. The wgpu suites additionally exercise the device path, and will
surface the `TrilInvert` host round-trip as a sync point.

---

## Summary of files touched

| | file | change | |
|---|---|---|---|
| A.1 | `crates/skoro-core/src/error.rs` | 6 variants + 6 `Display` arms | 🟢 |
| A.2 | `crates/skoro-core/src/id.rs` | derive `PartialOrd, Ord` | 🟢 |
| A.3 | `crates/skoro-graph/src/compiler.rs` | zero-default the VJP panic; drop `observations` | 🔴 0 sites — **blocker** |
| A.4 | `crates/skoro-graph/src/point_map.rs` | `retract_flat*`; `inverse_retract` by reference | 🟢 + 🔴 1 site |
| A.5 | `crates/skoro-graph/src/graph.rs` | privatize the 5 fields + `ModelServer.graph`; `add_mapped -> SkoroResult` | 🔴 **largest** |
| A.6 | `crates/skoro-model/src/hash_map_builders.rs` | `with_quantity` | 🟢 |
| A.7 | `crates/skoro-model/src/q.rs` | drop `mark_likelihood`; privatize `client` and `id`; drop `get_server` | 🔴 13 sites |
| A.8 | `crates/skoro-model/src/client_ops.rs` | `ModelBuilderOps`; move the 3 inherent methods onto it | 🔴 2 sites |
| A.9 | `skoro-model/src/distribution.rs`, `skoro-normal/src/distribution.rs` | `whitened_residual`; `log_density -> SkoroResult` | 🔴 3 impls |
| B.1 | `crates/skoro-graph/src/tangent_layout.rs` | new module | |
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
