# Sequential Estimation

A design for how time enters a Skoro model, who owns which piece of mutable state, and how a model
that grows with a data stream stays bounded.

Companion to `flattened-tangent-spaces.md`. That document specifies the *algebra* an EKF and a
Gauss-Newton iteration both consume — a stable ordering of tangent spaces, Jacobian assembly, a
dense SPD solve. This one specifies the *architecture around it*: what a "time step" is, what an
estimator is allowed to touch, and what must happen for a filter to run forever without the graph
growing forever. The graph-surgery primitives that the bounded-growth story rests on are specified
separately in `graph-marginalization.md`.

## 1. Motivation

Gauss-Newton can be stateless. Hand it a model, a point, and a layout, and it hands back a step;
run it again tomorrow on a bigger model and nothing carries over. An Extended Kalman Filter cannot.
It has a strict notion of causality — a prior, a prediction, an innovation, in that order — and it
carries a covariance from one step to the next.

In principle a filter could simply respect the DAG: each stochastic root is a prior, each path
between stochastic nodes induces a prediction, each path from a stochastic node to an observed one
with no stochastic node in between induces an innovation. That framing is appealing and it is not
wrong. It is also not enough, for one reason: it says nothing about **forgetting**. A model that
grows with the data stream becomes too expensive to estimate, and the DAG view offers no account of
which quantities stop being worth estimating or what replaces them.

So this document is about time and forgetting, and it defers the filter itself.

There is nothing to build on today. Grepping `time`, `chain`, `markov`, `marginal`, `temporal`,
`sequential`, `kalman`, `filter` across `skoro-graph`, `skoro-model`, and `skoro-core` returns only
prose. The only multi-instance axis in the workspace is the i.i.d. batch dimension.

> **Terminology trap.** This codebase uses "sequence" pervasively — `SequenceMutator`
> (`skoro-core/src/sequence.rs`), `CompiledSequence` (`skoro-graph/src/compiler.rs:50`),
> `SequenceSampler`, `SequencedDiffOperation` — to mean **the linear instruction tape of the
> compiled program, addressed by `usize` slot index**. It never means a time series. A grep for
> "sequence" gives exactly the wrong impression, and any new temporal type must avoid the word.

### Non-goals

No smoother. No data association beyond timestamp matching. No loop closure, no outlier rejection,
no adaptive noise estimation. Those are algorithms; this is the substrate they would sit on.

## 2. Three ownerships, one writer each

This is the organizing claim, and it answers the question that motivated the document — *when an
estimator wants to marginalize, does it mutate the model?*

| owner | owns | who mutates it |
|---|---|---|
| **Model** (`ModelServer<B>`) | graph structure; node identities; what depends on what | the Timeline, single-threaded |
| **Timeline** | which ticks exist; the retention policy; when a tick retires | the driver loop |
| **Estimator** | numerical state — the covariance `P`, the point map, linearization points | its own `step` |

> **An estimator never mutates graph structure. A covariance never lives in the graph.**

`P` is an object on the tangent space *at the current point*, shape `[batch, N, N]`, keyed by a
`TangentLayout` (`flattened-tangent-spaces.md:60-68`). It has no node, no `Id`, and no place in
`Graph.nodes`. Estimators are therefore emphatically **not** stateless, and that is the point: it is
precisely because no estimator owns structure that two of them can read the same model at once.

Section 6 qualifies this. Retirement requires numerics only an estimator can supply, so an estimator
does participate in structural change — but as the *supplier of a prior*, never as the agent of
mutation. The Timeline remains the only writer.

### Why single-writer is forced

It would be reasonable to read the workspace's `Arc<RwLock<ModelServer<B>>>`
(`skoro-model/src/model.rs:432`) as an invitation to build a model from several threads. It is not.

`ModelClient::add_quantity` (`model.rs:553-566`) deliberately releases the write lock in the middle
of the operation, with a comment saying why:

```rust
// maybe slow, but currently the price for ergonomic interface: we need to drop the server
// to add the log-density term (or else deadlock will occur).
drop(server);
let density = distr.log_density(&x);
```

`distr.log_density(&x)` builds mapped nodes *through the client*, which retakes the write lock. So
registering a stochastic node is **not atomic**, and a concurrent reader can observe a stochastic
node with no density attached. `update_quantity` (`model.rs:626-639`) repeats the dance.

Compounding it, every `read()`/`write()` is `.unwrap()`ed (`model.rs:472, 478`), so a panic while a
lock is held poisons the model and every subsequent client call panics.

> Single-writer model mutation is a **requirement of the existing code**, not a stylistic
> preference. §8 builds the concurrency story on it rather than around it.

## 3. The lag spectrum is one retention policy

Filtering, fixed-lag smoothing, and batch estimation differ in exactly one parameter: how many ticks
stay in the graph.

| policy | ticks retained | forgetting mechanism | graph surgery |
|---|---|---|---|
| `lag = 0` | 1 | the covariance `P` | **none** |
| `lag = L` | L+1 | re-root the blanket onto a joint prior | per tick |
| `lag = ∞` | all | none | none |

The endpoints are the interesting part.

### `lag = 0` — the filter *is* the marginalization

A plain EKF needs no marginalization machinery whatsoever, and it is worth saying loudly because it
is the cheapest path to a working filter.

One tick's worth of structure lives in the graph: a state leaf `x`, a process map `f(x)`, a
measurement map `h(x)`. Advancing does not extend the graph — it **overwrites the leaf slot** with
the evaluated value of the mapped node. The information about everything before the current step is
carried entirely in `P`, which is to say: a Kalman filter's covariance *is* the marginal prior over
its state, and propagating it *is* the marginalization. There is nothing left in the graph to
retire because nothing was ever added.

This also sidesteps the structural constraint from `flattened-tangent-spaces.md:294-303` for free —
Jacobian columns must be `Parameter`/`Stochastic` leaves and rows must be `Mapped` nodes, which is
exactly the shape of a one-tick template.

> **The state leaf should be a `Parameter`, not a `Stochastic`.**

This is worth being precise about, because "the filter *is* the marginalization" is a claim about
which entry points the filter uses, not about the model. A `Stochastic` leaf added through
`add_quantity` gets a density node marked into `log_likelihoods` (`graph.rs:290`), and that prior
factor would then coexist with `P`, which is supposed to carry the same information. Nothing
double-counts in practice — an EKF only ever calls `evaluate_ids_jvp` / `evaluate_ids_vjp`, never
`log_density` — but relying on that is relying on an estimator's discipline rather than on the
model being right.

A `Parameter` leaf removes the question. It has no distribution and contributes nothing to
`log_likelihoods`, and it is a first-class Jacobian column regardless: the VJP collection loop
matches `CompiledStep::Parameter` and `CompiledStep::Sampler` alike (`compiler.rs:354-357`), and
`non_op_forward` seeds its tangents and frames at `compiler.rs:536-545`. The prior lives in `P₀`
where it belongs.

This is also the sharper reason `lag = 0` stays specialized rather than being unified away. At
`lag > 0` a windowed estimator *does* evaluate `log_density`, so retirement must materialize a real
density node — which is exactly what `graph-marginalization.md` §5 is about. The two endpoints put
the prior in different places on purpose.

The two are consistent, and they converge more than they diverge. At `lag > 0` a correlated marginal
is carried by a likelihood factor over quantities that have been *demoted to `Parameter`s* — so the
window head ends up the same kind as the `lag = 0` state leaf. The difference is only whether a
factor supplies its density. Nothing is ever promoted into a stochastic node; the only kind change
in the design removes structure rather than inventing it.

### `lag = ∞` — batch

Nothing retires; the graph is the whole history. This is what Gauss-Newton over a fixed dataset
already does today, and it needs nothing from this document.

### `lag = L` — the interpolation

L+1 ticks are retained. When tick `k` falls off the back, it is retired: the stochastic nodes on its
blanket receive a joint marginal carrying the information it held, and its own nodes are removed. The
mechanism is in `graph-marginalization.md`; the policy — *when* — is the Timeline's.

At `L = 0` the blanket is the whole next state and the "joint prior" is a Gaussian over it, which is
precisely `P`. The degenerate case of the general mechanism is the recycle loop.

> **Keep the `lag = 0` endpoint specialized.** Presenting the spectrum as one policy is a
> description, not an implementation mandate. Materializing `P` as a `Tril` prior node at every step
> costs a density evaluation and a graph mutation per tick, where holding it in the estimator costs
> neither. Treat `lag = 0` as an optimized special case of one mechanism — not as a separate design,
> and not as something to unify away.

### State advance replaces values

`flattened-tangent-spaces.md:229-244` already identifies the missing operation and this document
gives it a name. The obstruction is a keying mismatch:

- `ForwardOutput.evaluations` is a `PointMap<B>` keyed by **mapped** ids (`compiler.rs:98`);
- the input `points` is a `ValueMap<B>` keyed by **leaf** ids;
- there is no id-remapping write between them.

```rust
/// Write the evaluated values of `from` into the leaf slots named by `into`.
///
/// Errors with `LayoutDimensionMismatch` unless `from.same_shape(into)`.
pub fn advance<B: Backend>(
    points: &mut ValueMap<B>,
    evaluations: &PointMap<B>,
    from: &TangentLayout,   // mapped nodes: f(x_k)
    into: &TangentLayout,   // leaf nodes:   x_k
) -> SkoroResult<()>;
```

The pairing is positional, checked by `TangentLayout::same_shape`
(`flattened-tangent-spaces.md:118-121`). Note the distinction this preserves:

> **Retraction applies a correction** — `retract_flat(x, δ)` for an EKF update or a Gauss-Newton
> step. **Advance replaces values** — `x_{k+1} = f(x_k)`.

Two are confusable and reaching for the wrong one is silent. While here: `ForwardOutput.observations`
(`compiler.rs:99`) is never written anywhere in the workspace — observed nodes lower to
`CompiledStep::Constant` and land in `constants`. Anyone reading it gets an empty map. It should be
deleted rather than documented.

## 4. Time: `TimeSequence`, `Chain`, `Chains`

Bookkeeping. No inference lives here, and nothing in this section knows what a filter is.

### `TimeSequence<T>`

An ordered `usize → T` with a `SequenceId`, where `T` is `i64`, `f64`, or any user clock. Ticks are
allocated in order and never reordered; the sequence is append-only under the single writer. It is
the authority on *what times exist*, and nothing else is.

### `Chain<B, Q>`

An ordered `usize → Q<B, T>` — one chain per quantity *role*, belonging to exactly one sequence. A
sensor bias has a chain; the measurement it produces has another; the state has a third. Retiring a
tick pops from the front of every chain in the sequence.

The user's anticipated difficulty here — needing to mutate several chains at once, e.g. evolving a
bias and its measurement together, and the borrow checker objecting — **does not arise**:

`Q<B, T>` holds its `ModelClient` **by value** and has no lifetime parameter
(`skoro-model/src/q.rs:21-28`):

```rust
pub struct Q<B: Backend, T> {
    pub id: Id,
    pub client: ModelClient<B>,       // owned; an Arc clone
    pub(crate) guard: Arc<NodeDropGuard>,
    _phantom: PhantomData<T>,
}
```

It is `Clone`, `Hash` and `PartialEq` (`q.rs:39, 50, 56`), keeps the server alive through the `Arc`
strong count, and can be stored in any collection — which is exactly how `Normal<B, M>` already
holds `mean: Q<B, M>` as a field (`skoro-normal/src/distribution.rs:79-86`). So you **read `Q`s out
of chains as cheap `Arc` clones, drop the borrow, and then build**. No chain is borrowed across a
model-building call.

> Chains need neither a concurrency container nor a mutex. Reads are `Arc` clones; writes are
> appends under the single writer of §2.

### `Chains<B>`

The type-erased registry: `Id → &dyn DynChain<B>` erased, `ChainKey<B, T> → &Chain<B, T>` typed.
Model it on the precedent already in the workspace rather than inventing a scheme —
`MODEL_SERVERS: DashMap<Id, Weak<dyn Any + Send + Sync>>` (`model.rs:33-40`) with downcasting
recovery in `ModelClient::from_id` (`model.rs:459-468`) is the same problem already solved once.

The registry is what makes the automation pay off: a model-builder and an estimator that are handed
the same `ChainKey<B, T>` can each walk the chain without knowing anything about the other.

### Multiple time sequences

Supported by construction, with one rule:

> **The entity is the `(SequenceId, tick)` pair, never the tick alone.**

A chain belongs to exactly one sequence, and no operation implicitly relates ticks across sequences.
Cross-sequence relationships are explicit `Associator` objects, which are themselves ordinary
model-building components — IMU preintegration between two camera ticks is the worked example: it
consumes a range of one sequence and produces a factor between two ticks of another.

This is deliberately the expert tier. A single-sequence model needs no `SequenceId` in its API
surface, and the design should keep it that way.

## 5. Markov structure without graph manipulation

Declare it once, get it everywhere:

```rust
pub trait Transition<B: Backend> {
    type State;
    fn extend(&self, ctx: &mut TickCtx<B>, prev: &Self::State, dt: f64)
        -> SkoroResult<Self::State>;
}
```

`extend` is written once and instantiated per tick. Because a tick's only inputs are the previous
tick's state and this tick's exogenous data, **the Markov property holds by construction**: the
*stochastic blanket* of tick `k` — the leaf quantities reachable downward through mapped and density
nodes — is exactly tick `k+1`'s state, with no graph analysis required.

That is the whole payoff. A general blanket computation over an arbitrary DAG is a real graph
algorithm; a blanket that falls out of the builder is a field lookup. Retirement becomes O(1) in the
size of the model. `graph-marginalization.md` §3 still specifies `Graph::stochastic_blanket` for the
general case, but a model built through `Transition` never needs to call it.

Non-Markov structure remains expressible — a user can reach past `Transition` to the ordinary model
API — it just costs the general blanket computation at retirement, which is the right price signal.
It also risks the *separability* precondition of `graph-marginalization.md` §4: a model whose
intermediate quantities entangle retired and surviving state in a way no blanket can absorb cannot be
marginalized at all. `Transition`-built models satisfy separability by construction.

## 6. Marginalization: who is allowed to do it

The user's sketch proposes:

```
new_joint_density = integrate(joint_density, [Q<B, T0>, ..., Q<B, Tn>])
```

**This cannot be a graph operation**, and the reason settles the ownership question on principle
rather than taste.

The marginal of a nonlinear model is not Gaussian, and in general has no closed form. Producing one
requires two things the graph does not have:

1. a **linearization point** — the values at which to linearize the retired quantities' relationships;
2. a **target family** — what to project the true marginal into (a Gaussian over the blanket, in
   every practical case).

So the signature is closer to:

```rust
fn integrate<B: Backend>(
    model: &ModelClient<B>,
    points: &ValueMap<B>,        // the linearization point
    retire: &[Id],
    projector: &impl MarginalProjector<B>,
) -> SkoroResult<()>;
```

And now the ownership question answers itself:

> **Only something holding a linearization point can marginalize, and estimators are what hold
> linearization points.**

Hence the split:

- The **Timeline** decides *when* a tick retires. It is the writer, and it sequences the operation.
- One designated **`prior_source` estimator** supplies the *numerics*. It does not mutate; it
  produces a prior.

This makes "two estimators at once" well-defined rather than order-dependent: **many readers, one
prior-source.** Estimators that are not the prior source are unaffected by retirement except that
quantities they were tracking cease to exist — which is a condition they must handle anyway, and
which the Timeline can signal. The graveyard (`graph-marginalization.md` §6) is the backstop: an
estimator that keeps using a retired `Q` gets `QuantityRetired` naming the id and the reason, rather
than a `NodeNotFound` that cannot distinguish "never existed" from "marginalized away".

If two estimators disagree about what should be retired, that is a modelling conflict and should
surface as one. The retention policy belongs to the Timeline; estimators may advise it, and it
decides.

## 7. Type erasure and cleanup: scoped capture, not per-component RAII

The question was whether developers must write cleanup logic for every quantity their model-building
component adds. **They should not.**

`TickCtx<B>` records every `Id` created inside a `Transition::extend` call. A tick's node set is
therefore captured automatically, and retiring the tick means *remove the captured set, minus the
blanket*. The developer writes `extend` and nothing else.

The precedent is already in the workspace one level down. `NodeDropGuard`
(`skoro-model/src/model.rs:408-426`) holds an `Id`, an `AtomicBool` marking whether the node is
still a mapped leaf, and an `Arc<SegQueue<Id>>`; dropping the last `Q` for a mapped leaf enqueues
its id, and `ModelServer::cleanup()` (`model.rs:198`) drains the queue. Scoped capture is the same
idea applied to a *group* of nodes with a known lifetime, and it composes with what exists: the
guards continue to reclaim intermediate mapped nodes; capture handles the tick's roots.

The same guards do double duty for the graveyard. Because every `Q` clone shares one
`Arc<NodeDropGuard>`, an epitaph for a retired id can be pruned exactly when the last handle that
could misuse it drops — see `graph-marginalization.md` §6, including the cost that reuse carries.

### Two traps around identity

**`IdGenerator: Clone` copies the counter.** `Graph` derives `Clone` (`graph.rs:34`) and
`IdGenerator`'s manual `Clone` (`skoro-core/src/id.rs:11-15`) reads the atomic and constructs a new
one at the same value:

```rust
impl Clone for IdGenerator {
    fn clone(&self) -> Self {
        Self(AtomicU64::new(self.0.load(Ordering::Relaxed)))
    }
}
```

So two clones of a graph issue **colliding ids** from that point forward, forever. Any design that
clones a tick template to instantiate ticks is standing on a landmine. Instantiate by re-running
`extend` against the live graph; do not clone.

**Ids are never reclaimed.** `IdGenerator` is a monotonic `AtomicU64` (`id.rs:24`) with no free
list, and `Graph::remove` (`graph.rs:740`) does not return them. This is fine — `u64` at even a
kilohertz tick rate is not a real limit — but it means id exhaustion is not the bound on how long a
filter can run, and node-count is. Which is the point of §3.

## 8. Concurrency

An actor shape, forced by §2:

| thread | does | touches the model |
|---|---|---|
| **Ingest** | buffer raw measurements into a `SegQueue` | never |
| **Driver** | drain the queue, associate times, extend the model, step the estimator | sole writer |
| **Readers** | consume estimates via `server_call` (read lock) | read only |

`crossbeam-queue` is already a dependency (`skoro-model/Cargo.toml`), used for the cleanup queue, so
the ingest side needs nothing new.

The user anticipated that time-association would need a lock, because it requires both the
unassociated inputs and the time sequence. Under single-writer it needs none: association happens on
the driver thread, which already owns both.

### Hoist the compiled sequence

This is the load-bearing performance point, and it is also what keeps the read locks short enough
for the actor shape to work.

Every evaluation entry point compiles from scratch and throws the result away — `graph.rs:839, 869,
920, 992, 1004, 1027, 1051, 1077, 1101, 1139`. Each compile does a topological sort plus a boxing
`op.lower(..)` per node, and `generate_from` compiles over *every node in the graph*
(`graph.rs:991-992`). `Graph::evaluate`'s own doc comment (`graph.rs:1183-1185`) warns about the
cost.

But the reuse hook is already public: `compile_backward_from` is `pub` (`graph.rs:1366`) and all four
execution methods take `&self` (`compiler.rs:167, 215, 284, 373`). In recycling mode the graph does
not mutate between ticks, so:

> **Compile once, reuse across ticks, execute outside the read lock.**

This is both why `lag = 0` is cheap and why the driver is not stalled behind readers holding a long
read lock through repeated recompilation.

Nothing detects staleness — there is no compiled-sequence field on `Graph` and no invalidation
mechanism, because nothing currently persists one. A hoisted sequence must therefore be invalidated
explicitly on any structural change. The Timeline is the natural owner of that invalidation, since
it is the only writer.

## 9. Is this an ECS?

Structurally, yes — the workspace already is one:

| ECS concept | Skoro |
|---|---|
| entity | `Id` (and `(SequenceId, tick)` in the temporal layer) |
| component storage | `Graph.nodes: HashMap<Id, Node<B>>`; `Chain<B, T>` as a dense column |
| system | an estimator's `step`, a `Transition::extend` |
| archetype | quantity type + node kind |

Use the vocabulary — it is the clearest way to describe `Chains<B>` — but **do not take an ECS
dependency**. The archetypes are statically known, the chain count is tens rather than thousands,
and there is no dynamic-composition requirement anywhere in the design. A bespoke registry over the
`DashMap`/`Weak` pattern already proven at `model.rs:33-40` is smaller than the dependency and
matches the workspace's habit of building the narrow thing (its own `Backend` trait over burn, its
own autodiff engine).

## 10. The estimator boundary

Ground the boundary in `TangentLayout` rather than in a new trait hierarchy, so this document and
`flattened-tangent-spaces.md` describe one system instead of two.

> A **batch** estimator takes a layout and a point, and returns a delta.
>
> A **sequential** estimator takes a layout, *plus a rule relating consecutive layouts*, and carries
> numerical state across the boundary.

That is the entire difference. The rule already has a name — `TangentLayout::same_shape`
(`flattened-tangent-spaces.md:118-121`) — specified there for exactly the EKF's `state` /
`state_next` pairing. Everything sequential is batch plus that rule, and the carried state is `P`.

Concretely, `SequentialEstimator` differs from a batch estimator by three things and no more: it
holds `P`, it holds the layout pairing, and it may be designated `prior_source` (§6). It gains no
authority over structure.

The corollary from §2 restated, because it is the thing most likely to be got wrong: **`P` lives in
the estimator, keyed by a layout, and never in the graph.**

## 11. Phasing

Ordered so each phase is independently useful and each is testable before the next begins.

1. **`lag = 0` EKF.** Needs `TangentLayout`, Jacobian assembly, the dense SPD solve (all from
   `flattened-tangent-spaces.md`), plus `advance` from §3. Needs **no** temporal layer, **no**
   marginalization, **no** temporal bookkeeping, **no** graph surgery of any kind. This is a working
   filter, and it is reachable without touching any of the graph defects.
2. **The temporal layer.** `TimeSequence`, `Chain`, `Chains`, `Transition`, `TickCtx`. Pure
   bookkeeping, verifiable without any estimator, and it makes `lag = ∞` sliding-window Gauss-Newton
   expressible.
3. **Retirement.** The graph work of `graph-marginalization.md`, yielding `lag = L`. Smaller than it
   looks, and partly done: `demote_to_parameter` is implemented on the graph, server, client and
   `ModelClientOps`, a single-node blanket is `update_stochastic`, and a joint marginal is
   `mark_likelihood` — all existing API. What remains is the stochastic-blanket and dead-set walks,
   the separability check, and the graveyard.
4. **Multiple sequences.** `SequenceId` plumbing and `Associator`.

Phase 1 is the one to start. It proves the state-advance path end to end and it is the phase whose
requirements are already fully specified elsewhere.

## 12. Verification

Follow the existing layout — per-backend entry points over a shared `common/` module, as in
`skoro-linalg/tests/{ndarray_f32,ndarray_f64,wgpu_f32}.rs` and `skoro-se3/tests/common/mod.rs:296-306`.
Heed the workspace's own doctrine (`.claude/skills/skoro-diff-operation/references/testing.md`):
hand-computed constants generated from the implementation lock in whatever the implementation
already did.

1. **Exactness at `lag = 0`.** On a linear-Gaussian model the recycling loop must reproduce the
   closed-form Kalman filter, not merely converge to it.
2. **Advance/retract are distinct.** A model where `f` is not the identity must give different
   answers for `advance` and `retract_flat` with the same tensor — a test that fails if the two are
   ever conflated.
3. **Window/batch agreement.** A `lag = L` model with retirement and the equivalent `lag = ∞` batch
   model must give the same posterior on a linear-Gaussian problem, where marginalization is exact.
   This is the test that justifies the whole retirement mechanism.
4. **Tag survival.** `Q<B, S_j>` evaluates identically before and after its tick's predecessor
   retires. The user's requirement that tags keep naming quantities, made checkable.
5. **Bounded growth.** `graph.nodes.len()` is bounded across many ticks at `lag = L`, and
   `Chain::len()` tracks the retention policy. Catches capture leaks (§7).
6. **Retired handles fail loudly.** A `Q` held across its tick's retirement returns `QuantityRetired`
   from the fallible APIs and panics informatively from operator sugar, and its epitaph is pruned once
   the last handle drops. The graveyard's growth must be bounded by *live retired handles*, not by
   history — ids are never reclaimed (§7), so an unpruned graveyard is the leak that replaces the one
   retirement was meant to fix.
7. **Registration is not observed half-done.** A reader holding the read lock while the driver
   registers a stochastic quantity never observes a stochastic node without its density — the
   specific non-atomicity at `model.rs:553-566`. Note the failure mode of the broader concurrency
   story is a *panic* on a poisoned lock (§2), not a wrong answer, so there is nothing weaker worth
   asserting here.
8. **Hoisting is sound.** A hoisted `CompiledSequence` reused across ticks agrees with per-tick
   recompilation, and is correctly invalidated by a structural change (§8).

Run with `cargo test --workspace`. The wgpu suites additionally exercise the device path.
