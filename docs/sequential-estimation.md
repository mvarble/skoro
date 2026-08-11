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

There is a second gap, and it is the one that decides the architecture. The DAG view also says
nothing about **structure that changes from tick to tick**. A tick's dynamics are driven by data that
arrives from outside the model, so the edge between one state and the next is a different subgraph
each time. Whether a measurement is present at all is a per-tick fact, so there is no fixed `h` with
a fixed codomain. Both gaps are answered by the same loop — *build a tick, estimate, retire a tick* —
and neither is answered by a graph built once and re-valued.

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

The single-writer discipline is not a rare-event property. Because every tick both adds and removes
nodes (§3), the Timeline mutates structure continuously — it is the busiest thing in the system, not
a garbage collector that wakes occasionally. Everything §8 says about the actor shape follows from
that, and so does the fact that the estimator's read of the model is a read even during a filter's
own step.

`P` is an object on the tangent space *at the current point*, shape `[batch, N, N]`, keyed by a
`TangentLayout` (`flattened-tangent-spaces.md:60-68`). It has no node, no `Id`, and no place in
`Graph.nodes`. Under retention-1 (§3) it is not a tensor either: at batch 1 it is a host `f64` matrix, since
this same ownership line is the host/device and precision boundary
(`estimator-linear-algebra.md` §1). Estimators are therefore emphatically **not** stateless, and that is the point: it is
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

## 3. Retention is a policy over a subsequence

Filtering, fixed-lag smoothing, keyframe smoothing, and batch estimation differ in exactly one thing:
**which ticks stay in the graph**. Not *how many* — a dynamic smoother maps a time sequence
`t₀ < … < tₙ` to an arbitrary subsequence `t_{m₀} < … < t_{mₖ}`, dense near the present but keeping
anchors near the beginning. "Lag" names only the special case where the retained set is a suffix.

| policy | ticks retained | forgetting | blanket size |
|---|---|---|---|
| retention-1 (filter) | 1 | **deletion** — `P` already *is* the marginal | n/a |
| fixed-lag | the last `L+1` | marginalize the oldest | 1 (the chain head) |
| keyframe / dynamic | an arbitrary subsequence | marginalize interior ticks | **> 1, routinely** |
| batch | all | none | n/a |

The organizing claim, and the one everything below unpacks:

> The graph carries **structure**; the estimator carries **information**. At retention-1, forgetting
> is **deletion**, because `P` already holds everything the retired tick knew. At retention > 1,
> forgetting is **marginalization**, because the graph holds information about the window that only
> an estimator can project.

### Retention-1: forgetting is deletion, not marginalization

A plain EKF needs no marginalization machinery whatsoever — no blanket walk, no separability check,
no marginal projector. That much of the old "the filter *is* the marginalization" claim survives, and
it is worth saying loudly, because it is still the cheapest path to a working filter.

What does **not** survive is the mechanism. An earlier draft of this section had one tick's worth of
structure living in the graph permanently — a state leaf `x`, a process map `f(x)`, a measurement map
`h(x)` — with advance *overwriting the leaf slot* and the graph never changing shape. That is wrong
for this system, for two independent reasons, either of which alone would sink it:

- **The dynamics are driven by exogenous data.** A tick's process map is
  `f(x_k, dataₖ)`, and `dataₖ` differs every tick. The subgraph between the current state leaf and
  the next state is rebuilt each tick, not re-valued (§5).
- **Measurements are conditional.** Which factors exist at tick `k` depends on what data arrived.
  There is no fixed `z = h(x)` with a fixed codomain to keep a slot for (§5).

A template would have to be the union over every shape a tick can take, with a way to switch factors
off — and the graph has no such switch. So:

> **The graph grows and shrinks every tick, at every retention policy.** There is no template to
> recycle and no observed node to re-value.

The retention-1 loop is therefore five steps, and the order is forced:

1. **Build** `f(x_k, dataₖ)` from the live state leaf `x_k`.
2. **Linearize** — `F = ∂f/∂x_k`, a Jacobian from the leaf `x_k` to the mapped node `f`.
3. **Advance** — evaluate `f` and write its value into a **fresh `Parameter` leaf** `x_{k+1}`.
4. **Build this tick's measurements** on `x_{k+1}`, then update: `H`, the innovation, `δ`, `P`.
5. **Remove tick `k`'s captured node set** (§7).

Step 3 is what makes step 5 legal, and this is the structural point to hold on to:

> **The value-write is what severs the graph.** After `x_{k+1}` is seeded by value rather than
> wired to `f(x_k, dataₖ)`, tick `k`'s cone is childless, and `Graph::remove` accepts it.

That is precisely the role the blanket rewire plays in the general retirement sequence
(`graph-marginalization.md` §7, step 4 → step 5). The two policies do the same thing in the same
order; they differ in whether the severed edge is replaced by a *value* or by a *prior*.

Retention-1 stays the cheapest policy — but for a different reason than the old draft gave. It is not
that it avoids graph mutation; it doesn't. It is that **no marginal has to be computed**: there is no
blanket to walk, no linearization to project, no factor to materialize. Only nodes to remove.

#### The state leaf is a `Parameter`, not a `Stochastic`

A `Stochastic` leaf added through `add_quantity` gets a density node marked into `log_likelihoods`
(`graph.rs:290`), and that prior factor would then coexist with `P`, which is supposed to carry the
same information. Nothing double-counts in practice — an EKF only ever calls `evaluate_ids_jvp` /
`evaluate_ids_vjp`, never `log_density` — but relying on that is relying on an estimator's discipline
rather than on the model being right.

A `Parameter` leaf removes the question. It has no distribution and contributes nothing to
`log_likelihoods`, and it is a first-class Jacobian column regardless: the VJP collection loop
matches `CompiledStep::Parameter` and `CompiledStep::Sampler` alike (`compiler.rs:354-357`), and
`non_op_forward` seeds its tangents and frames at `compiler.rs:536-545`. The prior lives in `P`
where it belongs.

The reason is about *estimators*, not about retention. A **filter** needs no process factor in the
graph at all, because `P` and `F` carry it; a **smoother** does, because the window is scored by
`log_density` and the transition is one of its factors. So a retained window builds its states with
`add_quantity` and its head ends up a `Parameter` again after demotion
(`graph-marginalization.md` §5) — the same kind as the retention-1 leaf, reached from the other
direction. Nothing is ever promoted into a stochastic node; the only kind change in the design
removes structure rather than inventing it.

#### A tick is built in two passes, not one

Step 4 builds the measurement on the **fresh leaf `x_{k+1}`**, never on the mapped node
`f(x_k, dataₖ)`. This is forced by `flattened-tangent-spaces.md` §5 — Jacobian columns must be
leaves — and by the update, which retracts `x_{k+1}`. A measurement hung off the mapped node yields a
Jacobian with respect to the *previous* state and a correction applied to a quantity the measurement
did not see.

The consequence for the API: a **numeric step sits between the two construction steps**. A tick
cannot be one builder closure. `TickCtx` (§7) must support suspending construction, letting the
driver evaluate and advance, and resuming — while still capturing every id created across both
passes. Getting this wrong is silent: the model builds, the filter runs, and the linearization point
is wrong.

#### Peak footprint is two ticks

`x_k` has to stay alive through steps 1–3 — it is the input to `f`, the point of evaluation, and the
column of `F`. Only after `x_{k+1}` is seeded can tick `k` go. So even at retention-1 the graph
transiently holds two ticks, and "bounded growth" means bounded at about two, not at one. Any test or
budget written against a one-tick footprint will fail correctly and be debugged wrongly.

### The retention policy

Because the retained set is a subsequence rather than a suffix, the policy is a trait, not a scalar:

```rust
pub trait RetentionPolicy<T> {
    /// Given the retained ticks in ascending time and the tick just appended,
    /// name the ticks to retire, in elimination order.
    fn select(&self, retained: &[TickRef<T>], appended: TickRef<T>) -> Vec<TickRef<T>>;
}
```

`KeepLast(1)` is the filter, `KeepLast(L)` fixed-lag, `KeepAll` batch, and `Keyframes` the dynamic
smoother — log-spaced, say, dense near the present with anchors near `t₀`. The policy names *which*
and *in what order*; it never names a mechanism.

> **The mechanism is chosen by blanket cardinality, never by the policy.** A blanket of one has a
> marginal that factorizes, and `update_stochastic` re-roots it in place. A blanket of more than one
> has a correlated marginal, and it becomes `demote_to_parameter` plus a joint `mark_likelihood`
> factor (`graph-marginalization.md` §5).

Retiring the *oldest* retained tick gives a blanket of one; retiring an *interior* one gives a
blanket of two — its neighbours on both sides. Keyframe retention does the latter continuously, so
the joint-factor path is the norm under this policy, not the exotic case.

Which makes **fill-in** a cost the policy has to price. Eliminating an interior tick installs a factor
directly between its neighbours, so repeated interior elimination densifies the factor graph: the
survivors accumulate edges they did not have. Two consequences for a policy implementation — prefer
eliminations with small blankets, and treat elimination *order* as part of the decision, since
eliminating `t_i` before `t_j` and the reverse leave differently connected graphs. This is ordinary
variable-elimination fill-in, and it is the reason `select` returns an ordered `Vec` rather than a
set.

### Batch

Nothing retires; the graph is the whole history. This is what Gauss-Newton over a fixed dataset
already does today, and it needs nothing from this document — except the cost note in §8, since a
graph that never shrinks makes per-tick *compilation* grow without bound.

### What would make recycling legal later

The template idea is not wrong in general, only inapplicable here, and the preconditions are worth
recording so a future model can claim them rather than rediscover the argument:

1. **Exogenous data of fixed shape**, entering through a `Constant` or `Observed` node of fixed
   dimension, so that a tick's process subgraph is structurally identical from tick to tick.
2. **An unconditional measurement set of fixed codomain**, so no factor ever has to appear or vanish.

Given both, `observe` / `unobserve` re-valuing plus a hoisted `CompiledSequence` (§8) is sound, and a
constant-structure model — a fixed-rate propagation with an always-present measurement — could take
that path. It is a deferred optimization for a special case, behind the same seam as the general
loop. It is not the general path, and nothing in this design should assume it.

### State advance replaces values

`flattened-tangent-spaces.md` §4 ("Retraction is not state advance") already identifies the missing
operation and this document
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

The signature is unaffected by per-tick construction. What changes is only what `into` names: the
**next tick's newly created leaves**, rather than the same leaves written in place. The pairing is
still positional, checked by `TangentLayout::same_shape`
(`flattened-tangent-spaces.md:118-121`), and both layouts are now per-tick objects (§10). Note the
distinction this preserves:

> **Retraction applies a correction** — `retract_flat(x, δ)` for an EKF update or a Gauss-Newton
> step. **Advance replaces values** — `x_{k+1} = f(x_k, dataₖ)`.

Two are confusable and reaching for the wrong one is silent.

> **A third name collides with these two.** `advance` here is the *value-writing helper* above. The
> per-tick *model-building* step — the one that creates `f(x_k, dataₖ)` as nodes — is
> `Transition::extend` (§5), and it is natural to want to call that `advance` as well. Keep them
> distinct: `extend` adds structure, `advance` moves numbers, and a tick calls `extend` then
> `advance` in that order.

While here: `ForwardOutput.observations`
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
sensor bias has a chain; the measurement it produces has another; the state has a third.

> **Retirement removes from the middle, not only the front.** Under keyframe retention (§3) the
> retained set is a subsequence, so a chain must support **interior removal**. Popping the front is
> the fixed-lag special case.

`Chain` is therefore indexed by *retained position*, and the `(SequenceId, tick)` pair below stays
the identity — the two must not be conflated, or an interior retirement silently renumbers every
later tick. `TimeSequence` remains append-only and keeps naming times that no longer have quantities;
that is what lets a graveyard error (§6, `graph-marginalization.md` §6) say *when* something was
retired and not just that it was.

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

## 5. Building a tick

Every tick adds nodes (§3), so the builder is not a convenience layer — it is the thing that runs
most often in the whole system. Declare it once, get it everywhere.

### Exogenous data

```rust
pub trait Transition<B: Backend, Data> {
    type State;
    fn extend(&self, ctx: &mut TickCtx<B>, prev: &Self::State, data: &Data)
        -> SkoroResult<Self::State>;
}
```

`extend` is written once and **re-run against the live graph every tick** — never against a clone,
which issues colliding ids (§7), and never against a fresh model, which issues *aliasing* ones (§7).
`Data` is what makes the per-tick subgraph differ: an IMU increment, a wheel-odometry delta, a
control input, a `dt`. The old `dt: f64` parameter is the degenerate `Data`, and it is the only case
in which two ticks produce structurally identical subgraphs.

`TickCtx<B>` carries the model client and captures every `Id` created inside the call (§7). It must
also survive the numeric step in the middle of a tick (§3, "A tick is built in two passes"): the
driver calls `extend`, evaluates and advances, then builds measurements — and the capture must span
both passes.

### Measurements are conditional

Whether a factor exists at tick `k` is a fact about the data that arrived, not about the model. So a
measurement is a **visitor** offered a per-tick, type-erased data bag; it adds a factor only when its
data is present:

```rust
associated_data.get::<Data>(visitor.key())
    .map(|data| {
        let z = ctx.add_quantity(visitor.distr(state, data))?;
        z.observe(&visitor.adapt(data))?;
        Ok(z)
    })
    .transpose()
```

Three things follow, and all three are load-bearing for the estimator boundary (§10):

- The measurement `TangentLayout` is **rebuilt per tick**, from whichever factors fired.
- `M` varies from tick to tick, and `M = 0` is a legal, ordinary tick — predict with no update.
- `R` is block-diagonal over the factors that fired, not a fixed matrix.

`z` must come from `add_quantity`, because `Graph::observe` accepts only `Stochastic` or
already-`Observed` nodes (`graph.rs:347-362`) — a `Mapped` or `Parameter` node cannot be observed.
That is the same constraint from the other side as `flattened-tangent-spaces.md` §5's "columns are
leaves".

> **The erasure idiom is not the one it looks like.** `MODEL_SERVERS`
> (`model.rs:33-40`) is keyed by `Id`, not by type, so it is not the precedent for a type-keyed bag.
> The workspace has **no `TypeId` uses and no `Any`-keyed map at all**. What it does have, three
> times over, is *a generic newtype erased to a `dyn` trait object with `as_any()` recovery* —
> `DynContinuous<T>` / `DynDiscrete<T>` (`skoro-graph/src/quantity.rs:192, 228`), `KindedSampler`
> (`graph.rs:175`), and `DynQuantityType::compare(&dyn Any)` (`quantity.rs:210-215`). Follow that:
> a `KindedVisitor<B, Data, V>(V)` stored as `Arc<dyn DynVisitor<B>>`. This is genuinely new ground
> in this workspace and should be built like its neighbours rather than like a general ECS.

### The growth gap in `ModelClientOps`

The visitor above calls `ctx.add_quantity(..)`, and it is worth being exact about why it cannot call
it on a `&impl ModelClientOps<B>` today. The trait (`skoro-model/src/client_ops.rs:46` — note, *not*
`model.rs`) does grow the graph: `add_constant` (`:50`) and `add_parameter` (`:57`) are both on it,
as are `observe` (`:65`), `unobserve` (`:72`), `demote_to_parameter` (`:81`), `remove` (`:84`) and
`cleanup` (`:87`). What is missing is precisely the **stochastic** growth path — `add_quantity`
(`model.rs:533`) and `update_quantity` (`model.rs:615`) are inherent methods on the concrete
`ModelClient<B>` and appear on no trait. Marking is not a gap: `Q<B, Real>::mark_likelihood`
(`q.rs:178`) routes through the client the `Q` already owns, as does every operator that builds a
density.

So the gap is two methods, and the fix is a **separate** trait rather than a wider one:

```rust
pub trait ModelBuilderOps<B: Backend>: ModelClientOps<B> {
    fn add_quantity<K, D>(&self, distr: D) -> SkoroResult<Q<B, D::Qty>>
        where K: Kind<B>, D: Distribution<B, K>;
    fn update_quantity<K, D>(&self, tag: &Q<B, D::Qty>, distr: D) -> SkoroResult<()>
        where K: Kind<B>, D: Distribution<B, K>;
}
```

Widening `ModelClientOps` itself would break a deliberate design decision: `GaussianModelClient`
implements `ModelClientOps` while *withholding* `add_quantity` (`facade.rs:196-200`), exposing only
`add_gaussian_quantity` (`facade.rs:71`) — a generic `add_quantity` on the base trait would void that
facade's guarantee. `ModelBuilderOps` is implemented by `ModelClient<B>` alone; a facade that wants a
tick builder supplies its own narrower one. The methods are generic over `D: Distribution` and so not
dyn-compatible, which is fine — builders are monomorphized.

### Markov structure without graph analysis

Because a tick's only inputs are the previous tick's state and this tick's exogenous data, **the
Markov property holds by construction**: the *stochastic blanket* of tick `k` — the leaf quantities
reachable downward through mapped and density nodes — is exactly tick `k+1`'s state, with no graph
analysis required.

That is the whole payoff. A general blanket computation over an arbitrary DAG is a real graph
algorithm; a blanket that falls out of the builder is a field lookup. Retirement becomes O(1) in the
size of the model. `graph-marginalization.md` §3 still specifies `Graph::stochastic_blanket` for the
general case, but a model built through `Transition` never needs to call it.

> **The payoff holds for appends, and degrades under interior retirement.** Eliminating an interior
> tick installs a joint factor spanning its neighbours (§3), so after a few of them a retained tick's
> blanket is no longer "the next tick's state" — it is the union of retained ticks reachable through
> the factors incident on it.

The repair is small and belongs to the temporal layer, not the graph: maintain
`incident: HashMap<TickRef, HashSet<Id>>`, updated when a joint factor is installed and when one
falls into a dead set. A tick's blanket is then still a lookup, over one more table.
`Graph::stochastic_blanket` stays the reference oracle to test that table against (§12).

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

At retention-1 this section is vacuous, and usefully so: there is **nothing to marginalize**. `P` is
already the marginal, no prior is materialized, and the `prior_source` designation names nobody. The
role becomes load-bearing exactly when a window is retained, and it becomes *continuous* under
keyframe retention, where interior eliminations happen on most ticks.

Hence the split:

- The **Timeline** decides *when* a tick retires — by asking its `RetentionPolicy` (§3). It is the
  writer, and it sequences the operation.
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

`TickCtx<B>` records every `Id` created across a tick's two construction passes (§3). A tick's node
set is therefore captured automatically, and retiring the tick means *remove the captured set, minus
the blanket* — where at retention-1 the blanket is empty and the whole set goes. The developer writes
`extend` and a measurement visitor, and nothing else.

> **Capture is load-bearing, not a convenience.** Every tick creates nodes now, and `NodeDropGuard`
> reclaims **only mapped leaves** — `is_mapped_leaf` is set `true` solely by `add_mapped`
> (`model.rs:585`), while constants, parameters, stochastic nodes, and anything marked as a density
> all get `false`. A tick's state leaf, its observed measurement values, and its density nodes are
> therefore *unreclaimable* without explicit removal of the captured set. Without capture, the filter
> leaks by construction.

The precedent is already in the workspace one level down. `NodeDropGuard`
(`skoro-model/src/model.rs:408-426`) holds an `Id`, an `AtomicBool` marking whether the node is
still a mapped leaf, and an `Arc<SegQueue<Id>>`; dropping the last `Q` for a mapped leaf enqueues
its id, and `ModelServer::cleanup()` (`model.rs:198`) drains the queue. Scoped capture is the same
idea applied to a *group* of nodes with a known lifetime, and it composes with what exists: the
guards continue to reclaim intermediate mapped nodes; capture handles the tick's roots.

### Removing a captured set

`Graph::remove` (`graph.rs:708`) takes `&[Id]` and is **leaf-only and all-or-nothing**: it errors
`NodeStillReferenced` if any target has a child outside the batch. A captured set must therefore be
handed over as **one batch**, and the batch must be closed downward — every child of a removed node
either removed with it or genuinely gone. There is no transitive teardown helper anywhere in the
workspace today, and one is needed; see §11, where it is the first phase.

What `remove` does get right: it prunes the removed ids from their parents' `children` sets and
de-registers a removed `LogDensity` from `log_likelihoods` and both index maps
(`graph.rs:740-753`), so a tick's measurement factors leave cleanly along this path.

The same guards do double duty for the graveyard. Because every `Q` clone shares one
`Arc<NodeDropGuard>`, an epitaph for a retired id can be pruned exactly when the last handle that
could misuse it drops — see `graph-marginalization.md` §6, including the cost that reuse carries.

### Three traps around identity

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

> **The same property is a *feature* for snapshot/restore, and that is worth keeping.** Counter
> preservation is a hazard only when both graphs stay live. Take a snapshot, mutate, and put the
> snapshot *back* — `let snap = client.server_call(|s| s.graph.clone())` then
> `client.server_call_mut(|s| s.graph = snap)` — and every `Q` in user hands stays valid, because
> every id means the same thing. The clone is shallow (`op`, `sampler` and `quantity_type` are all
> `Arc`), so it costs two `HashMap` reallocations and no tensor copies. That is the primitive to
> build retirement rollback on. Two caveats: `cleanup_queue` lives on `ModelServer`, not `Graph`, and
> is not snapshotted, so stale ids left in it can make a later `cleanup()` panic
> (`model.rs:206`); and restoring a snapshot taken at a *lower* counter value re-issues live ids —
> only ever restore forward.

**A fresh model per tick aliases ids instead of colliding.** The natural reaction to "there is no
template to recycle" is to build a new `ModelServer` each tick and let the old one drop. Do not.
`Graph::new` gets an `IdGenerator` starting at 0 (`id.rs:22`, `graph.rs:106`), `Id` is a bare `u64`
newtype carrying no server provenance (`id.rs:5`), and every graph lookup uses the raw `Id`. A `Q`
retained from the previous tick therefore does not error against the new model — it **silently names
a different node**. `Q::PartialEq` compares `server_id`, but nothing on the lookup path does. This is
the `Clone` landmine from the other side, and it is worse, because there is no point at which it
announces itself.

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

### Hoist the compiled sequence — within a tick

This is the load-bearing performance point, and it is also what keeps the read locks short enough
for the actor shape to work. What changed is its *scope*.

Every evaluation entry point compiles from scratch and throws the result away — `graph.rs:839, 869,
920, 992, 1004, 1027, 1051, 1077, 1101, 1139`. Each compile does a topological sort plus a boxing
`op.lower(..)` per node, and `generate_from` compiles over *every node in the graph*
(`graph.rs:991-992`). `Graph::evaluate`'s own doc comment (`graph.rs:1183-1185`) warns about the
cost. Nothing anywhere caches a `CompiledSequence`.

The old form of this point assumed the graph does not mutate between ticks. It does (§3), so:

> **Compile once per tick, reuse within the tick, execute outside the read lock. A compiled sequence
> does not survive a tick boundary.**

There is still real work being saved. A single tick issues several evaluation calls over overlapping
subgraphs — `F` by JVP, the advance evaluation, `H` by VJP, and a `log_density` pass for any
estimator that scores — and each recompiles today. The reuse hook is already public:
`compile_backward_from` is `pub` (`graph.rs:1366`) and all four execution methods take `&self`
(`compiler.rs:167, 215, 284, 373`), so a driver can compile a tick's targets once and run all four
passes against it without any new API.

**A second consequence, and it is new: bounded node count now bounds latency, not just memory.**
`evaluate_ids*` costs O(reachable subgraph) and `generate_from` costs O(all nodes) *per tick*, so an
unretired history makes every tick more expensive than the last, whether or not memory is a concern.
This is an argument for retirement independent of the one in §3, and it is the one that bites first
on a long batch run.

Cross-tick reuse is available only under the constant-structure preconditions recorded in §3, and a
hoisted sequence then needs explicit invalidation on any structural change — nothing detects
staleness, because there is no compiled-sequence field on `Graph` and no invalidation mechanism. The
Timeline is the natural owner of that invalidation, since it is the only writer.

## 9. Is this an ECS?

Structurally, yes — the workspace already is one:

| ECS concept | Skoro |
|---|---|
| entity | `Id` (and `(SequenceId, tick)` in the temporal layer) |
| component storage | `Graph.nodes: HashMap<Id, Node<B>>`; `Chain<B, T>` as a dense column |
| system | an estimator's `step`, a `Transition::extend`, a measurement visitor |
| archetype | quantity type + node kind |

The measurement visitor of §5 is the closest thing here to a real ECS system: it is offered every
tick, queries a type-keyed bag for the component it cares about, and does nothing when that component
is absent. If the analogy ever earns a dependency, it is there — and it still does not, for the
reasons below.

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

### Layouts are per-tick objects

> **This supersedes `flattened-tangent-spaces.md` §7's "Layouts built once: `state`, `state_next`,
> `meas`."**

Since every tick creates new nodes (§3), every layout over those nodes is rebuilt each tick. This is
cheap — a `TangentLayout` holds no tensors, carries no backend parameter, and is a `Vec` of blocks
plus an index — and it changes nothing about the algebra:

- `state` and `state_next` are rebuilt per tick and still paired positionally by `same_shape`. The
  pairing now relates *consecutive ticks' quantities* rather than two fixed sets, which is what it
  was always meant to express.
- `meas` is rebuilt from whichever measurement factors fired (§5). Its total `M` varies from tick to
  tick, so `H`, `S`, `K` and `R` are all differently shaped each time. A sequential estimator must
  therefore allocate per tick rather than once, and must handle **`M = 0`** — a predict-only tick,
  where the update is skipped entirely and `P` is left at its predicted value.

None of this touches the host `f64` boundary below: a fresh `[1, M, N]` at batch 1 is a small
allocation, and reshaping per tick is what the host path is good at.

The corollary from §2 restated, because it is the thing most likely to be got wrong: **`P` lives in
the estimator, keyed by a layout, and never in the graph.**

That line does double duty. `estimator-linear-algebra.md` §1 takes it as the host/device boundary as
well: everything the estimator owns numerically is host `f64` at batch 1, and the only tensors
crossing are `J` inward and `δ` outward. Nothing about the estimator boundary above changes — the
`TangentLayout` grounding is exactly what makes the conversion a two-call affair.

## 11. Phasing

Ordered so each phase is independently useful and each is testable before the next begins.

1. **Graph teardown.** A new first phase, forced by §3: a filter that builds every tick must be able
   to delete every tick. Needs a **transitive teardown helper** — `Graph::remove` is leaf-only and
   all-or-nothing (§7), so something has to compute the downward closure of a captured set and hand
   it over as one closed batch — and it needs `Graph::remove`'s unresolved
   `// TODO: check if I need to remove children` (`graph.rs:760`) resolved rather than inherited.

   > Say the uncomfortable part plainly: **phase 1 now rests on the least-tested path in
   > `skoro-graph`.** `graph-marginalization.md` §8's defect 5 is a phase-1 blocker, not phase-3
   > cleanup, and the removal path needs tests before anything is built on it.

   *What is not a blocker here:* the orphaned-`LogDensity` leak (`graph-marginalization.md` §8.1).
   `unregister_density` runs only from `update_stochastic` and `demote_to_parameter`, and a
   retention-1 filter — `Parameter` state leaf, no process factor, measurements via `add_quantity` +
   `observe`, teardown via `remove` — calls neither. It becomes a blocker at phase 4.
2. **Retention-1 EKF.** Needs `TangentLayout`, Jacobian assembly, the dense SPD solve (all from
   `flattened-tangent-spaces.md`), plus `advance` from §3 and phase 1's teardown. The solve is a
   dependency on the `skoro-dense` crate rather than on new backend kernels — see
   `estimator-linear-algebra.md`, which is worth reading before this phase starts. Needs **no**
   temporal layer, **no** marginalization, **no** blanket walk, **no** projector. This is a working
   filter. It is no longer reachable without touching graph structure, which is why phase 1 exists.
3. **The tick layer.** `TimeSequence`, `Chain` (with interior removal), `Chains`, `TickCtx`,
   `Transition<B, Data>`, `ModelBuilderOps`, the associated-data bag and the measurement visitor.
   Mostly bookkeeping and verifiable without any estimator; it is what makes varying `M` and `M = 0`
   real, and it makes batch sliding-window Gauss-Newton expressible.
4. **Retirement.** The graph work of `graph-marginalization.md`, yielding fixed-lag. Smaller than it
   looks, and partly done: `demote_to_parameter` is implemented on the graph, server, client and
   `ModelClientOps`, a single-node blanket is `update_stochastic`, and a joint marginal is
   `mark_likelihood` — all existing API. What remains is the stochastic-blanket and dead-set walks,
   the separability check, the orphaned-density fix, and the graveyard.
5. **Keyframe retention.** `RetentionPolicy` beyond `KeepLast`, interior elimination, joint factors
   as the norm rather than the exception, incident-factor tracking (§5), and fill-in-aware policies.
   This is the phase the dynamic smoother actually needs, and it is deliberately last among the
   estimation phases because every mechanism it uses is exercised by phase 4 first.
6. **Multiple sequences.** `SequenceId` plumbing and `Associator`.

Phases 1 and 2 are the ones to start, in that order. Together they prove the build-advance-delete
loop end to end, and their requirements are already fully specified elsewhere.

## 12. Verification

Follow the existing layout — per-backend entry points over a shared `common/` module, as in
`skoro-linalg/tests/{ndarray_f32,ndarray_f64,wgpu_f32}.rs` and `skoro-se3/tests/common/mod.rs:296-306`.
Heed the workspace's own doctrine (`.claude/skills/skoro-diff-operation/references/testing.md`):
hand-computed constants generated from the implementation lock in whatever the implementation
already did.

The linear-algebra layer beneath these has its own tests in `estimator-linear-algebra.md` §9 — the
conversion round-trips and the batch-1 guard in particular, since those are the new code the checks
below run on top of.

1. **Exactness at retention-1.** On a linear-Gaussian model the build-advance-delete loop must
   reproduce the closed-form Kalman filter, not merely converge to it.
2. **Advance/retract are distinct.** A model where `f` is not the identity must give different
   answers for `advance` and `retract_flat` with the same tensor — a test that fails if the two are
   ever conflated.
3. **Window/batch agreement.** A fixed-lag model with retirement and the equivalent batch model must
   give the same posterior on a linear-Gaussian problem, where marginalization is exact.
   This is the test that justifies the whole retirement mechanism.
4. **Tag survival.** `Q<B, S_j>` evaluates identically before and after its tick's predecessor
   retires. The user's requirement that tags keep naming quantities, made checkable.
5. **Bounded growth, at every policy.** `graph.nodes.len()` is bounded across many ticks — including
   at **retention-1**, where the bound is about two ticks (§3) and not one. Also assert
   `log_likelihoods.len()` bounded, and that no `Node::LogDensity` sits in `graph.nodes` while absent
   from `log_likelihoods` — the orphan that pins its parent cone against removal
   (`graph-marginalization.md` §8.1). `Chain::len()` tracks the retention policy. Catches capture
   leaks (§7).
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
8. **Hoisting is sound, within a tick.** A `CompiledSequence` compiled once and reused across a
   tick's several passes agrees with per-call recompilation. The negative half matters as much: a
   sequence must **not** survive a tick boundary, and a test that reuses one across a structural
   change must fail (§8).

The next four are new, and they are the ones that pin the per-tick construction story. Write 9 and 10
first — they are the ones a template-recycling implementation passes by accident.

9. **Exogenous data actually enters.** Two ticks with identical `x_k` but different `dataₖ` produce
   different `x_{k+1}`, and different `F`. An implementation that quietly reuses last tick's
   structure passes every test above and fails this one.
10. **Capture is exact.** The set `TickCtx` captured for tick `k` equals the set removed at teardown:
    nothing of the tick survives, and nothing still live is taken. Check across the numeric step in
    the middle of the tick (§3), since a capture that resets between the two passes loses the
    measurement nodes.
11. **Conditional measurements.** A tick with no measurement (`M = 0`) is predict-only and leaves `P`
    at exactly the predicted covariance. A run of ticks with *different* measurement subsets agrees
    with a batch model containing the same factors — which is what makes the visitor's "sometimes
    absent" behaviour a modelling statement rather than a bug.
12. **Interior retirement.** Retiring `t_i` from a retained `t_{i-1}, t_i, t_{i+1}` produces a
    two-element blanket and a single joint factor over the neighbours, and the result agrees with the
    batch posterior on a linear-Gaussian model. This is the keyframe analogue of test 3 and the gate
    on `graph-marginalization.md` §5 being the *common* path rather than the exotic one.

Run with `cargo test --workspace`. The wgpu suites additionally exercise the device path.
