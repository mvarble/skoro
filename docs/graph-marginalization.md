# Graph Marginalization

The graph surgery that lets a model forget: giving a stochastic node a new distribution, replacing a
correlated joint with a likelihood factor, and removing what no longer means anything.

Companion to `sequential-estimation.md`, which specifies *who* may marginalize and *when*. This
document specifies *how*, in terms of `skoro-graph`. It is phase 4 of that document's ordering — a
retention-1 filter needs the *removal* half of this document (§8's defects, in particular) but none
of the marginalization half, because its covariance already is the marginal.

## 1. What the graph means, and what marginalization does to it

> The `Graph` encodes a conditional distribution of quantities `(x₀ … xₙ)` given `(z₀ … z_m)`.
> Marginalization projects it to `(x_k … xₙ)` given `(z_j … z_m)`.

Everything below follows from reading that literally.

- **Surviving stochastic nodes were already stochastic.** `x_k` does not become a random variable
  during marginalization; it always was one. It acquires a *different distribution* — the projection
  of the joint onto its coordinates.
- **Intermediate quantities stop meaning anything.** `f(x₀)` was a deterministic function of a
  quantity that no longer exists. It is not reinterpreted, re-rooted, or approximated; it is removed,
  along with the density terms that referenced it.
- **The conditioning set shrinks too.** An observation whose likelihood is absorbed into the new
  marginal drops out of `(z₀ … z_m)`. This is why the retirement closure reaches observed nodes and
  not only stochastic ones.

The load-bearing rule, and the one that determines the whole design:

> **Mapped nodes never become stochastic.** Randomness is never invented where there was a
> deterministic function.

That rule is absolute. A second one is an **amendment** to an earlier, stricter formulation, and is
called out as such because it is easy to read past:

> **Only density-carrying nodes are retired** — not, as first stated, only *stochastic* nodes.

The two differ because a joint marginal is carried by a likelihood factor over `Parameter`s (§5).
After the first retirement a window head is a `Parameter` whose density lives in that factor, and the
next retirement must be able to reach it. The spirit of the stricter rule survives: demotion
`Stochastic → Parameter` *removes* structure, which is the opposite of the mapped-to-stochastic
promotion the first rule forbids.

Two requirements shape the rest:

> **The tags survive.** After retirement, a surviving `Q<B, Sⱼ>` still names a quantity in the model,
> at the same `Id`, with the same quantity type. A user who stored a `Q` in a struct — which they
> can, since `Q` owns its client and has no lifetime (`skoro-model/src/q.rs:21-28`) — must not find
> it silently broken.
>
> **The prior is joint.** The true marginal over `S₀ … Sₘ` is correlated. Applying independent
> marginals to each is not an approximation of it, it is a different model — and for a filter, an
> overconfident one.

## 2. Re-rooting a stochastic node already exists

For a single-node blanket, marginalization needs **no new graph primitive at all.**

> **Amendment.** An earlier formulation called this "the Markov-chain case, and the common one". The
> first half stands; the second does not. A single-node blanket arises when the *oldest* retained
> tick is retired — the fixed-lag case. Under the keyframe retention of `sequential-estimation.md`
> §3, *interior* ticks are retired routinely, and an interior tick's blanket is its neighbours on
> both sides. So §5's demote-and-factor path is the norm for a dynamic smoother, and the path below
> is the special case. Neither is optional; the mechanism is picked by blanket cardinality, never by
> the policy.

`Graph::update_stochastic` (`graph.rs:527`) does exactly the required job:

- validates the node is `Stochastic` and the sampler's kind matches (`graph.rs:537-548`);
- rejects dangling parents and cycles (`graph.rs:560-582`);
- rewires the `children` registry off the old parents and onto the new (`graph.rs:584-606`);
- unregisters the old density from `log_density_to_stochastic`, `stochastic_to_log_density` and
  `log_likelihoods` (`graph.rs:609-620`);
- swaps in the new sampler and parent set (`graph.rs:623-624`).

`ModelClient::update_quantity` (`model.rs:617-640`) wraps it and then builds and marks the
replacement density. Together they *are* re-rooting: `x_k`'s distribution stops being
`Normal(f(x_{k-1}), Q)` and starts being its marginal.

Once that rewire has happened, the retired cone is childless and `Graph::remove` (`graph.rs:740`)
clears it unchanged. There is no obstruction and no new kind-changing operation.

This path applies when the marginal **factorizes** — a single-node blanket whose marginal is a
distribution in its own right, which is what retiring the head of a chain gives you. When the blanket
has more than one element the marginal is correlated and does not factorize, and §5 takes over: the
blanket is demoted to `Parameter`s and the marginal becomes a likelihood factor. The two paths share
the same bookkeeping; they differ in where the density ends up.

### The cycle check runs upward, not downward

`update_stochastic_impl` (`graph.rs:567-582`) already walks *up* from the new `parent_ids` through
`parent_ids()`, erroring if it reaches the node being re-rooted. That is the correct direction and it
is already written — reuse it rather than restating the condition.

Walking downward for descendants would be both wrong and nearly vacuous: re-rooting *replaces* the
parent set, so the new parents are the prior's parameters — a `Tril` covariance constant, typically —
and never the retired state. The check earns its keep only when a caller supplies an existing node as
a prior parameter.

### Hazard: stale caches

`add_mapped_impl` constant-folds at registration through `check_cache` (`graph.rs:1282-1326`): if
every parent has a concrete value, it lowers the op, runs `forward` at `batch_size = 1`, and stores
the result on the node as `Some(ConstantValue)`. And `is_differentiable_parent` (`graph.rs:71-86`)
treats a cached mapped value as **non-differentiable**:

```rust
Node::Mapped(MappedNode { value, quantity_type, .. }) =>
    quantity_type.kind() == QuantityKind::Continuous && value.is_none(),
```

Changing a node's parents invalidates every constant fold downstream of it. Left stale, those
descendants are silently dropped from the tape: the values stay right and the derivatives come back
wrong, which is the worst failure mode available.

`marginalize` must therefore clear cached values on every descendant of a re-rooted node. **Do not
write a second walk:** `unobserve` (`graph.rs:496-518`) already does exactly this, breadth-first over
`children`, clearing `value.take()` on `Mapped` and `LogDensity` nodes and carrying a `visited` set.
Factor it out and call it from both.

Take `unobserve`'s version specifically. The corresponding walk in `observe` (`graph.rs:415-461`) has
**no `visited` set**, so on a diamond DAG it re-checks a descendant once per path — see §8.

## 3. The stochastic blanket

The structural half of retirement, and the only part the graph can do alone:

```rust
/// The quantities that need a new distribution once `retire` is integrated out: walk
/// down through `children`, passing through `Mapped` and `LogDensity`, and stop at the
/// first node of any other kind on each path.
pub fn stochastic_blanket(&self, retire: &[Id]) -> SkoroResult<Vec<Id>>;
```

The stopping rule is *kind-agnostic by construction*. `Mapped` and `LogDensity` are the only kinds
that have both parents and children, so stopping at "anything else" means stopping at the downstream
**leaves** — `Stochastic`, `Parameter`, and `Observed`. Stating it that way matters, because after a
previous retirement a window head is a `Parameter` whose density lives in a likelihood factor (§5),
and a walk written to look for `Stochastic` would run straight past it into the factor.

> **`Observed` terminates the walk but is never an element of the blanket.**

This distinction is easy to collapse and expensive to get wrong. An `ObservedNode` is a fixed value
(`node.rs:122`) — there is nothing to distribute over. When the walk reaches an observed `y` whose
density depends on the retired set, that likelihood is *absorbed* into the new marginal, and both the
density node and `y` itself join the dead set. `y` leaves the conditioning set, exactly as §1
describes.

The kind-agnostic stopping rule is also what makes *repeated* elimination work. After one interior
retirement the surviving neighbours are `Parameter`s carrying a joint factor (§5); the next
elimination has to reach them, and a walk written to look for `Stochastic` would run straight past
into the factor. Under keyframe retention that is not a corner case — it is most retirements after
the first.

Many models never call this. As `sequential-estimation.md` §5 argues, a model built through a
per-tick `Transition` has the Markov property by construction, so tick `k`'s blanket is exactly tick
`k+1`'s state — a field lookup rather than a graph walk. But that holds for the *append* direction
only: once interior retirements start installing joint factors, a tick's blanket is the union of
retained ticks reachable through the factors incident on it, and the temporal layer must track
incidence to keep it a lookup (`sequential-estimation.md` §5). `stochastic_blanket` is the escape
hatch for models not built through `Transition`, and the reference implementation to test either
answer against.

### The dead set

Everything that stops meaning anything. Computable with machinery that already exists:

1. **Candidates** — `retire` plus everything reachable downward from it through `children`.
2. **Live** — the ancestor closure of the *surviving* `log_likelihoods`, which is precisely what
   `topological_sort(targets)` computes (`graph.rs:1332`, a backward walk through `parent_ids`). Run
   it *after* the blanket rewire, so the severed edges are already gone.
3. **Dead** = candidates ∖ live.

The old density of a re-rooted node falls out of this correctly. `update_stochastic_impl` unregisters
it from the three maps but leaves the node in `self.nodes` (`graph.rs:609-620`), so it is a candidate
(it descends from the retired set) and is no longer live (it is no longer a likelihood). It is dead,
and gets removed — which incidentally fixes the orphan leak of §8.1 along this path, though not for
standalone `update_stochastic` calls.

## 4. Separability, and the trap in stating it naively

Marginalization has a precondition:

> An intermediate quantity `g(x₀, xₙ)` that depends on both a retired quantity and a surviving one is
> a **runtime error**.

Right in spirit, but **it must not be checked on ancestry alone** — an ancestry-only check rejects
the one model this design exists to serve.

`Normal::log_density` (`skoro-normal/src/distribution.rs:168-178`) builds exactly that shape:

```rust
let colored_tangent = self.mean.inverse_retract(x);
```

For `x_k ~ Normal(f(x_{k-1}), Q)` this is a `Mapped` node whose parents are `[f(x_{k-1}), x_k]`,
mixing the retired `x_{k-1}` and the surviving `x_k`. Every Gaussian Markov chain contains one per
tick.

The distinction is not ancestry but **consumption**:

> An entangled intermediate is fine when its only descendants are density terms that are themselves
> being absorbed. It is an error when it is an ancestor of something that **survives**.

`colored_tangent`'s only descendant is `x_k`'s transition density, which marginalization removes and
replaces. So it lands in the dead set as collateral and the chain marginalizes cleanly. This is the
regression test in §9 that a naive implementation fails.

Two failures remain, and they deserve different errors because only one is the caller's to fix:

1. **Insufficient blanket.** The entangled node feeds a density over another *surviving* variable —
   `y ~ N(g(x₀, xₙ), R)` with `y` observed and `xₙ` outside the supplied blanket. Resolvable: absorb
   that factor too and put `xₙ` in the blanket. `stochastic_blanket` computes a blanket that avoids
   this by construction, so the error fires only when a caller supplies a smaller one.
2. **Irreducibly entangled.** The entangled node is consumed by something that is not a removable
   density — a surviving mapped node the user still holds a `Q` to. There is no factor to absorb and
   no blanket that helps. Marginalization is impossible for this model.

Both are structurally what `remove`'s `NodeStillReferenced` check (`graph.rs:743-751`) already
catches from the other side: a dead node with a live child. But that error names a child id and says
nothing about marginalization. Detect it up front in `marginalize` and report the retired quantity,
the surviving quantity, and the entangling intermediate — the three things a user needs to restructure
their model.

## 5. Joint marginals are likelihood factors

When the blanket has more than one element the marginal is correlated and does not factorize. The
representation is:

> **Demote each blanket quantity to a `Parameter`, and attach the joint marginal as a single
> free-standing likelihood factor over all of them.**

**Implemented.** `Graph::demote_to_parameter`, surfaced as `ModelServer::demote_to_parameter`,
`ModelClient::demote_to_parameter`, and a `ModelClientOps` trait method taking a `&Q<B, T>`:

```rust
/// Demote a `Stochastic` node to a `Parameter` at the same `Id`: drop its sampler and
/// parent edges, and unregister its density.
///
/// Demoting an existing parameter is a no-op. Errors `NotStochastic` / `NodeNotFound`.
pub fn demote_to_parameter(&mut self, id: Id) -> SkoroResult<()>;
```

It is the only new primitive, and it reuses two helpers factored out while building it:
`unregister_density`, extracted from `update_stochastic_impl`, and `clear_cached_descendants`,
extracted from `unobserve`. Covered by `skoro-normal/tests/demote.rs`.

Note on the cache walk: it is **defensive, not load-bearing**. `check_cache` yields `None` for a
`Stochastic` parent *and* for a `Parameter` one, so nothing downstream of a stochastic quantity is
ever constant-folded and demotion cannot strand a fold. The call guards the invariant — the node's
value is no longer determined by its parents — against a future widening of the fold rule. Where the
walk *is* load-bearing is `unobserve`, since an `Observed` node does carry a value and its
descendants genuinely do fold.

The factor itself needs nothing new. `Graph::mark_likelihood` (`graph.rs:252`) converts a mapped
real-valued node into a `LogDensity` with `stochastic_id: None` and pushes it onto `log_likelihoods`
(`graph.rs:290`) — a free-standing factor belonging to no single quantity, which is exactly what a
joint marginal is.

Three properties fall out, and they are what make the representation work:

- **Demoted quantities are ordinary Jacobian columns.** `Parameter` is matched by the VJP collection
  loop alongside `Sampler` (`compiler.rs:354-357`) and seeded with tangents and frames by
  `non_op_forward` (`compiler.rs:525-545`). The EKF's `H = ∂h/∂x` over a marginalized state needs no
  special handling at all.
- **The bookkeeping is untouched.** A factor has no owning stochastic node, so
  `log_density_to_stochastic: HashMap<Id, Id>` (`graph.rs:43`) never has to represent one density
  over many quantities.
- **Rolling windows accumulate no structure.** Retiring one member of a joint means dropping the old
  factor — which references a retired quantity and so lands in the dead set on its own — and
  attaching a smaller one over the survivors. There is no joint *object* to carve up, so a window
  that retires one quantity per tick neither grows nor needs a second kind of surgery.

The tags survive as §1 requires: same `Id`, same quantity type, same downstream edges. What a demoted
quantity loses is its own sampler and its *observability* — `observe` requires `Stochastic` or
`Observed` (`graph.rs:357-372`).

### What this costs, honestly

A correlated joint is expressed for *inference*, not for *generation*, and the representation is
chosen accordingly. The consequences:

1. **A correlated joint prior cannot be sampled.** Demoted quantities have no samplers, and a
   likelihood factor describes a density without describing how to draw from it, so `generate` cannot
   produce a correlated draw over them. This is the deliberate trade: marginalized models are
   inference-only, `log_density` is the authority, and samplers on marginalized quantities are
   best-effort at most.

   It is narrower than it sounds. `Normal<B, M>` already expresses correlated *same-manifold*
   quantities as a **single** node with a `Tril` covariance factor, and that stays fully generative.
   What is lost is the cross-manifold case — a jointly Gaussian `(SE3, Euclidean3)` one wants to
   forward-simulate. The structurally right answer there is a product manifold, which does not exist
   today and is not foreclosed by anything here.
2. **`generate` degrades to `generate_from`, not to failure.** A `Parameter` returns `MissingPoint`
   whenever it is absent from `points`, even under `use_model_samplers: true`
   (`compiler.rs:525-530`), and `generate(n)` is just `generate_from(&empty, n)` (`graph.rs:980`).
   Any model with a free parameter already required seeding, so this is not a new class of loss.
3. **Finding the factors to absorb needs a walk, not a lookup.** A factor has `stochastic_id: None`
   and so appears in neither direction of the density bookkeeping; answering "which factors touch
   this quantity" means walking `children`. The dead-set computation of §3 already performs exactly
   that reachability, so no new mechanism is needed.
4. **Interior elimination causes fill-in.** Retiring a quantity installs a factor directly between
   its blanket members, so the survivors acquire edges they did not have. Under the keyframe
   retention of `sequential-estimation.md` §3 this compounds: eliminate enough interior ticks and a
   sparse chain becomes a dense clique over the keyframes, and every subsequent blanket is larger
   than the last. This is ordinary variable-elimination fill-in and the representation above does not
   avoid it — nothing can, at fixed accuracy. What it means for the caller is that **elimination
   order is part of the decision**, not an implementation detail: a retention policy should prefer
   eliminations with small blankets, and `RetentionPolicy::select` returns an ordered list for this
   reason. The bound to watch is `log_likelihoods.len()` and the arity of the largest factor, not
   `nodes.len()` alone.

### Why there is no `FactorGraph`

Costs 3 and 4 are both adjacency questions — *which factors touch this quantity*, *which quantities
would this elimination couple* — so it is fair to ask why a bipartite variable↔factor structure is
not stored alongside `Graph`. It is the structure iSAM and its relatives are built on, and the
absence is deliberate rather than an oversight.

**No documented estimator queries adjacency.** All of them are layout-driven: `jacobian_jvp` and
`jacobian_vjp` (`flattened-tangent-spaces.md` §5) take caller-supplied row and column layouts, and
whatever built the nodes already knows their ids. An EKF's `F` and `H`, a Levenberg-Marquardt `J`
over free leaves and whitened residuals — none of them ever asks the graph what is connected to what.
The one algorithm that structurally *is* an adjacency query, `stochastic_blanket` (§3), is answered
by construction for `Transition`-built models.

> **That agreement carries an expiry condition, and it is a scope decision rather than a fact about
> estimators.** What makes adjacency unnecessary is `estimator-linear-algebra.md` §5's commitment to
> dense, host `f64`, batch 1, `N` in the tens. A factor graph earns its keep by exploiting sparsity —
> block-sparse `J`, sparse Cholesky, an elimination ordering — and none of that is reachable at those
> sizes. A batch smooth over hundreds of poses would revoke the argument.

**The cost of adding one is larger than it looks.** `Graph` is a *generative DAG*, not a factor
graph: a factor is a `LogDensity` node sitting atop an arbitrary computation subgraph, since
`Normal::log_density` builds `inverse_retract` → `vector_product` → … beneath it. A `FactorGraph`
would therefore be a **derived projection**, not a second source of truth, and it would need
invalidating on `observe`, `unobserve`, `update_stochastic`, `demote_to_parameter`, `remove` — and on
**every node constructor**, because `add_mapped` can make a new leaf an ancestor of an existing
`LogDensity` and so change that factor's variable set. The invalidation surface is the whole
mutation API, not five entry points.

That would be a fourth index over the same arena, joining `log_likelihoods`,
`log_density_to_stochastic` and `stochastic_to_log_density` — three that §8.1 already documents
falling out of sync with `nodes`, in a workspace with no working invalidation mechanism anywhere:
stale constant folds are repaired by a manual `clear_cached_descendants` walk, and compiled sequences
have no invalidation at all (`sequential-estimation.md` §8). Fix the three that leak before adding a
fourth with a harder contract.

**The one real cost here needs locality, not adjacency.** §3's dead-set computation takes `live` as
the ancestor closure of the *surviving* `log_likelihoods` via `topological_sort` (`graph.rs:1332`) —
O(live model) per retirement, which under keyframe retention runs on most ticks. Bipartite adjacency
would not fix it; the ancestor walk is still the ancestor walk. The fix is a **locality bound**: a
retired tick's dead set is confined to its captured node set plus the factors severed at step 4 of
§7, which the `TickCtx` capture of `sequential-estimation.md` §7 already supplies. A known cost with
a known fix, and no new structure in it.

**Where the need genuinely appears, it is already conceded — at a fraction of the scope.**
`sequential-estimation.md` §5's `incident: HashMap<TickRef, HashSet<Id>>` tracks the joint factors
installed by retirement. It is strictly smaller than variable↔factor adjacency — tick-keyed, and
blind to process and measurement factors — but it appears in exactly the place theory predicts: a
keyframe policy choosing *which* tick to eliminate must compare candidate blankets before committing
to one, and that is the query pattern a walk-per-query serves worst. Note it is a **policy** cost,
not a mechanism cost. Any single retirement is fine with a walk; choosing among retirements is what
wants an index. If a `FactorGraph` is ever built here, that is the pressure that will justify it, and
`incident` is the thing it should grow out of.

## 6. The graveyard

Retirement destroys nodes that user code still holds `Q<B, T>` handles to. Those handles must fail
loudly rather than silently, and the failure must say *why*.

```rust
pub struct Epitaph<B: Backend> {
    pub quantity_type: Option<Arc<dyn DynQuantityType<B>>>,
    pub reason: RetirementReason,
    pub epoch: u64,
}

pub enum RetirementReason {
    /// Integrated out by a marginalization.
    Marginalized,
    /// A mapped or density node that died with its ancestors.
    Collateral,
    /// Explicitly removed by the user.
    Removed,
}
```

**One choke point.** `register_children` (`graph.rs:1269`) is the single function every node
constructor funnels through — `add_stochastic_impl` at `graph.rs:189` and `add_mapped_impl` at
`graph.rs:230`. Check the graveyard there and return `SkoroError::QuantityRetired { id, reason }`
instead of `DanglingParent`, which cannot distinguish "never existed" from "marginalized away".

**Fallible APIs error; operator sugar must panic.** This split is forced, not chosen:
`std::ops::Add for Q` returns `Q`, not `Result`, so `a + b` over a retired quantity has nowhere to
put an error. What the graveyard buys there is accuracy. Today `add_mapped_impl` panics with

```rust
self.register_children(id, &parent_ids)
    .expect("internal: mapped node added with dangling parent");
```

under a comment asserting that a dangling parent here "indicates an internal bug rather than user
error" (`graph.rs:227-231`). After marginalization that comment is false: it is ordinary user error,
and the panic should name the id, the reason and the epoch. Meanwhile `add_quantity`, `observe`,
`update_quantity` and the `evaluate*` family all return `SkoroResult` already and should surface
`QuantityRetired`. Add `Q::is_retired()` for a non-panicking pre-check.

`check_cache`'s `.expect("Parent node should be in graph.")` (`graph.rs:1284`) is reachable the same
way and needs the same treatment.

**Bounding growth.** Ids are never reclaimed — `IdGenerator` is a monotonic `AtomicU64`
(`skoro-core/src/id.rs:24`) and `remove` does not return them — so an unbounded graveyard is a leak.
Prune with the mechanism already in the workspace: `NodeDropGuard` holds an id and an
`Arc<SegQueue<Id>>` and pushes on drop (`model.rs:408-426`); `ModelServer::cleanup()` drains it
(`model.rs:198`). Make the push unconditional and let `cleanup` disambiguate — id in the graveyard ⇒
prune the epitaph, mapped leaf ⇒ remove the node. Since all `Q` clones share one
`Arc<NodeDropGuard>` (`q.rs:39`), an epitaph outlives exactly the handles that could misuse it.

**This reuse is not free.** The graph cannot reach a guard at retirement time — the guard is owned by
`Q`, which is owned by user code — so guards cannot be marked retroactively. `cleanup()` must
therefore do a graveyard lookup on *every* dropped `Q`, not only retired ones, turning an atomic load
into a `HashMap` probe. `cleanup()` runs at the head of `add_constant`, `add_parameter` and
`add_mapped` (`model.rs:496, 517, 580`), so this is a mildly hot path. Cheap, but real, and worth
measuring rather than assuming.

One residual case: a node retired while no `Q` referenced it leaks an epitaph nothing will ever
consult. Provide `Graph::sweep_graveyard()` for explicit control; the `epoch` field also makes an
age-based policy possible.

## 7. The retirement sequence

In order, and the order is forced:

1. **`stochastic_blanket(retire)`** — structural; the graph alone. Which quantities need a new
   distribution. Observed terminators are noted but not included (§3).
2. **Separability check** — no dead-set node may be consumed by a survivor (§4). Fails here, before
   any mutation, with an error naming the entangling intermediate.
3. **The marginal.** The designated `prior_source` estimator computes the joint marginal over the
   blanket at the current linearization point and materializes it as graph nodes.
   *This step cannot be done by the graph* — see `sequential-estimation.md` §6. The projector's
   numerics are host `f64` like the rest of the estimator's state
   (`estimator-linear-algebra.md` §1); only the materialized `Tril` factor crosses back into the
   graph as a tensor.
4. **Install it.** If the blanket has one element and its marginal factorizes, `update_stochastic`
   re-roots it in place (§2). Otherwise `demote_to_parameter` each blanket element and attach the
   joint marginal with `mark_likelihood` (§5). Either path clears downstream constant folds via the
   walk factored out of `unobserve`.
5. **Remove the dead set** — now legal, because step 4 severed the blanket's parent edges. The
   previous joint factor, if any, falls in on its own: it references a retired quantity. Every
   removed id gets an epitaph (§6).

Steps 1, 2, `demote_to_parameter`, and step 5's graveyard bookkeeping are new. Everything else is
existing API. Step 3 lives outside `skoro-graph` entirely.

## 8. Defects to fix along the way

Following the precedent of `flattened-tangent-spaces.md` §5, "A defect to fix along the way". Each is independently reachable
from a public path and none should ship as-is.

One was found and fixed while implementing §5:

> **`unobserve`'s guard arms were inverted.** It matched `Stochastic => {}` and
> `Observed => return Ok(())`, then immediately destructured the node as `Observed` with an
> `unreachable!()` fallback. So `unobserve` on an actually-observed quantity silently did nothing,
> and on an already-stochastic one panicked. There were no tests for `unobserve` anywhere in the
> workspace. Fixed, with a regression test in `skoro-model/tests/plumbing.rs` that fails against the
> old arms.

Two of them are now **phase blockers** rather than cleanup, because `sequential-estimation.md` §3 has
every retention policy — including retention-1 — building and deleting nodes on every tick. Removal
is on the hot path of the first phase, and it is the least-tested path in `skoro-graph`.

What remains:

1. **Orphaned densities — a phase-4 blocker.** `update_stochastic_impl` drops a density from all
   three bookkeeping maps (now via the extracted `unregister_density`) but leaves the `LogDensity`
   node in `self.nodes`; `demote_to_parameter` inherits the same behaviour. The marginalization path
   sweeps these up via the dead set (§3), but a standalone call still strands one.

   Sharper than first stated, because it is not merely a leak: the orphan **keeps its `parent_ids`**,
   so its parents' `children` sets stay non-empty and it blocks `Graph::remove` on the *entire cone
   above it*. Nor can it be reclaimed automatically — `mark_density` cleared its `is_mapped_leaf`
   flag (`model.rs:129`) so `cleanup()` never sees it, and `update_quantity` dropped the only
   `Q<B, Real>` naming it (`model.rs:627-636`), so no caller has an `Id` to pass to `remove`. The
   only escape today is walking `graph.nodes` for `LogDensity` nodes absent from `log_likelihoods`.

   It is *not* a phase-1 blocker: `unregister_density` runs only from `update_stochastic` and
   `demote_to_parameter`, and a retention-1 filter calls neither. It becomes one the moment a window
   is retained.
1a. **`Graph::remove` has no transitive teardown helper.** It is leaf-only and all-or-nothing —
   `NodeStillReferenced` if any target has a child outside the batch (`graph.rs:714`) — so every
   caller must compute the downward closure itself and pass a closed batch. Retiring a tick needs
   exactly that closure, at every retention policy, so the helper belongs in `skoro-graph` next to
   `stochastic_blanket` rather than in each caller. **Phase-1 blocker.**
2. **`observe`'s descendant walk lacks a `visited` set** (`graph.rs:415-461`), unlike
   `clear_cached_descendants`. Quadratic on a diamond DAG. Fold it into that helper.
3. **`Graph::add_mapped` returns a bare `Id` and panics on a dangling parent**
   (`graph.rs:207, 230-231`), where every sibling constructor returns `SkoroResult`. This is the
   graveyard's motivating case (§6): the panic's comment claims user error is impossible here, and
   marginalization makes it routine.
4. **`check_cache`'s `.expect("Parent node should be in graph.")`** (`graph.rs:1284`) — reachable
   from a retired parent, same treatment as 3.
5. **`Graph::remove`'s unresolved `// TODO: check if I need to remove children`** (`graph.rs:760`) —
   retirement exercises this path hard and it should not be uncertain. **Phase-1 blocker**: per-tick
   teardown runs this path once per tick from the very first phase, and nothing tests it today.
6. Carried from `flattened-tangent-spaces.md` §5, "A defect to fix along the way": `compiler.rs:363` panics
   `"Point should have a cotangent vector."` when a continuous leaf in the compiled subgraph
   receives no tape write. Default to zeros or return an error.

## 9. Verification

Per-backend entry points over a shared `common/`, following
`skoro-linalg/tests/{ndarray_f32,ndarray_f64,wgpu_f32}.rs`. Note that `skoro-graph` has 173 test LOC
against 3,097 source LOC, and all four tests in `graph_tests.rs` use `unimplemented!()` dummies with
zero evaluation calls — so no evaluation path is covered there today. These tests are building on
thin ice and must exercise real operations.

Not all of these are equal. The gates are where the bugs will be. Write them first.

`skoro-normal/tests/demote.rs` already covers the demotion half (items 1, 3 and 4 below); the rest
land with the retirement machinery.

### Gates

1. **Demoted quantities are Jacobian columns.** ✅ *Covered.* A `Parameter` whose only density comes
   from a joint factor must still receive cotangents — without this an EKF cannot differentiate a
   measurement with respect to a marginalized state, and the whole representation collapses. It
   passed on the first run, which is the point of §5: no machinery is needed. Extend it to finite
   differences through the retraction, per
   `.claude/skills/skoro-diff-operation/references/testing.md`, rather than the hand-computed
   constants it uses today — in this workspace such constants are generated from the implementation
   and lock in whatever it already did.
2. **Separability, both directions.** A Gaussian Markov chain marginalizes cleanly *despite*
   `Normal::log_density`'s entangled `inverse_retract` node (`skoro-normal/src/distribution.rs:169`)
   — the regression that a naive ancestry-only check fails (§4). An irreducibly entangled model
   errors, naming the intermediate. **This is the highest-value gate that remains**, and the one
   that fails silently.
3. **Demotion detaches the density.** ✅ *Covered.* A demoted quantity stops contributing its own
   term to `log_density`, its parent edges are gone, and its `Id`, quantity type, and dependents are
   untouched.
4. **A joint factor scores the model.** ✅ *Covered* for the structural claim: two demoted quantities
   under one `mark_likelihood` factor score correctly and both receive gradients. Still to add: the
   closed-form Gaussian value, once a marginal projector exists to produce it.
5. **Correlation is preserved.** A joint factor with non-diagonal covariance must produce a different
   posterior than the product of its marginals. Guards §1 against an implementation that quietly
   factorizes — the most plausible wrong implementation, and one that passes every test above.

Note what is *not* on this list any more. An earlier draft made "demotion invalidates caches" the
top gate; that was wrong. `check_cache` yields `None` for `Stochastic` and `Parameter` parents alike,
so nothing downstream of a stochastic quantity is ever folded and demotion cannot strand one. The
cache walk is a defensive guard, not a repair — see §5.

### Rolling behaviour

5a. **Per-tick teardown.** Ahead of everything else on this list in time, because
   `sequential-estimation.md` §11 phase 1 depends on it: a tick's captured node set is removed in one
   `Graph::remove` call, the call succeeds (nothing outside the batch still references it), and
   afterwards no `Node::LogDensity` sits in `graph.nodes` while absent from `log_likelihoods`. This
   is the gate on defects 1a and 5 of §8, and the only one on this list a retention-1 filter needs.
6. **Factor replacement.** Retiring one member of a joint drops the old factor — verify it leaves via
   the dead set rather than by explicit bookkeeping — and installs a smaller one over the survivors.
   A rolling window over many ticks must keep both `graph.nodes.len()` and `log_likelihoods.len()`
   bounded.
7. **Graveyard.** Building from a retired `Q` returns `QuantityRetired` on fallible paths and panics
   informatively on operator sugar; epitaphs are pruned when the last `Q` drops.

### End-to-end

8. **Exact where it should be.** On a linear-Gaussian model, marginalization is exact, so a retired
   window and the equivalent batch model must agree. The justification for the whole mechanism, and
   shared with `sequential-estimation.md` §12.

### Structural checks

Cheap, expected to pass immediately. Worth having as regressions, not worth writing first.

9. **`stochastic_blanket` agrees with construction.** On a `Transition`-built chain, the computed
   blanket equals the next tick's state; an observed terminator is excluded from it; and a
   previously-demoted `Parameter` head *is* included — the case a `Stochastic`-only walk misses.
10. **Removal succeeds after the rewire and fails before it.** `NodeStillReferenced` before step 4,
    `Ok` after. Pins the ordering of §7.
