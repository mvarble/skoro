# Skoro: what is built, and what inference still needs

This file replaces a document of the same name that was deleted incidentally in `785ff1a`
("Cleaning up graph operations and invoking tests"). Recover the old text with
`git show 785ff1a^:docs/skoro.md` — 798 lines, worth reading once for its argument and not
otherwise, since it is stale in both directions: it names `NlsStructure`, `ResidualCertificate`,
`record_joint` and a `skoro-levenberg-marquardt` crate that are nowhere in the tree, while four of
its five gaps and its one defect have since closed. **Its section numbering is deliberately not
preserved**; the comments that cited it have been repointed here.

Organized by what a caller can rely on rather than by the crate stack. §1 and §2 are the surfaces
that exist and are tested. §3 is marginalization, expressed in the graph and computed on the host by
partial elimination. §4 is the estimator's view, including the block-sparse elimination underneath it. §5 is the gap ledger, ordered by which algorithm
each gap blocks. §6 records where the in-tree documentation has drifted from the code.

Every claim below names a path. Every claim that something *does not exist* was checked, not
assumed, and §5 gives the searches.

---

## 1. What model-building guarantees today

### 1.1 The two objects, and the two tiers

`ModelServer<B>` owns the graph; `ModelClient<B>` is a cheap cloneable handle to it
(`crates/skoro-model/src/model.rs`). The only routes from a client to the `Graph` itself are
`ModelClient::server_call` and `server_call_mut`, so everything else goes through one of two traits.

That split is the load-bearing design decision of the model tier, and its reason is worth quoting
(`crates/skoro-model/src/client_ops.rs`, on `ModelBuilderOps`):

> Every call that changes the log-density `p`, the proposal `q`, the set of free or observed
> quantities, or the batch sizes the model accepts is here: adding terms and parameters,
> observing, demoting, and removing. `ModelReaderOps` may add nodes, but only ones that change
> none of those.

The line is drawn by what a call changes, not by whether it can break a reference. `observe`,
`demote_to_parameter`, `add_parameter` and a leaf `remove` cannot break a reference, and each
changes the model being estimated, so all of them are builder calls. `Q` has no `observe` for the
same reason: a `Q` carries a full client, so a method on it would reach past a read bound.

The two removals differ only in how they treat references:

| | Behaviour |
| --- | --- |
| `remove` | Leaf-only, all-or-nothing. Refuses a node a survivor still points into (`NodeStillReferenced`) **or still binds** (`NodeStillBound`). Will not break a reference. |
| `remove_closure` | Takes the downward closure, and the referrers with it. Cannot fail on a live child. |

An estimator holding only `ModelReaderOps` therefore cannot change the model it was handed. Each
negative claim is a compile-fail case in `crates/skoro-model/tests/compile_fail/`, asserted from
`tests/bindings.rs`.

There is deliberately no tier between the two. Marginalizing is demotion, a joint factor over the
blanket, and a closure removal (§3), and a tier that granted the first alone would permit a demoted
quantity with nothing compensating for it.

### 1.2 Ownership

A `Q<B, T>` is the ownership token; a bare `Id` is a weak reference. `cleanup` is a mark-and-sweep
garbage collector, which runs in two phases:

- **Mark** walks out from a set of *roots*, the nodes that must survive, and marks every node it can
  reach. Marked means *keep*.
- **Sweep** goes over every node in the graph and removes each one the mark phase did not reach.

So the traversal is over the ids to keep, and the removal set is everything else. Nothing ever
looks for garbage directly.

The mark phase walks the skoro `Graph` itself, meaning the same nodes, not a copy or a separate
structure. What differs is the set of edges it follows. The `Graph`'s data edges point from parent
to child, which is the direction values flow during evaluation. The mark phase instead follows
*references*, from a node to whatever it depends on:

- a node to its parents (the data edges walked backwards);
- a density to the quantity it scores;
- a node to whatever it binds.

"Reachable" in what follows means reachable along those references. It never means downstream in
the data flow.

The roots are the ids live handles name plus the graph's likelihood terms. The pieces, in the order
the collector uses them:

**Roots.** Two sources, unioned at the start of every sweep (`ModelServer::cleanup`, `model.rs`):

- The ids a live `Q` names. Each `Q` carries a `NodeDropGuard` that registers its id in the server's
  `HandleRegistry` (`crates/skoro-model/src/cleanup.rs`) and deregisters it on drop; clones share
  the one guard, so the id leaves the set when the last clone goes.
- Every id in `Graph::log_likelihoods` (`crates/skoro-graph/src/graph.rs`). The name undersells it:
  this is the whole term arena, not only the free-standing factors `mark_likelihood` adds. Each
  quantity's own density (`mark_log_density`), each proposal-only term (`mark_log_proposal`) and
  each free-standing factor lands there, and the field's comment says so.

**Edges followed.** Marking walks *upward* from a root through three kinds of reference:

1. Parent edges — whatever the node reads.
2. The scored quantity (`Graph::stochastic_of`) — a density marks the quantity it scores even if
   it never reads it, since the `Distribution` contract does not require it to.
3. Bindings (`Graph::bindings_of`) — a term marks its residual, Fisher root and every other node
   it binds, so those live exactly as long as the term does, with no handle of their own.

It never walks downward. Holding `x` does not keep a child `y = f(x)` alive; only a handle on `y`,
or a term that reads `y`, does.

**Sweep.** Everything unmarked is removed, children before parents, through the same leaf-only
`Graph::remove` a caller would use.

**Why this cannot change `p` or `q`.** Both are sums over terms in `log_likelihoods`, and every one
of those terms is a root. Each term therefore keeps everything it reads, the quantity it scores,
and everything it binds, so no summand of either density can lose an input. The guarantee comes from
what is rooted, not from which node kinds may be collected.

**How it relates to leaf-only removal.** The sweep is exactly this rule applied until nothing
changes: *remove a node that has no surviving children, that no surviving node binds, and that no
handle names.* Take `y = f(x)` and `z = g(y)`, with both handles dropped and no term reading either.
A single application of the rule takes `z`, which makes `y` a leaf, which needs a second
application. The collector `cleanup.rs` replaced worked roughly like that. It kept a queue of
candidates and popped each one once, so `y` stayed in the graph for good unless the mutation that
orphaned it put it back on the queue. Mark-and-sweep takes `z` and `y` in one pass, because neither
is marked. The same fact is why the sweep never trips `Graph::remove`'s refusals: the unmarked set is
closed downward (a marked child would have marked its parent), so it never contains a node that a
survivor still points into or binds.

The rule applies to every node kind, not only `Mapped`. A `Constant` or `Parameter` that no handle
names and no term reaches is collected as well. For a `Parameter` that changes neither `p` nor `q`,
since no term depends on it. It does take the parameter out of the model's free set, though, and
any bare `Id` or `TangentLayout` built earlier over it then gets `NodeNotFound`. A stochastic
quantity built through `add_quantity` is always kept, because its density is a term and the
scored-quantity edge marks it.

A stacked tangent's private anchors and per-block tangents are deliberately *not* roots. They are
ancestors of the `δ` node the caller holds, so they are marked while `δ` has a handle and swept
once it does not. Rooting them would leave `2k + 1` nodes behind on every tick of a sliding window.

Three properties follow, all documented on `ModelServer::cleanup`:

- Nothing has to *report* an orphaning. Reachability is a property of the graph, recomputed, not an
  event a mutation must hand back.
- Every node kind is collectable — a `Constant` or `Parameter` nothing reaches is garbage exactly as
  a `Mapped` node is.
- The sweep cannot fail.

It runs at the mutations that *reduce* reachability (`remove`, `remove_closure`,
`demote_to_parameter`) and deliberately not on the way in, because sweeping before every `add_*`
makes building an `n`-node model cost `O(n²)`. A workload that only grows holds its garbage until it
asks; `graph_node_count` is the bound such a caller watches. Dropping a `Q` never collects by
itself: it only sets the registry's dirty flag, and a `cleanup` that finds the flag clear returns
without marking anything.

An `Id` kept across a `cleanup()` may name nothing afterwards, which every id-taking method reports
as `NodeNotFound`.

### 1.3 What a distribution is

`Distribution<B, K>` (`crates/skoro-model/src/distribution.rs`) requires exactly four methods —
`quantity_type`, `sampler`, `parent_ids`, `factor` — and mentions no binding role. `factor` returns
a `Factor<B>`: a log-density node plus whatever `(Role, Handle)` pairs its builder chose to declare.
The model binds `ARGUMENT` to the scored quantity and each of `parent_ids()` itself, so a
distribution declares only the other roles, or none.

**`Factor.bindings` is a request, not a store.** The graph is the only record of bindings
(`Graph::bindings_of`, §1.2). A factor's `bindings` field lists bindings it wants recorded, and
`bind_factor` (`model.rs`) copies each one into the graph once, just after marking the density. The
`Factor` is then dropped, and nothing reads the field again: `factors()` returns `FactorInfo`, built
from the graph's bindings. The list is also incomplete, because it leaves out the `ARGUMENT`
bindings the model adds itself. There are two reasons it holds `Handle`s rather than recording
bindings directly. First, a bound node such as a residual or Fisher root is usually not an ancestor
of the log-density. Until the binding exists, the handle's drop guard is the only thing keeping the
collector from removing it. Second, `factor` only has `Q`s to work with, and a `Q` cannot `bind`,
since binding lives on `ModelBuilderOps`. So the distribution lists the bindings it wants, and the
model records them while it holds the write lock.

**`Normal<B, M>` is the only implementation in library code.** Every other impl in the tree is a
test double, and `JointNormal` is not one at all — it is a free-standing factor, because a
distribution is a conditional over *one* quantity type. Two consequences that shape everything
downstream:

- Anything non-Gaussian is a bespoke `Distribution` today.
- A bespoke distribution that offers no Fisher root is legal, and §2.3 is how a consumer finds out.

`add_quantity` is atomic: if `distr.factor(&x)` fails, the partially added quantity and everything
its factor built are removed before the error returns (`model.rs`, `add_quantity_with`).

### 1.4 Batch pinning

Two calls pin a model to a batch size, after which any other batch is an error rather than a
broadcast:

- `add_batched_constant` — one value per row.
- `stacked_tangent` — its frozen linearization anchors are batch-pinned constants, so a batch of `N`
  independent problems carries `N` independent linearization points.

Everything else broadcasts. This is worth knowing before a filter is written against a batch of
one and then run against many.

---

## 2. What introspection exposes

### 2.1 The house rule

There are no capability flags, no `supports_*` predicates, and no capability traits anywhere in the
workspace. This is a decision, not an omission — a `GaussianModelClientOps` bound once existed and
was removed. From `crates/skoro-normal/tests/facade.rs`:

> What the old `GaussianModelClientOps` bound expressed — "this client can add Gaussians" — was
> never a claim about what a model *contained*, and every client that can grow a model can add one.

And from the `client_ops` module header:

> Neither says anything about what a model *contains* — for that, read its terms and their
> bindings, through `ModelReaderOps::factors` and `ModelReaderOps::bound`.

So: a model's shape is read off its terms and bindings at runtime. There is no type-level
description of a model, and `ModelReaderOps` is deliberately not dyn-compatible — every consumer is
statically generic over it.

### 2.2 The reading surface

| Call | Returns | What it answers |
| --- | --- | --- |
| `factors()` | `Vec<FactorInfo>` | Every term, in term order, with `density`, `scored`, `in_density`, `in_proposal`, `arguments` |
| `terms()` | `Vec<Term>` | The same without the `ARGUMENT` join |
| `bound(id, role)` | `Vec<Id>` | What `id` bound under one role, in binding order |
| `bindings_of(id)` | `Vec<(Role, Id)>` | Everything `id` declared, without knowing the roles up front |
| `log_density_of(&q)` | `Option<Id>` | The term scoring a stochastic quantity |
| `graph_node_count()` | `usize` | The bound a retention policy watches |
| `tangent_layout(&ids)` | `TangentLayout` | Flat tangent coordinates, in the caller's order |
| `leaf_support(&ids)` | `Vec<Id>` | The free leaves `ids` depend on, walking parents but never through a leaf: a Jacobian row's structural sparsity |

The roles a term may carry are constants in `crates/skoro-model/src/roles.rs`: `ARGUMENT`,
`RESIDUAL`, `NORMALIZER_PARENT`, `LOCATION`, `WHITENING`, `FISHER_ROOT`. A `Role` is a newtype over
`&'static str`, and lives in `skoro-core` so that `SkoroError` can name one.

### 2.3 The idiom: arity is the capability test

Every reader matches on the arity of a `bound` call — `[] => skip`, `[x] => use`,
`_ => Err(ShapeMismatch)`. Two packagings of that idiom exist, and they are complements:

- `GaussianBlock::of` (`crates/skoro-normal/src/structure.rs`) turns three such calls into an
  `Option`: `Some` when a term binds exactly one each of `RESIDUAL`, `LOCATION`, `WHITENING`. It
  answers "is this term Gaussian-shaped?"
- `coverage::terms_missing(client, role)` and `coverage::require_role(client, role)`
  (`crates/skoro-model/src/coverage.rs`) answer the inverse: which scored terms a reader *cannot*
  see.

The second exists because the skipping is silent. `information_blocks` drops a term with no
`FISHER_ROOT`, `residual_layout` a term with no `RESIDUAL`, and `GaussianBlock::all` any term that
is not Gaussian-shaped — and in all three cases what comes back describes a sub-model with nothing in
the returned value saying so. Paired with a gradient, which sums over every term, the result is a
well-conditioned solve of the wrong system; the solve is `D×D` against `D` however many terms
contributed rows, so no shape check downstream can catch it. `require_role` is the precondition an
algorithm asserts before it assembles anything; `docs/fisher-scoring.md` §7 states the condition and
§8 the failure it prevents.

### 2.4 Erasure, and the runtime queries it leaves

The build API is typed (`Q<B, T>`, where `T` is a `PhantomData` tag); the arena is erased. Nodes hold
`Arc<dyn DynQuantityType<B>>`, and that trait is where the genuine runtime reflection lives
(`crates/skoro-graph/src/quantity.rs`):

- `kind()` — the continuous/discrete discriminator.
- `as_manifold() -> Option<&dyn DynManifold<B>>` — `None` for a discrete quantity.
- `inverse_retraction_op() -> Option<Box<dyn DynGraphOperation<B>>>` — the escape hatch that lets a
  factor be built over quantities whose manifolds are known only at run time, which is what
  `stacked_tangent` uses.
- `cardinality() -> Option<usize>` — `Some` only for finite discrete support.
- `as_any()` for downcasting, used by `information_blocks` to check that a bound Fisher root really
  is an `EuclideanMatrixD`.

### 2.5 One trap

`TangentLayout` is **not** revalidated against the graph after construction
(`crates/skoro-graph/src/tangent_layout.rs`). Reusing one across `add_*`, `remove`, or
`demote_to_parameter` is a caller error, and it is not detected. Layouts are also laid out by
*manifold dimension*, never storage size: `SO3` occupies 3 slots and `SE3` 6, not 9 and 12.

---

## 3. Marginalization

Expressing a marginal is graph work; computing one is host work. Both halves exist and are tested,
and the test that joins them (§3.3) is exact on a linear-Gaussian chain.

### 3.1 What is built and tested

Retiring a quantity from a model means three things: sever its distribution, express what it leaves
behind as a factor over its Markov blanket, and reclaim what is now unreachable. All three work.

| Piece | Where | What it does |
| --- | --- | --- |
| `demote_to_parameter` | `graph.rs`, `model.rs`, `client_ops.rs` (builder tier) | Strips a quantity's sampler, parents and own density, keeping its `Id`, type, and every child edge. **Returns the parents it severed**, since only a caller tracking handles can decide if they are now garbage. |
| `stacked_tangent` | `graph.rs` (builder tier) | `δ = [x₁ ⊖ p₁ ; … ; x_k ⊖ p_k]` as one `EuclideanD(Σdᵢ)`. Builds `2k+1` nodes; each `pᵢ` is a frozen batch-pinned constant in a private arena. |
| `JointNormal` / `add_joint_factor` | `crates/skoro-normal/src/joint.rs` | A Gaussian over several existing quantities, linearized at a frozen point, entering the model as a free-standing factor over them. |
| `whitening_from_precision` | `crates/skoro-dense/src/factor.rs` | Turns a precision `Λ` a caller already has into the lower `W` with `WᵀW = Λ`, packed for a `Tril`, plus its log-determinant, by one Cholesky of the reversed `Λ` — no inversion. For hand-built joints; a computed marginal (§3.3) arrives as a `W` already. |
| `remove_closure` | `client_ops.rs` (builder tier) | Retires the cone, taking referrers with it. |
| `cleanup` | `model.rs` | Reclaims whatever that made unreachable. |

`crates/skoro-normal/tests/window.rs` runs the whole loop: a linear-Gaussian chain with a
three-member window over twelve ticks, each tick demoting the blanket member, attaching a joint
factor over it, and retiring the oldest state-measurement pair with everything below it. The factor
count, the row count of `J`, and the graph's node count must all hold exactly steady afterwards —
"a leak or a dangling record is invisible in one tick and unmissable over twelve."

### 3.2 The product manifold was never needed

The deleted document called a **product manifold** the blocker here: a joint over a blanket holding
more than one quantity has no `Distribution` to describe it, and `Normal<B, M>` is a single
manifold. The code settled this, and the answer is no product manifold is required.

What a joint factor needs is a joint *tangent*, and stacking each member's own `x ⊖ p` gives one
without merging anything: the members keep their ids, their storage, and their other children. A
product manifold would have to re-home them into a single quantity, which is exactly what
`demote_to_parameter` exists to avoid. `crates/skoro/tests/stacked_tangent.rs` demonstrates it over
a heterogeneous blanket — it lives in the umbrella crate because it is the only one that can see
`SE3` and `Tril` alongside `ModelClient`.

The load-bearing property there is dimension, not storage. An `SE3` member contributes 6 columns and
never 12, so a blanket of `SE3` + `EuclideanD(3)` stacks to 9 rather than 15. Getting it wrong yields
a well-shaped residual that no downstream shape check would reject.

### 3.3 The computation half: partial elimination

What retiring `x` leaves on its blanket is the Schur complement of the curvature, restricted to the
factors adjacent to `x`. It is computed without forming `Λ` or inverting anything:

| Piece | Where | What it does |
| --- | --- | --- |
| `linearize` | `crates/skoro-linearize/src/lib.rs` | The model at a point as a block-sparse `GaussianFactorGraph<Id>` (§4.2), from the `RESIDUAL` rows or the `FISHER_ROOT` rows. |
| `marginalize_out` | `skoro-linearize`, over `skoro_dense::marginal::marginalize_out` | Eliminates `x` from **only the factors touching it**, and squares what they leave on the blanket into one lower-triangular `W` and right-hand side `d`. On the host the factors elsewhere are never read — folding them in would count them twice once the joint is attached. (The device side still differentiates every term: `linearize` is not local.) |
| `marginalize` | the same | The marginal of the *whole* graph on chosen keys, by eliminating every other key (COLAMD with the kept keys constrained last). For reporting and standard errors, not for a window. |
| `joint_normal_from` | `skoro-linearize` | The `JointNormal` for a `Marginal`: ids and block order taken from the marginal, `W` as a `Tril` constant, the normalizer `−½D ln 2π + ln|det W|`, anchored at the linearization point with the marginal's right-hand side as an offset, `r = W (x ⊖ at) − d`, so its value and Jacobian at `at` are exactly the eliminated rows on any manifold. (Moving the anchor to the mean `at ⊕ W⁻¹d` instead is exact only on a flat manifold: `SE3`'s retraction is not a group exponential, and `crates/skoro/tests/linearize_se3.rs` pins the difference.) |

The square root comes out lower-triangular because the last QR runs on the columns reversed and is
flipped back (`crates/skoro-dense/src/marginal.rs`): `R'ᵀR' = PΛP` for an upper `R'`, so
`W = P R' P` is lower with `WᵀW = Λ` — the shape a `Tril` can hold.

`crates/skoro-linearize/tests/window.rs` runs the loop for real: a linear-Gaussian chain,
linearized at zero on every tick so every marginal carries a right-hand side, a three-member window
over ten ticks, and at the end the windowed estimate of the live states must equal the full-batch
estimate to `1e-9`. The Laplace marginal is **exact** for a linear-Gaussian model and an
approximation otherwise; linearization-point drift in a nonlinear window is the caller's policy.
`crates/skoro-normal/tests/window.rs` keeps asserting the bookkeeping with a stand-in whitening.

### 3.4 Two obligations nothing can check

- **`JointNormal`'s block order.** `ids` fixes the order the tangents stack in, and `W` must be
  expressed in those same coordinates. Nothing verifies the correspondence, because nothing can — a
  `Tril` of the right width is a valid whitening for any ordering. `joint_normal_from` keeps it by
  construction, taking the ids from the `Marginal` that `W` came from; a hand-built `JointNormal`
  still carries the obligation.
- **A joint factor binds no `LOCATION` or `WHITENING`**, having no single mean node. So
  `GaussianBlock` cannot see it, and a filter reading `H = ∂μ/∂x` and `R = L Lᵀ` off the bindings
  will skip it silently — see §2.3.

An under-constrained blanket — a direction no retained factor determines — is the honest failure of a
marginal over a poorly observed window. Elimination reports it as `RankDeficient { position }` and
`whitening_from_precision` as `NotPositiveDefinite`; neither returns `NaN`s.

---

## 4. What an estimator still needs

### 4.1 The step loop already composes

Nothing is missing from the *primitives* of a damped Gauss-Newton or Fisher-scoring step. The loop
is expressible today, block-sparse end to end:

```rust
// The columns: Parameter/Stochastic leaves. A PointMap is what retracts, and `as_ref()` is its
// ValueMap view for evaluation.
let free = model.tangent_layout(&ids)?;
let mut at = point_map().with_quantity(&x, x0).build();   // skoro_model::hash_map_builders

// The rows, block by block: `J_t δ ≈ −r_t` per term, each over only the leaves it reaches.
// `linearize` asserts `require_role` itself (§2.3), and refuses a residual whose normalizer moves.
let graph = linearize(&model, at.as_ref(), &free, Curvature::Residual)?.remove(0).graph;

// Damp by row augmentation (§4.2), order, eliminate. The junction tree depends only on the
// structure, so a loop builds it once and reuses it while the structure holds.
let damped = graph.damped(lambda)?;                                // √λ I per key
let tree = JunctionTree::new(&damped, &Ordering::Colamd)?;         // skoro_dense::elimination
let step = eliminate(&tree, &damped)?.tree.back_substitute()?;     // skoro_dense::bayes

// Step on the manifolds.
at = at.retract_flat(&free, to_flat_tensor(&step, &free, &device)?)?;
```

`retract_flat` applies each block through its own manifold's retraction, which is the only
manifold-aware step in the iteration — everything above it is arithmetic in ℝⁿ.

`Curvature::Fisher` gives the Fisher-scoring rows instead — `C_t J_t` with no right-hand side — and
the gradient of `log p` beside them; the step is then `tree.solve_normal(&gradient)`. It is the
route for any distribution that binds a root, and the only correct one when a term's covariance is
itself free (a heteroscedastic Gaussian), which the residual route refuses with
`NormalizerNotConstant`.

The dense readers remain for small models and for tests: `jacobian_vjp` / `jacobian_jvp` over a
layout, `information_root` / `information` (the keyed blocks scattered back into `free`'s columns),
and `residual_vector`, the stacked `r` in `residual_layout`'s order, which is what a dense `lstsq`
pairs with `J`.

### 4.2 Crossing to the host

The Jacobian blocks and `δ` are the only tensor-typed values in the iteration; everything between
them is host `f64` and has no representation in the graph. `Dense::split_tensor` moves a whole batch across in
**one** transfer and then slices host-side — the obvious alternative, `narrow` per row, is `N` device
reads, and on a cube backend each one fences the queue.

The step is QR on the row-augmented system:

```
(JᵀJ + λI) δ = −Jᵀr    ⟺    min ‖[J; √λI] δ − [−r; 0]‖
```

The right-hand form is solved directly, so the iteration runs at condition `κ` rather than `κ²`, **at
every damping value** rather than only at `λ = 0`. Forming `JᵀJ` would square the conditioning;
avoiding it is the entire argument for the host boundary, since `burn` has no QR.

**The QR is block-sparse**, and it is GTSAM's elimination at the level of keys:

1. `jacobian_blocks` (`crates/skoro-model/src/jacobian.rs`) differentiates each row only over the
   free leaves it reaches (`leaf_support`), in **one** reverse pass: rows with disjoint supports are
   coloured alike and share frames, so a chain of any length needs a handful of frames rather than
   one per row, and one compile rather than one per term. `information_blocks` and
   `residual_blocks` are built on it.
2. `GaussianFactorGraph` (`crates/skoro-dense/src/factor_graph.rs`) holds the rows as factors
   `‖Σ_k A_k δ_k − b‖²` over keys. `Ordering::Colamd` runs faer's COLAMD on the **block** pattern —
   one entry per factor and key; `ConstrainedLast` stands in for the CCOLAMD faer lacks.
3. `JunctionTree` (`elimination.rs`) is the symbolic elimination: separators, the elimination tree,
   and chains merged into cliques. It holds no numbers, is checked against the graph's `Structure`
   on use, and is reused across iterations, batch rows, and ticks of unchanged topology — never
   keyed on the graph epoch, which a window bumps every tick.
4. `eliminate` (`bayes.rs`) runs **unpivoted** Householder QR on each dense front
   `[A_frontal | A_separator | b]`; column pivoting would mix the partition. The result is a
   `BayesTree` of conditionals `R x_f + S x_s = d`, plus the factors left on any kept keys. A front
   that does not determine a key is `RankDeficient { position }` — gauge freedom, or a key reached
   by no informative row — and a key no factor touches is `KeyUnconstrained`. The threshold is
   `1e-12` relative to the front's column norm, which assumes rows computed in `f64`: under an `f32`
   float policy an exact gauge freedom leaves diagonals near `1e-7`, and comes back as a huge step
   rather than an error.
5. From the tree: `back_substitute` (the least-squares step), `solve_normal(g)` (the Fisher step,
   `(RᵀR)⁻¹g`, whose solve is sensitive at `κ²` as any normal-equations solve is), `solve_upper(z)`
   (`R⁻¹z`: a draw from the Laplace approximation for `z ~ N(0, I)`), and `log_det_r`
   (`½ ln det AᵀA`, for a Laplace evidence).

Everything in 2–5 is indifferent to what the rows mean: a whitened residual Jacobian with `b = −r`,
a Fisher root with `b = 0`, and a damping prior are the same object. The pieces that assume the
least-squares reading are `residual_blocks`, `Curvature::Residual`, `back_substitute`, the
elimination's accumulated `error` (`min ½‖Aδ − b‖²`, the predicted objective an LM gain ratio
needs), and a marginal's right-hand side `d`. A term with no closed-form root has no row form; a
Hessian form `[G g; gᵀ f]` eliminated by Cholesky is where it would join (`GaussianFactor` is
`#[non_exhaustive]` for it), once per-term Hessians are assembled; `hessian_hvp` today assembles
the whole model's.

This is also the fence that keeps the two triangular-solve implementations apart, and deliberately:
`TrilInvert` is a *differentiable graph operation* on every backend and stays there, while
`skoro-dense` is host `f64` for the estimators. `crates/skoro-linalg/tests/common/tril.rs` is the
check that they have not drifted — the packed forward substitution and the host triangular solve must
compute the same inverse.

Two limits on the boundary:

- **A host `f64` solve is not `f64` end to end.** Under an `f32` float policy the Jacobian arrives
  already rounded and `δ` is rounded back. What it buys is `f64` *inside* the solve, which is where
  the conditioning lives. A filter that wants an `f64` state runs `Flex` under the `f64` policy,
  where the round trip is a `memcpy`.
- **A `Dense` is exactly one matrix.** `from_tensor` rejects a `[b, m, n]` with `b != 1` as
  `BatchNotSingleton` rather than silently taking slice 0. A batched solver reaches the
  factorizations through `split_tensor` / `split_vector` / `stack_vector`, so the batch axis is
  handled *around* the factorizations: `linearize` returns one graph per batch row, all of one
  structure, so one junction tree serves every row.

`lstsq` commits an algorithm to the host path deliberately: burn 0.21 has no QR, so unlike the other
factorizations it could not be reimplemented on `Tensor<B, 3>` behind the same name.

### 4.3 The column rule

> **Jacobian columns must be `Parameter` / `Stochastic` leaves.**

A column is something an estimator moves, and an estimator moves the free leaves. `jacobian_jvp` and
`jacobian_vjp` check their columns before either pass runs, and a `Mapped` or `LogLikelihood` column
is `SkoroError::MappedColumn` (`crates/skoro-model/src/jacobian.rs`). Before the check, such a column
came back as a **zero column**, because `vjp_forward` collects cotangents only at leaf steps. That
zero cannot be told apart from the legitimate zero of a leaf with no path to any row, which is an
ordinary occurrence. The check runs up front, not on the pass's output, so it also catches a mapped
column that reaches no row. `information*` inherits it through `jacobian_vjp`.

`Constant` and `Observed` columns are accepted and come back zero. `observe` turns a stochastic
column into an observed one at the same id, and a filter's layout must not start failing because it
observed its state.

Interior columns are **left out by design, not deferred**. Optimization never needs one: the passes
already apply the chain rule *through* mapped nodes to reach the leaves. For a filter chain this is
natural, since `x_k` is a leaf and `f(x_k)`, `h(x_k)` are mapped. A linearization at an interior
node is a different question, for example `∂h/∂x₁` where `x₁ = f(x₀)` in an all-mapped chain.
`compile_between` answers it: `CompiledStep::Given` is a leaf that collects cotangents and may stand
for a mapped node (see `Graph::compile_between` and §4.5). A correlated joint needs no
interior column either. It is modelled as a likelihood factor over `Parameter` leaves, and those are
ordinary columns.

Rows are unrestricted. Any continuous node works, and `jacobian.rs` fills in leaf rows itself
rather than leaving them to either pass, since `∂x/∂x` is the identity and neither pass would seed
it.

### 4.4 What is missing around the loop

- **No `Estimator` abstraction**, and no line search, trust region, damping schedule, or convergence
  test. `GaussianFactorGraph::damped` / `damped_diagonal` give the damped rows and `Elimination::error`
  the predicted objective, but choosing `λ` is the caller's job.
- **Second-order assembly is dense and whole-model.** `hessian_hvp(client, points, free)`
  (`crates/skoro-model/src/hessian.rs`) assembles the `D × D` Hessian of `log p` from `D`
  Hessian–vector passes, one per layout coordinate. Column `i` is `d/dt g(x ⊕ t eᵢ)`, `g` the
  gradient in the chart at each point: the ordinary Hessian on flat manifolds, and on curved ones
  equal to the pullback's Hessian only at a stationary point (it is not symmetrized). There is no
  per-term or block-sparse Hessian yet, which is what a Cholesky-eliminated Hessian factor would
  need. `information*` remains the *Fisher* information, the expected curvature.
- **No incremental elimination.** Every solve re-eliminates the whole tree; there is no iSAM2-style
  update that relinearizes and re-eliminates only the affected cliques, and no Bayes-tree shortcut
  for a marginal — each marginal query eliminates afresh (the symbolic part can be reused through
  `marginal_with`). Nor is there METIS or any iterative solver: COLAMD is the one ordering.

Two things that are *not* missing. The directional `_jvp` / `_vjp` / `_hvp` primitives are on
`ModelReaderOps`, so an estimator bounded on the read tier does get raw access to them. And compiled
programs are cached (§4.5): `information_blocks` compiles twice per call, but a solver loop that does
not mutate the model between iterations pays for those two compiles once — and its junction tree,
likewise, once.

### 4.5 The compiler underneath

Every evaluation above compiles the part of the graph it needs into a `CompiledSequence`, a flat,
index-based list of steps, and one sequence serves all four passes (`forward`, `jvp_forward`,
`vjp_forward`, `hvp_forward`). Three properties an estimator leans on:

- **Programs are cached, keyed on `(epoch, cuts, targets)`.** A target list alone is not a sound key:
  `observe`, `demote_to_parameter`, `remove` and constant folding change a node's kind or value at a
  fixed `Id`, which changes which inputs are differentiable. So every graph mutator bumps an epoch,
  including the `add_*` family, and a changed epoch drops the whole cache. Targets are keyed verbatim,
  because their order fixes every slot index; cuts are sorted and deduplicated, because only their
  membership matters. The cache clears itself past `MAX_CACHED_SEQUENCES` rather than growing, and a
  cloned `Graph` starts with an empty one. A workload that mutates every tick, such as the sliding
  window, clears the cache every tick and gains nothing from it. That is expected.
- **A compile can be cut.** `Graph::compile_between(cuts, targets)` stops at a frontier of interior
  nodes whose values the caller supplies. The trap it avoids: a cut must not lower to a constant step,
  or every consumer below it is marked non-differentiable and the values come back right while the
  derivatives come back silently wrong. A cut node lowers to `CompiledStep::Given` instead, a leaf
  whose value comes from `points` (`MissingPoint` if absent) and whose cotangent is collected, like a
  parameter's; it is reported among `ForwardOutput::variables`. `Graph::support(id)` is the forward
  counterpart, the nodes that depend on `id`, sorted. The Jacobian functions deliberately do not take
  cuts (§4.3).
- **Dead slots are released.** `CompiledSequence::last_use` records each slot's last reader, and a pass
  drops an operation slot there. Leaf slots are never dropped, because the VJP pass reads them after
  the tape has run. `log_weight` evaluates both of its sums in one program, through
  `Graph::log_density_terms_signed`.

The tests that pin each failure mode are `crates/skoro-graph/tests/compiler_cache.rs`,
`cut_compilation.rs`, `slot_liveness.rs` and `signed_terms.rs`.

---

## 5. Gap ledger

Ordered by what each gap blocks rather than by size. Recorded in `docs/roadmap.md`
unless noted.

| Gap | Blocks | Status |
| --- | --- | --- |
| No RNG seeding; no per-quantity sampling | Any MCMC or SMC — no reproducibility, no per-chain RNG | Not recorded anywhere; found by search |
| No reparameterization gradients (derivatives through samplers) | Reparameterized variational inference; `∇ log q` and `∇ log w` at fixed draws exist | `roadmap.md` |
| Fisher assembly is not tape-differentiable | Riemannian-manifold HMC specifically | `roadmap.md` |
| No `Estimator`, line search, or convergence test | Every deterministic estimator (§4.4) | `roadmap.md` |
| Joint proposals over several quantities | Block proposals; `Term` already has the shape | `roadmap.md` |
| Deferred bindings | Expensive-root families, very large models | `docs/deferred-bindings.md` — designed, postponed |

Closed on 2026-09-23, and so no longer rows: block-sparse `J` and a sparse solver (§4.2), the
Schur complement / marginal precision (§3.3), and residual assembly matching the Jacobian rows
(`residual_blocks`, `residual_vector`). Closed on 2026-09-24: dense Hessian assembly
(`hessian_hvp`, §4.4), and first-order gradients of `log q` and `log w` at given points
(`log_proposal_ord1`, `log_weight_ord1`), the score-function half of the old "derivatives of `q`"
row.

### 5.1 The sampling gap, in detail

It is the least documented and the most blocking, so it is worth stating precisely. Generation is
whole-model only: `generate(batch_size)` and `generate_from(points, batch_size)`. **There is no way
to sample a single quantity by id**, and no RNG control at all — `NormalSampler::sample` calls
`Tensor::random` with no seed parameter, and nothing in the workspace threads a seed or a generator.
An MCMC implementation could not make a run reproducible or give each chain its own stream.

What *does* exist is the proposal/target split: `add_proposal` puts a term in `q` and not in `p`, and
`log_weight` is their difference — "a difference, not a sub-sum", since a proposal-only term enters
it negatively. `crates/skoro-normal/tests/proposal.rs` composes importance sampling end to end from
those primitives, scoring by hand on the host. It is a demonstration that the pieces compose, not a
reusable sampler: no weight accumulator, no resampling, no ESS.

### 5.2 Searches behind the "does not exist" claims

```
rg -i 'seed_rng|set_seed|B::seed'                      # 0
rg -i 'hmc|nuts|leapfrog|metropolis|resampl|particle'  # 2, both prose
rg -i 'NlsStructure|ResidualCertificate|record_joint'  # 0
```

### 5.3 Open items tracked elsewhere

`docs/TODO.md` holds one, not duplicated here: deferred bindings.

---

## 6. Where the in-tree documentation is stale

None known. The four spots this section listed were repaired on 2026-09-23: the `skoro-dense` entry
point count, `demote_to_parameter`'s claim
that the graph reports what it orphans, `cleanup`'s references to the collector it replaced, and a
dropped clause on `stacked_tangent`. Record a new one here with its path.
