# The graph compiler: caching, cut compilation, and step improvements

Three changes to `skoro-graph`'s compiler: a compiled-program cache, cut compilation, and two
small step improvements. **All three have shipped.**

This is the first document in this directory that was committed *before* its work rather than
after, so it is written forward — as a design and an execution order — and was amended section
by section as each one landed. It is left in that shape rather than rewritten into a
description, because the reasoning about what *would* go wrong is the part worth keeping: two
of its three sections exist to prevent a failure that returns correct values and silently wrong
derivatives, and both are now pinned by mutation-tested cases named in the verification below.

## What the compiler did before

`Graph::compile_backward_from(&[Id])` was the only compile entry point in the workspace. It did
backward reachability from a target set (`topological_sort`), lowered every reachable node to
exactly one `CompiledStep`, and returned a `CompiledSequence` that all four passes — `forward`,
`jvp_forward`, `vjp_forward`, `hvp_forward` — ran over. One sequence serves all four; the pass
is chosen by which method the caller invokes, not by anything in the artifact. That part is
unchanged, and deliberately so.

Line numbers below are as they stood when each section was written, and are not maintained;
names are. Three things followed, and this document addresses each:

1. **Nothing is cached.** All ten internal call sites build a fresh sequence and drop it.
   `information_blocks` compiles twice per call, and a solver loop pays that every iteration.
2. **The only expressible program is "these targets, from all their ancestors."** There is no
   way to stop at a frontier of interior nodes whose values you already have. This is the cut
   compilation item in [roadmap.md](roadmap.md), and it is why [skoro.md](skoro.md) §4.3 has
   to warn that Jacobian columns must be `Parameter`/`Stochastic` leaves.
3. **Three small defects sit behind a long freeze.** The compiler and the autodiff tape had
   not been touched in 69 commits as of `a155631`: a reachable `unwrap()` panic, no slot
   liveness, and the two-pass `log_weight`.

**Which workloads the cache actually helps.** The beneficiaries are loops that evaluate
repeatedly without mutating: the Fisher-scoring / Levenberg–Marquardt loop, and
`information_blocks`. A mutate-then-evaluate workload such as the sliding window
(`crates/skoro-normal/tests/window.rs`) adds and removes nodes every tick, so under the
invalidation rule below it clears the cache every tick and gains nothing. That is expected,
not a defect, and it is recorded here so it is not discovered as a surprise.

**Out of scope, deliberately:** fusion, CSE, reordering for locality, making the pass a
compile-time variant (one sequence serving four passes is load-bearing — do not break it),
sequence slicing and splicing (dense positional indexing makes reindexing cost more than
recompiling into the cache), per-seed dead-step elimination (seeds are runtime data; it would
multiply cache entries), block-sparse `J`, and sparse solvers. The last two stay in
[roadmap.md](roadmap.md).

---

## 1. A graph epoch and a compiled-sequence cache

### Why the key is not the target set

A target-set key alone is **unsound**. `observe`/`unobserve`/`demote_to_parameter`/`remove`/
`check_cache` change `value: Option<_>` or node kind *at a fixed `Id`*, which flips
`is_differentiable_parent` (`graph.rs:100-114`) → `input_diffs` → `constant_inputs` /
`forward_only` at `graph.rs:2302` and `:2334`. The same target list must be able to compile to
a different sequence. The key is `(epoch, targets)`.

### 1.1 `Graph::epoch`

A private `u64`, starting at 0. A private `fn touch(&mut self)` bumps it and then calls the
existing `debug_validate_indices()` (`graph.rs:509`); all 12 existing
`debug_validate_indices()` call sites become `touch()`. That makes "bump the epoch" the same
greppable obligation the graph already has for index validation.

**Every mutator bumps, including the `add_*` family.** Bumping only on mutation of *existing*
nodes would be sound — `IdGenerator` issues monotonically and never reuses
(`crates/skoro-core/src/id.rs:24-30`), so a freshly issued `Id` cannot be an ancestor of any
pre-existing node — but that invariant would have to be defended forever, and none of the
workloads driving this work add nodes between evaluations. "Do not bump on pure addition" is
the named escape hatch if a builder-interleaved workload ever appears. It is not implemented.

Do **not** derive the obligation list from `grep "fn [a-z_]*(&mut self"`. It cannot match a
generic mutator, because `<K, T, D>` sits between the name and the paren, so it silently drops
`add_constant` (`graph.rs:517`), `observe` (`:1003`), `observe_batched` (`:1032`) and the rest
of the `add_*` family. Use `grep -n "&mut self" crates/skoro-graph/src/graph.rs`, which finds
31 sites.

Two need attention because they are *not* in the `debug_validate_indices` set: `check_cache`
(`graph.rs:2129`, reached from `add_mapped_impl` at `:778`) writes `MappedNode::value`, and
`clear_cached_descendants` (`:1333`) clears it — both flip differentiability. `bind`
(`:151`) is the one mutator that provably need not bump, since the compiler never reads
bindings, but it bumps anyway to keep the rule free of exceptions.

### 1.2 The cache

A new private field on `Graph`:

```rust
compiled: Mutex<(u64, HashMap<Vec<Id>, Arc<CompiledSequence<B>>>)>,
```

Interior mutability is required: the evaluation methods take `&self`, and `ModelClient`
reaches the graph through `Arc<RwLock<ModelServer<B>>>`
(`crates/skoro-model/src/model.rs:640-642`). `CompiledSequence` is safe to share —
`SequenceMutator` is `'static + Send + Sync` (`crates/skoro-core/src/sequence.rs:1`),
`DynQuantityType` is `Send + Sync` (`crates/skoro-graph/src/quantity.rs:176`), and the four
passes take `&self` and allocate every mutable slot vector locally.

```rust
pub fn compile_backward_from(&self, targets: &[Id]) -> SkoroResult<Arc<CompiledSequence<B>>>
```

Lock; if the stored epoch differs from `self.epoch`, clear the whole map and store the new
epoch — every entry is stale by construction, so there is no eviction question to answer.
Then look up `targets.to_vec()`; on a miss, compile, `Arc`, insert, return.

Epoch invalidation handles staleness but **not** unbounded growth: target lists are
caller-supplied, so within one epoch the caller controls key cardinality. A single constant
`MAX_CACHED_SEQUENCES` (64) clears the whole map on overflow. Clear-on-overflow rather than
LRU: one branch, no recency bookkeeping, and the pathological case it guards against is rare
enough that thrashing does not matter.

**The key is the target slice verbatim, not sorted or deduped.** `topological_sort` seeds its
stack with the targets and pops LIFO, so target order fixes the step order and therefore every
slot index. Canonicalizing would change the compiled artifact. A solver loop passes the same
slice each iteration, so verbatim keys hit.

### 1.3 `Clone`

`Graph` is `#[derive(Clone)]` at `graph.rs:57`, and `Mutex<HashMap<_, Arc<_>>>` is not
`Clone`. `Clone` becomes hand-written: every existing field cloned, and the clone gets an
**empty** cache at the source's epoch. That is also the correct semantics.

### 1.4 The `NodeNotFound` fix ships here

`topological_sort` tolerates unknown ids — `self.nodes.get(&current)` is an `if let` at
`graph.rs:2222` — and emits them verbatim, which
`crates/skoro-graph/tests/graph_mutation.rs:1422` pins. But `compile_backward_from` then does
`self.nodes.get(&id).unwrap()` at `graph.rs:2250`, so a stale target id panics rather than
returning `SkoroError::NodeNotFound`. Only `log_density_terms` (`graph.rs:1560-1562`)
pre-checks membership today. The fix belongs here rather than in §3 because it is the same
signature change over the same call sites: the ten in `graph.rs` (1568, 1607, 1658, 1728,
1740, 1763, 1787, 1813, 1837, 1875) gain a `?`, and three in
`crates/skoro-graph/tests/erasure.rs` (368, 397, 421) gain an `.unwrap()`.
`Arc<CompiledSequence>` derefs transparently, so the pass calls are unchanged.

---

## 2. Cut compilation and `support`

### The trap, first

**A cut boundary must not lower to `CompiledStep::Constant`.** If it does,
`is_differentiable_parent` returns `false` for it, every downstream operation gets
`constant_inputs: true`, those steps never reach the tape, and **the values come back right
while the derivatives come back silently wrong.** That is verbatim the failure
`clear_cached_descendants`'s doc comment (`graph.rs:1324-1332`) exists to prevent, and the
same shape as [skoro.md](skoro.md) §4.3's zero column. A cut boundary needs a *differentiable
leaf* step kind. Getting that kind right is the whole of cut compilation; the traversal change
is trivial beside it.

### 2.1 `CompiledStep::Given`

`Given { id, quantity_type }`. Its value is supplied by the caller in
`ForwardInput::points`, and its **cotangent is collected** — so it behaves like `Parameter`
everywhere derivatives are concerned, and unlike `Constant` everywhere. Unlike `Parameter`, it
may stand for a `Mapped` or `LogLikelihood` node.

### 2.2 `Graph::compile_between(&self, cuts: &[Id], targets: &[Id])`

Same return type and same cache as §1; the key becomes a small `CacheKey` struct rather than
an overloaded `Vec<Id>`. **Sort and dedup `cuts` in the key; leave `targets` verbatim.** Cuts
only terminate the traversal and never seed it, so only their membership is significant;
targets seed it, so their order fixes every slot index.

`topological_sort` grows a cut-aware sibling: on reaching a node in `cuts`, do not push its
parents, and emit it as a leaf.

### 2.3 A cut node's differentiability comes from the cut, not from the node

In the lowering loop, a parent that is a cut node contributes `input_diffs = true` whenever
its quantity type is `Continuous` — regardless of whether the underlying `MappedNode` carries
a folded `value`. This routes through a wrapper around `is_differentiable_parent` rather than
an edit to it, so the uncut path is provably unchanged.

### 2.4 Execution

`non_op_forward` (`compiler.rs:510`) gains a `Given` arm: read from `points`,
`SkoroError::MissingPoint` if absent, `ensure_batch`, insert into `out.variables`, and seed
from the tangent and vector channels exactly as the `Parameter` arm does. `Given` joins the
two leaf-collection loops — `compiler.rs:364-394` (VJP, including the zero-cotangent fill for
a continuous leaf that reaches no target) and `compiler.rs:483-495` (HVP).

`out.variables` currently means exactly "`Parameter` or `Sampler` leaf" (`ForwardOutput`'s own
doc, `compiler.rs:106-115`), so putting `Given` there is a semantic decision rather than a
detail. Its two non-test readers were audited:

- `crates/skoro-model/src/jacobian.rs:94` (`fill_leaf_rows`) writes an identity block for a
  row that is a variable and also a column. For a cut node that is **correct** — it is a free
  input to the compiled program, so `∂x/∂x = I`.
- `crates/skoro-model/src/information.rs:117` reads `variables` only as the last of three
  lookup fallbacks. Unaffected.

So `Given` belongs in `variables`, not in `evaluations` and not in a third map. The
`fill_leaf_rows` behaviour needs its own test: the cut-versus-uncut equivalence test below
compares derivatives at the boundary and will not catch a wrong output map.

### 2.5 `Graph::support`

`support(&self, id: Id) -> Vec<Id>` — a forward walk over the existing
`children: HashMap<Id, HashSet<Id>>` through `children_of`. `children` is a `HashSet`, so
unlike `topological_sort` — which gets determinism free from `parent_ids().rev()` on a `Vec` —
a forward walk has none. **Sort before emitting**, and say so in the doc comment.

### 2.6 What this does to §4.3

`Given` makes an interior mapped node expressible as a Jacobian column. The conservative branch
is taken: `jacobian_vjp` and `jacobian_jvp` keep taking leaf columns, and [skoro.md](skoro.md)
§4.3 is reworded from "impossible" to "not what the default API does — `compile_between` is how
you ask for it." Widening `jacobian.rs` is separate, later work, and it is now a row in the gap
ledger rather than a standing restriction.

One consequence worth stating plainly: because `jacobian.rs` still compiles uncut, **nothing in
the tree can hand `fill_leaf_rows` a `Given` yet**. The argument in §2.4 that `variables` is the
right map for one — that an identity block is correct for a free input — is reasoned, not
exercised. The test that covers it today asserts at the compiler level, that a cut value is
reported among the variables and not among the constants. Widening `jacobian.rs` is what would
put the rest of that argument under test.

---

## 3. Slot liveness and fusing `log_weight`

### 3.1 Slot liveness

Each slot's last consuming step is computed at compile time and stored on `CompiledSequence`
(`last_use: Vec<usize>`, a pure function of `steps`, exposed as `CompiledSequence::last_use`).
Each pass releases a slot at the step that was the last to read it. Two rules, easy to
conflate:

- **Gather uses from every step kind.** `SequenceMutator::input_indices()` is on
  `DynSequenceSampler` as well as `DynSequenceOperation`, and a `Sampler` step reads
  `point_seq` at `compiler.rs:566` — its inputs can be operation slots.
- **Drop only `Operation` slots**, never `Constant`, `Parameter`, `Sampler`, or `Given`.
  `vjp_forward`'s zero-cotangent fill reads `point_seq[index]` for dims and device at
  `compiler.rs:379-388`, *after* `tape.backward`, so a dropped leaf slot would panic there.
  Operation results are already cloned into `out.evaluations` before the drop, and the tape's
  `VjpStep`/`HvpStep` carry their own memoized states
  (`crates/skoro-operation/src/diff/step.rs:9, 39`), so nothing else reads them.

### 3.2 Fusing `log_weight`

`ModelClientOps::log_weight` (`crates/skoro-model/src/client_ops.rs:650-659`) calls
`graph.log_density_terms` twice and subtracts; each call reaches `sum_terms`
(`graph.rs:1566`) and compiles. A signed sibling —
`Graph::log_density_terms_signed(points, plus, minus)` — compiles once over the union, runs
one `forward`, and sums with signs. `log_weight` calls it; `log_density` and `log_proposal`
keep using `log_density_terms`. The result changes only to rounding, and the existing identity
test (`crates/skoro-model/tests/terms.rs:156`) should still hold — if it becomes
tolerance-sensitive, that is the fusion working, not a regression.

---

## Verification

Run per section; each must be green before the next starts.

```
cargo test --workspace
cargo clippy --workspace --all-targets
cargo fmt --check
```

**Known non-green baseline:** four `skoro-so3` `wgpu_f32` tolerance misses
([TODO.md](TODO.md) §2). Record the pre-change failure list and compare against it, rather
than asserting "all green."

### §1 — new tests in `crates/skoro-graph/tests/`

Behavioral and precision-agnostic, so a plain topic-named integration test file, not a
per-backend binary: `crates/skoro-graph/tests/compiler_cache.rs`.

They need **no observer at all**, which is better than the compile counter this document first
called for. `compile_backward_from` returns an `Arc`, so `Arc::ptr_eq` asks the question a cache
actually exists to answer — "did the caller get the *same* artifact back" — directly, and no
test-only state goes into `Graph`. Eviction is asked the same way: request a key, push past the
cap, request it again, and assert it is a *different* `Arc`. `MAX_CACHED_SEQUENCES` is public
for that test.

- **Hit:** two compiles of the same targets return the same `Arc`.
- **Miss on target change:** different target lists get different `Arc`s, and the first stays
  live.
- **Target order is part of the key:** `[prod, sq]` and `[sq, prod]` are different programs,
  asserted against the emitted step order so the test cannot go vacuous.
- **Invalidation, table-driven over all 31 `&mut self` sites:** compile, mutate, compile,
  assert the cache cleared. `observe`/`unobserve`/`demote_to_parameter`/`remove` are checked
  for *correctness* too, not just for the count — assert the recompiled sequence's
  `constant_inputs` actually flipped. The `add_*` family is included: under the
  bump-on-everything rule they must clear as well, and the test is the only thing that keeps
  that rule from eroding.
- **Overflow:** more than `MAX_CACHED_SEQUENCES` distinct target lists in one epoch clears the
  map rather than growing it.
- **`Clone` starts empty.**
- **`NodeNotFound`:** compiling a removed target returns the error instead of panicking.
- **Reuse:** the cache is the first thing in the tree that hands one artifact to two callers, so
  two properties that were previously implementation details become contract. A reused sequence
  must evaluate each set of points on its own merits — any surviving slot state would show as
  the previous call's answer — and two threads must be able to run one sequence at once.

Every invalidation test was mutation-tested: with the bump removed from `touch()`, all eight
fail; with the cap removed, the eviction test fails. None of them is vacuous.

### §2 — the load-bearing tests

`crates/skoro-graph/tests/cut_compilation.rs`. Two tests carry the feature, and both were
mutation-tested:

- **Cutting above a fold clears `constant_inputs` on the consumer**, and a VJP through that cut
  reaches the boundary. Reverting the cut-aware differentiability fails exactly these two and
  nothing else — which is the silent failure mode demonstrated rather than described.
- **A cut pass composes with the pass below it.** Differentiating in one reverse pass, and
  differentiating to the cut then pushing that cotangent through the lower half, must agree.
  Dropping `Given` from the cotangent collection fails this and the VJP test, and nothing else.
  Finite differences alone would not catch it: a cut that drops a term is *self-consistently*
  wrong.

The numeric harnesses in `crates/skoro-graph/src/test_utils.rs` cannot be pointed at a cut
program — they drive `Graph::evaluate_*`, which compile uncut, and adding cut-aware evaluation
entry points is scope this section does not need. They reach the feature through an equality
instead: **a cut at a node gives the same first- and second-order derivatives as replacing that
node with a free variable**, and the free-variable graph is then checked against forward
differences by `test_vjp_numerically` and `test_hvp_numerically` at the workspace pairing
`DELTA = 1e-6`, `EPSILON = 1e-5`. Comparing both orders matters: a cut that reached the tape for
the first derivative but dropped the second would pass every other test in the file.

Also covered: a cut at a `Parameter` is a no-op; a cut off every path is ignored; a cut at a
stochastic node replaces its sampler, because a `Sampler` step holds parent slot indices the cut
has just pruned; a cut with no supplied value is `MissingPoint`; a cut naming no node is
`NodeNotFound`; cut order and repetition are not part of the cache key while target order is;
`compile_backward_from` is the empty-cut case and shares its cache entry; and `support` is the
forward closure, in a deterministic order.

### §3

`crates/skoro-graph/tests/slot_liveness.rs` and `crates/skoro-graph/tests/signed_terms.rs`.

- **Liveness.** `last_use` is a public accessor rather than a test-only observer — it is a real
  fact about a compiled program, and asserting on it needs no counter. Four cases: a chain
  releases each link at its only consumer; a shared slot lives until its *last* reader; a slot
  nothing reads dies where it is born; a sampler's inputs count as reads. Three mutations, three
  distinct catches: releasing one step early fails eight tests across the workspace, releasing
  leaf slots too fails the zero-cotangent test alone, and dropping the sampler branch from the
  read-gathering fails the sampler test alone.
- **`log_weight`.** `crates/skoro-model/tests/terms.rs` and
  `crates/skoro-normal/tests/proposal.rs` pass unchanged. "Compiles once" is counted, not
  inferred: the cache holds one program per distinct target list, so `cached_program_count`
  (behind `test-utils`, reading the cache's real state) distinguishes one compile from two.
  Reverting to a program per side fails it with 2 against 1.
- **Long-horizon:** `crates/skoro-normal/tests/window.rs` runs twelve ticks of
  mutate-then-evaluate and already asserts that factor count, `J` rows, and node count hold
  steady. It was left as it is: the window adds and removes every tick, so under the
  bump-on-everything rule the cache is cleared each tick and any growth assertion there is
  vacuous. §1's overflow test is what covers growth.

---

## Documentation this invalidated

Repointing was part of the work, not tidying after it, and two of these are the authoring
contract. All of them are done.

| File | What changed |
|---|---|
| `.claude/skills/skoro-diff-operation/SKILL.md` | Cited `compiler.rs:467` for "only surfaces `Parameter`/`Sampler` slots" — already stale, and §2 changed what they surface. Repointed by name. Its `is_differentiable_parent` citation went the same way, and now mentions the cut override from §2.3. |
| `.claude/skills/skoro-diff-operation/references/testing.md` | The same stale citation, repointed by name. |
| [skoro.md](skoro.md) §5 | The compiled-cache, cut-compilation and two-pass-`log_weight` rows retired. What replaced the first: block-sparse `J` and sparse solvers, still open. What is new: `jacobian_*` cannot take an interior column, which §2.6 made expressible without changing the API. |
| [skoro.md](skoro.md) §4.3 | Reworded from a compiler restriction to an API one, per §2.6. |
| [roadmap.md](roadmap.md) | Cut compilation and `log_weight` fusion removed; block-sparse `J` and sparse solvers kept, now joined by widening `jacobian_*`. |
| [deferred-bindings.md](deferred-bindings.md) §3.5.2 | Asserted "No compiler change (R10 still holds)". Re-checked and still true — the compiler reads no bindings — and the note now says so against a compiler that has since changed, rather than leaving a finished design silently uncorroborated. |
| `readme.md` | Its sub-DAG sentence gained a clause naming cuts. |
| [TODO.md](TODO.md) | Gained `retire_density_of`'s atomicity, found while auditing mutators for §1 and not fixed. |
