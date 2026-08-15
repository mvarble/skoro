# Architecture Assessment

Whether the workspace is serving its stated purpose: an estimator-agnostic probabilistic-model-building
API, over an efficient backend, on which many inference algorithms can be implemented — one of which
is sequential state estimation with smoothing.

A review document rather than a design one. It proposes no new API. Where it names a fix, the fix is
specified elsewhere; this document's job is to say what the structure is doing well, where it is not,
and why.

**Verdict up front.** The model-building layer is genuinely estimator-agnostic and well factored. The
*differentiation* layer is estimator-agnostic in interface and committed to dense whole-graph assembly
in implementation, and that commitment — not the linear algebra, which is correctly scoped and says so
— is what stands between the workspace and smoothing. The seam that has to move is Jacobian assembly.

## 1. What holds up

**Nothing in `skoro-graph` knows what an estimator is.** The arena stores nodes, edges, and density
bookkeeping; it exposes evaluation, JVP, VJP and HVP over caller-supplied targets. There is no filter
concept, no optimizer concept, no residual concept below `skoro-model`. That is the central claim of
the purpose statement and it is met.

**The evaluate/grow split is enforced by the type system, not by convention.** `NlsStructure`'s
supertrait is `ModelClientOps` (`skoro-model/src/structure.rs:159-163`), which supplies the whole
evaluation surface an optimizer needs and contains no way to add a node. An algorithm bounded on
`NlsStructure` can differentiate without holding anything that could invalidate the certificate it was
handed. This is the right shape and it generalizes past least squares.

**`ResidualCertificate` is a well-made unforgeable-evidence pattern.** Entries can only be added
through `record`, which demands a `WhitenedResidual` bound (`structure.rs:125-145`); a hand-built
certificate is empty and there is no way to insert a fabricated id. The module comment names the
condition under which every claim lapses. Load-bearing invariants stated with their own falsification
condition are rare and worth keeping.

**`TangentLayout` is the right abstraction and is not what needs to change.** It is an ordered
`Vec<TangentBlock { id, offset, dimension }>` with a reverse index — a variable ordering with block
structure, which is what GTSAM calls `Ordering` and Ceres calls parameter-block ordering. A
block-sparse assembler needs exactly this object. Likewise, flattening *vectors* is correct
independently of sparsity: `r` is dense, `δ` is dense, and `PointMap::retract_flat` stays as it is
even in a fully sparse pipeline. When §3 below says the flat design is the constraint, it means the
flat **Jacobian**, not the layout and not the vectors.

**The manifold discipline is sound.** Dimension-not-storage, points that never flatten, steps as
retractions rather than vector addition — these are the things a hand-rolled estimator gets wrong, and
they are structural here rather than documented.

## 2. Where it falls short: Jacobian assembly

`jacobian_vjp` (`skoro-model/src/jacobian.rs:136-149`) produces `J` by seeding `D_out` identity frames
and running **one reverse pass over the whole graph**. The cost is `D_out × |graph|` and is entirely
insensitive to how sparse `J` is. A factor graph's `J` is mostly zeros; none of that structure is
available to the estimator, because it is gone before the estimator sees anything.

Two concrete costs, neither currently documented:

- `TangentLayout::identity_frames` (`skoro-graph/src/tangent_layout.rs:264-279`) materializes
  `Tensor::eye(N)` and hands out a `[batch, d_j, N]` slice per block. The seeds alone are
  `batch × D_out²` elements, of which `batch × D_out` are nonzero.
- The reverse pass propagates all `D_out` frames through **every** factor's subgraph, when only `d_f`
  of them are nonzero on any given factor. The redundancy factor is roughly the number of factors.

**This is not fixable in the solver.** Handing an already-dense `J` to a sparse factorization means
recovering the pattern by scanning it — and what that yields is a *numerical* pattern, not a
structural one, so it can shift between iterates. Sparse direct methods earn their keep by computing a
symbolic factorization once and reusing it across iterations against a fixed pattern; recover the
pattern numerically and you re-analyze every iteration having already paid full dense assembly. Slower
at every size. `skoro-dense` is correctly scoped and honest about its scope; it is not where the
problem is.

**`flattened-tangent-spaces.md` §5's cost model has no column for the option that wins.** It frames
the decision as `D_in` frames (JVP) against `D_out` frames (VJP). Both are points on the same line —
one whole-graph pass propagating `D` seed frames. Block assembly, where each factor is differentiated
against only its own variables, is neither, and is absent from the document.

## 3. The circularity

This is the finding that matters most, because it explains why the shortfall in §2 has persisted
without anyone being wrong.

Three deferrals are recorded in three documents as if independent:

1. `estimator-linear-algebra.md` §8 defers the device-side batched dense solve until a batched
   estimator needs one.
2. `graph-marginalization.md:327-331` defers adjacency — no `FactorGraph` — and names its own
   revocation condition: "a batch smooth over hundreds of poses would revoke the argument."
3. `flattened-tangent-spaces.md` §5 never considers block assembly at all.

All three are gated on the same hypothetical consumer: the batch smoother at
`sequential-estimation.md` phases 3–5. And that smoother cannot be built efficiently until they land.

> **One deferral counted three times.** The architecture's current answer to "does it serve smoothing"
> is: *yes, once three separate things land, each of which is waiting for smoothing to exist.*

Each deferral is individually well-argued. The composition is what fails. Nothing in the roadmap
breaks the cycle, because every document defers to a consumer whose feasibility depends on the other
two.

**The way out is to pick one and build it before its consumer exists.** Block assembly is the right
choice: it is the only one of the three whose *interface* the other two depend on. A device-side
batched solve is a swap behind names `skoro-dense` already fixes. Adjacency is only needed in the
narrow, capturable form block assembly requires — which is the next point.

## 4. Adjacency is cheaper than the objection assumes

`graph-marginalization.md:333-347` rejects a `FactorGraph` on strong grounds: it would be a *derived
projection* requiring invalidation across the entire mutation API — including every node constructor,
since `add_mapped` can change an existing factor's variable set — and it would be a fourth index over
an arena whose existing three already fall out of sync.

That objection is correct about a *derived global index*. It does not apply to a **captured** one.
`ResidualCertificate::record` (`structure.rs:129-145`) is called by the builder at the moment the
factor is built, when its roots are already known. A `FactorRecord` extended with its root set and a
compiled subsequence is captured evidence, not a maintained projection — the same pattern the
certificate itself already uses, and the reason it is unforgeable.

The invalidation surface is real but bounded, and smaller than the §8 objection implies. It is
enumerated in the remediation plan; the load-bearing case is that `observe`/`unobserve` preserve both
the node id and its `parent_ids`, so a captured root set stays valid, while the node flips between
being a cotangent collection point and not — meaning a stale compiled sequence yields **right values
and wrong derivatives**.

## 5. Two structural weaknesses the review surfaced

**Derived state has no invariant discipline.** `Graph` maintains three indices whose consistency is
asserted in a doc comment (`graph.rs:34-39`) and checked nowhere. During this review a fourth
divergence was found: `unregister_density` (`graph.rs:672-685`) clears all three maps but never clears
`LogDensityNode::stochastic_id`, while `remove` reads that field to clean the maps — so an orphaned
density can delete a live density's reverse-map entry. The problem is not the specific bug; it is that
five methods write these indices independently and nothing validates the result. Any caching layer
built on top inherits this.

**Compiled structure is coupled to the value cache.** `is_differentiable_parent` (`graph.rs:75-89`)
returns `value.is_none()` for `Mapped` and `LogDensity`, and that feeds `constant_inputs` /
`forward_only` into every compiled step. So *which steps are on the tape* depends on which subtrees
happen to be constant-folded — value state, not topology. This is why compiled sequences cannot
currently be cached, and it is a layering inversion: an optimization (folding) determines a structural
property (tape membership).

**Leaky `Graph`/`Model` ownership.** `is_mapped_leaf: AtomicBool` lives on a guard in
`skoro-model/src/model.rs:417`, and `mark_density` clearing it (`model.rs:138`) is what makes orphaned
densities unreclaimable by `cleanup()`. A `Graph`-level fact — whether a node is still a reachable
leaf — is being tracked one layer up, with an atomic, for lifetime reasons. The split between what
`Graph` owns and what `Model` owns needs restating.

## 6. Summary

| Layer | Estimator-agnostic? | Serves smoothing? |
|---|---|---|
| `Graph`, `Q`, `Distribution`, manifolds | Yes, structurally | Yes |
| `TangentLayout`, vector flattening | Yes | Yes — survives a sparse rewrite unchanged |
| Jacobian assembly (`jacobian_jvp`/`vjp`) | In interface only | **No** — dense whole-graph by construction |
| `skoro-dense` | Yes, and correctly scoped | No, and says so |

The library is serving filtering well. It is not yet serving smoothing, and the reason is narrower
than "it needs sparse linear algebra": it needs block-structured Jacobian assembly, which needs
captured per-factor adjacency and cached compiled subsequences, which need the graph's derived state
to stop leaking. That is a chain of four concrete changes, not an architectural rewrite — and the
model-building layer that the purpose statement is really about comes through the review intact.
