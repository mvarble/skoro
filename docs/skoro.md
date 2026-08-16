# Skoro

Skoro is a probabilistic modelling library whose purpose is an **estimator-agnostic
model-building API over an efficient backend, on which many inference algorithms can be
implemented**. This document is the design record: what the layers are, what a real model looks
like, how the pieces move at run time, and what is still missing.

It replaces six earlier documents — `flattened-tangent-spaces.md`,
`estimator-linear-algebra.md`, `graph-marginalization.md`, `sequential-estimation.md`,
`model-structure-bounds.md` and `architecture-assessment.md` — which are recoverable from git. They
were individually careful and collectively hard to hold in view: each deferred to the others, one's
§6 was superseded by another, and the reading order was stated nowhere. Everything from them that
is still true is here.

**Read it front to back.** §1 is the layering, §2 is a worked model to argue over, §3 follows one
estimator call through every layer, §4 is what does not exist yet and the order to build it in.

---

## 1. Abstraction boundaries

Six layers. The useful way to state each is by **what it knows nothing about** — that ignorance is
what keeps the stack estimator-agnostic.

### 1.1 Tensors — `skoro-backend`

A thin layer over `burn`, used for tensor execution only. **`burn`'s autodiff engine is
deliberately unused**; Skoro has its own, because inference over manifolds needs derivatives that
are aware of retraction rather than of raw storage.

Three shape conventions carry the whole system:

| Alias | Shape | Meaning |
|---|---|---|
| `Point<B>` | `[batch, storage]` | A point on a manifold, in its storage encoding |
| `TangentVector<B>` | `[batch, dimension]` | One tangent vector |
| `TangentFrame<B>` | `[batch, dimension, frame]` | *Many* tangent vectors at once |

The **frame axis** is the one that matters. It lets a single forward or reverse pass carry many
derivative directions simultaneously, which is what makes assembling a Jacobian one pass rather
than `D` passes.

Knows nothing about: manifolds, graphs, probability.

### 1.2 Differentiable operations — `skoro-operation`

`DiffOperation` is the stateless mathematics of a single operation: `forward`, `vjp`, `hvp`. The
forward pass always computes eagerly, but *what it memoizes* is selected by `DiffModeState`:

- `ForwardOnly` — no derivative tracking.
- `Jvp` — forward map plus the pushforward on an input tangent frame.
- `Vjp` — forward map plus the state needed for the first-order cotangent pullback.
- `Hvp` — forward map plus state for both the pullback and the second-order sensitivity.

Both backward passes operate on *frames*, not single vectors.

Knows nothing about: which graph it belongs to, whether it is part of a density.

### 1.3 Manifolds and quantity kinds — `skoro-riemann`, `skoro-so3`, `skoro-se3`, `skoro-linalg`

`Manifold` / `DynManifold` supply `dimension`, retraction and inverse retraction. Implementations
span flat spaces (`Real`, `Euclidean<D>`, `EuclideanD`, `EuclideanMatrix`), matrix groups (`SO3`,
`SE3`) and the lower-triangular covariance-factor manifold (`Tril`).

**The invariant that shapes everything downstream: dimension is not storage.** `SE3` is
6-dimensional but 12 components wide. A Jacobian column is a *dimension*, so `SE3` contributes 6
columns and never 12. Points never flatten into tangent coordinates; only tangents do.

**Discrete quantities are excluded structurally, not by convention.** `Kind` is a sealed trait with
exactly two implementors, `Continuous` and `Discrete` (`skoro-core/src/kind.rs`). A discrete
quantity has no tangent space, so `inverse_retract` is undefined for it and no residual vector
exists — and because the kind is a type parameter threaded through the builder API, a discrete
quantity cannot reach a tangent layout at all. The discrete machinery lives in
`skoro-model/src/discrete/`.

### 1.4 The graph — `skoro-graph`

An arena of nodes and edges representing the model's logical structure:

| Node | Role |
|---|---|
| `Constant` | Fixed data |
| `Parameter` | Free leaf, to be inferred |
| `Stochastic` | A conditional random variable with a distribution |
| `Observed` | A `Stochastic` bound to a value — this is what conditions the model |
| `Mapped` | A deterministic transformation of other nodes |
| `LogDensity` | A real-valued mapped node that scores the model |

`compile_backward_from(targets)` topologically sorts the ancestor closure of `targets` and lowers
it into a `CompiledSequence` of `CompiledStep`s, replacing `HashMap` ids with contiguous indices
and type-erasing each operation. Evaluation walks the sequence forward, recording a `Tape`; the
reverse pass walks the tape.

**Derived state, and its invariant.** Alongside `nodes` the graph keeps `children`,
`log_likelihoods` and `stochastic_to_log_density`. All three are *derived* from `nodes`, and the
density direction's source of truth is the node itself (`LogDensityNode::stochastic_id`);
`stochastic_to_log_density` exists only to make the reverse lookup O(1). `Graph::validate_indices`
recomputes all three from `nodes` and reports the first disagreement, and runs as a debug assertion
after every mutating method — so a method that leaves an index inconsistent fails the test that
exercised it, rather than an unrelated later read.

Knows nothing about: what an EKF or an optimizer is.

### 1.5 The model facade — `skoro-model`

`ModelServer<B>` owns the graph; `ModelClient<B>` is a thread-safe handle over it, so users add
quantities and run passes without mutably borrowing the whole model.

`Q<B, M>` is the user-facing proxy: a node `Id`, a client handle, and a phantom manifold `M`. It is
where the ergonomics live — arithmetic operators, `inverse_retract`, `norm_squared` — all lowered
through `register_map1 … register_map12` into `Mapped` nodes backed by real `DiffOperation`s.

`TangentLayout` fixes an ordering over a set of quantities: `Vec<TangentBlock { id, offset,
dimension }>` plus a reverse index and a total. This is a **variable ordering with block
structure** — the same object GTSAM calls `Ordering`. `jacobian_jvp` and `jacobian_vjp` assemble a
Jacobian given a column layout and a row layout.

### 1.6 Estimators — `skoro-dense`, `skoro-levenberg-marquardt`

`skoro-dense` is host-resident `f64` linear algebra backed by `faer`: Cholesky, triangular and SPD
solves, column-pivoted-QR least squares. It is a **leaf crate** depending on nothing else in the
workspace, so only estimators pay for `faer`.

`skoro-levenberg-marquardt` minimizes `½Σ‖rᵢ(x)‖²` over any model whose factors carry whitened
residuals.

### 1.7 The type gate

This is the most load-bearing decision in the library and the one least visible from any single
file. Four traits form a deliberate lattice:

| Trait | Adds | Can grow the model? |
|---|---|---|
| `ModelClientOps` | evaluate, JVP/VJP/HVP, `observe`, `demote_to_parameter`, `remove`, `tangent_layout` | **No** |
| `ModelBuilderOps: ModelClientOps` | `add_quantity`, `update_quantity`, `mark_likelihood` | Yes |
| `NlsStructure: ModelClientOps` | `certificate()`, `residual_layout()` | No |
| `GaussianModelClientOps: ModelClientOps` | `add_gaussian_quantity` | Yes |

The consequence: **an estimator bounded on `NlsStructure` can evaluate and differentiate but cannot
invalidate the evidence it was handed**, because `ModelClientOps` contains no way to add a factor
(`skoro-model/src/structure.rs`). It needs no second client parameter and no runtime revalidation.

That evidence is `ResidualCertificate`. It is unforgeable not because the type is sealed but
because entries can only enter through `record`, which demands a `WhitenedResidual` bound — so a
hand-built certificate is empty and there is no way to insert a fabricated id.
`crates/skoro-model/tests/compile_fail/` pins this. The module comment states the condition under
which every claim in it lapses: *if `record` ever grows an unbounded sibling*.

---

## 2. A worked example

A linear-Gaussian hidden Markov model with a sliding window: the state chain grows, old states are
summarized and retired, and the window stays bounded.

> **This snippet is verified by reading, not by compiling.** Every call is checked against
> `crates/skoro-levenberg-marquardt/tests/common/mod.rs` and `crates/skoro-normal/tests/demote.rs`.
> Three calls, marked `GAP`, do not exist yet — they are the subject of §4.

```rust
use skoro_linalg::{matrix::matmul, tril::Tril};
use skoro_model::{ModelBuilderOps, ModelClientOps, model::ModelClient, q::Q};
use skoro_normal::distribution::Normal;
use skoro_riemann::{
    euclidean::{EuclideanD, VectorD},
    matrix::{EuclideanMatrixD, MatrixD},
};

const N: usize = 3;      // state dimension
const M: usize = 2;      // measurement dimension
const WINDOW: usize = 5; // states kept live

let model = ModelClient::<B>::new(Default::default());

// Constants: the prior, the two noise factors, and the two linear maps.
// `Tril` packs a lower triangle row-major, so a covariance is given by its factor.
let x0_mean = model.add_constant(EuclideanD(N), &VectorD::new(&PRIOR_MEAN));
let x0_cov  = model.add_constant(Tril(N), &VectorD::new(&PRIOR_FACTOR));
let dyn_cov = model.add_constant(Tril(N), &VectorD::new(&PROCESS_FACTOR));
let z_cov   = model.add_constant(Tril(M), &VectorD::new(&MEASUREMENT_FACTOR));
let f_mat   = model.add_constant(EuclideanMatrixD(N, N), &MatrixD::new(N, N, &TRANSITION));
let h_mat   = model.add_constant(EuclideanMatrixD(M, N), &MatrixD::new(M, N, &PROJECTION));

// The transition and measurement maps are ordinary quantity-level operations. Nothing about
// them is special-cased: each `matmul` becomes a `Mapped` node with a real `DiffOperation`.
let f = |x: &Q<B, EuclideanD>| matmul(&f_mat, x);
let h = |x: &Q<B, EuclideanD>| matmul(&h_mat, x);

// t = 0 — the prior is just a quantity with a distribution.
let mut x = model.add_quantity(Normal::new(x0_mean, x0_cov))?;
let mut states = vec![x.clone()];
let mut measurements = Vec::new();

for z_obs in observations {
    // Extend the chain: a new state conditioned on the last, and its measurement.
    x = model.add_quantity(Normal::new(f(&x)?, dyn_cov.clone()))?;
    let z = model.add_quantity(Normal::new(h(&x)?, z_cov.clone()))?;
    model.observe(&z, z_obs)?;
    states.push(x.clone());
    measurements.push(z.clone());

    if states.len() <= WINDOW {
        continue;
    }

    // ---- The window slides ----

    // 1. Estimate over everything still live.
    let free = model.tangent_layout(&states.iter().map(Q::id).collect::<Vec<_>>())?;
    let report = estimator.solve(&model, &mut points, &free)?;              // GAP 1

    // 2. Replace the oldest state with a factor on what it still touches, linearized at the
    //    estimate. Demotion strips its own density and its parents but keeps its identity, so
    //    everything downstream is undisturbed.
    let oldest = states.remove(0);
    let survivor = states[0].clone();
    model.demote_to_parameter(&oldest)?;
    let factor = estimator.marginal_likelihood(&model, &points, &oldest, &[survivor])?;  // GAP 2
    model.mark_likelihood(&factor)?;

    // 3. Retire the state and measurement the factor replaced, with everything below them.
    let stale = measurements.remove(0);
    model.remove_closure(&[oldest.id(), stale.id()])?;                      // GAP 3
}
```

### What the example is asserting

- **A model is built incrementally, by ordinary calls.** There is no "declare the whole graph
  first" phase, and no separate compilation step the user drives.
- **The dynamics are user code.** `f` and `h` are closures over `matmul`; a nonlinear model swaps
  them for anything expressible as `Q` operations, and the derivative machinery follows for free.
- **Conditioning is `observe`.** It converts a `Stochastic` node to `Observed` in place, keeping
  the same id and parents.
- **Marginalization is three moves**: demote the quantity so it no longer carries its own density,
  attach a factor that summarizes what was lost, then delete what the factor replaced.

### Corrections against the sketch this was based on

Each of these is a real constraint rather than a typo, which is why they are worth stating:

- `Normal::new(mean, std_dev)` takes both arguments **by value**, so a covariance reused across
  ticks needs `.clone()`. The `Q` proxy is cheap to clone by design.
- `demote_to_parameter` is **singular** and takes a `Q` tag, not a slice of ids. There is no
  `demote_to_parameters`.
- `observe` is a **model call**, `model.observe(&z, value)`, not a method on the tag.
- Using a `Vec` as a sliding window wants `remove(0)`; `rotate` reorders without dropping anything
  and silently retires the wrong elements.

---

## 3. How the parts move

One `solve` call, followed down. Levenberg-Marquardt is the reference estimator throughout.

### 3.1 Building the tangent blocks

`TangentLayout::from_ids(graph, ids)` resolves each id to its manifold, reads its `dimension`, and
assigns contiguous offsets in the caller's order. Ordering is caller-specified deliberately: a
filter's state ordering is a modelling decision, and a block structure the caller chose is one they
can read a covariance matrix against.

Two layouts are in play per solve: **`free`**, the columns — which quantities are being inferred —
and **`residuals`**, the rows, which `NlsStructure::residual_layout` derives from the certificate
in the order the factors were recorded.

A layout holds no tensors and no backend parameter. It is built once and reused across every
iteration.

### 3.2 Assembling the Jacobian

`jacobian_vjp(client, points, inputs, outputs)`:

1. `outputs.identity_frames(batch, device)` builds `Tensor::eye(D_out)` and hands each row block
   its `[batch, d_o, D_out]` slice.
2. One reverse pass over the whole graph carries all `D_out` frames.
3. Each input block comes back as `[batch, d_j, D_out]` — that is `Jᵀ` — concatenated in column
   order and transposed by `swap_dims(1, 2)`.

**The transpose asymmetry is where this pair fails silently.** `jacobian_jvp` needs none, and
getting it wrong yields a well-shaped, wrong `J` whenever `D_in == D_out`. Both paths route through
the single `identity_frames` helper so their index conventions cannot drift.

The structural constraint, which is a constraint on *how models are written*: **Jacobian columns
must be `Parameter`/`Stochastic` leaves and rows must be `Mapped`/`LogDensity` nodes.** Cotangents
are only collected at leaves and tangents only emitted at operations. A bad row is reported as
`NotMapped`; a bad column is *not* — it yields a silently zero column, and that is the one failure
mode here no return value announces.

**The cost, stated honestly.** One pass with `D_out` frames costs `D_out × |graph|`, and that is
**insensitive to sparsity**. A factor graph's `J` is mostly zeros and none of that structure is
available. Two specific consequences:

- `identity_frames` materializes `batch × D_out²` elements of which `batch × D_out` are nonzero.
- Every factor's subgraph is traversed with all `D_out` frames when only `d_f` of them are nonzero
  on it.

This is the binding constraint on the whole library, and §4 returns to it.

### 3.3 Crossing to the host

`J` and `δ` are the only tensor-typed values in an LM iteration; everything between them is host
`f64`. `Dense::split_tensor` moves the whole batch across in **one** transfer and then slices host
side — the obvious alternative, `narrow` per row, is `N` device reads, and on a cube backend each
one fences the queue.

The step itself is column-pivoted QR on the row-augmented system:

```
(JᵀJ + λI) δ = −Jᵀr    ⟺    min ‖[J; √λI] δ − [−r; 0]‖
```

The right-hand form is solved directly, so the iteration runs at condition `κ` rather than `κ²`,
**at every damping value** rather than only at `λ = 0`. Forming `JᵀJ` would square the conditioning;
avoiding it is the entire argument for the host boundary, since `burn` has no QR.

A host `f64` solve is not `f64` end to end: when `B::FloatElem = f32` the Jacobian arrives already
rounded and `δ` is rounded back. What it buys is `f64` *inside* the solve, which is where the
conditioning lives.

### 3.4 Taking the step

`points.retract_flat(free, δ)` — **a retraction, not vector addition**. This is why `free` is a
layout of dimensions: `δ` is 6 wide for an `SE3` pose, and `Exp` maps it onto the manifold. A zero
tangent is an exact no-op on every manifold, which is what lets one retraction serve both the
accept and reject branches of the iteration.

Per batch row, LM carries its own `λ`, its own cost, its own accept/reject and its own termination.
A batch is `N` *independent* problems — multi-start search — sharing one Jacobian pass, because
frames and batch are different axes.

### 3.5 Where caching happens, and where it does not

**Constant folding does happen.** `check_cache` evaluates a mapped node whose parents are all
concrete and stores the result on the node. `observe` extends this: fixing a quantity makes its
descendants concrete, so they fold too. The fold walk visits each descendant **in dependency
order**, which is load-bearing rather than an optimization — a plain visited-set walk can dequeue a
node before the last of its parents has folded, and refusing to revisit leaves it unfolded
permanently. Whether that happens depends on hash iteration order, so the bug would be
intermittent. `clear_cached_descendants` drops folds when a mutation makes a node non-constant; it
*may* use a plain visited set, because clearing is order-independent and idempotent.

**Compilation does not cache.** `compile_backward_from` runs fresh at ten call sites. Per LM sweep,
`step_columns` compiles for `jacobian_vjp`, again for the residual evaluation, and `costs()`
compiles again. Each call topologically sorts the ancestor closure and rebuilds every step.

**And it cannot cache today**, which is the deeper problem. `is_differentiable_parent` returns
`value.is_none()` for mapped nodes, and that feeds `constant_inputs` and `forward_only` into every
compiled step. So *which steps are on the tape* is derived from the fold cache — value state, not
topology. A sequence keyed on topology alone would go stale whenever a fold changed. This is a
layering inversion: an optimization determines a structural property.

---

## 4. What is missing, and the plan

The example in §2 names three calls that do not exist. They are not independent of the structural
issues in §3, and the ordering below reflects that.

> **On the defect ledger.** `graph-marginalization.md` §8 tracked seven numbered defects in
> `skoro-graph`. **All are now closed** — orphaned densities are discoverable and reclaimable,
> `remove` has a transitive teardown, the `observe` walk is ordered, `check_cache` reports rather
> than panics, and the `remove` TODO is resolved and tested. Nothing is lost by that document
> going away.

### 4.1 The three gaps

**GAP 1 — there is no `Estimator` abstraction.** `solve` is a free function over `NlsStructure`
(`skoro-levenberg-marquardt/src/lib.rs`). "Plug an estimator into the model" has no type. Propose a
trait in `skoro-model`, so LM implements it and an EKF later can too:

```rust
pub trait Estimator<B: Backend> {
    type Report;
    fn solve<M: NlsStructure<B>>(
        &self,
        model: &M,
        points: &mut PointMap<B>,
        free: &TangentLayout,
    ) -> SkoroResult<Self::Report>;
}
```

`LmOptions` becomes the implementing struct's own state, which is where damping schedules and
tolerances belong. Small and mostly mechanical.

**GAP 2 — there is no marginal likelihood.** The largest piece, and the one that makes windowed
smoothing possible.

At the estimate `θ̂`, a Laplace approximation over the Gauss-Newton model gives precision
`Λ = JᵀJ` in *tangent* coordinates. Retiring a block means marginalizing it out, which is the Schur
complement:

```
Λ_survivor = Λ_bb − Λ_ba Λ_aa⁻¹ Λ_ab
```

The numerics are composition of what `skoro-dense` already has — `cholesky` and `spd_solve` — not
new factorizations. What is needed:

- a `MarginalLikelihood` capability on the estimator, returning a `Q<B, Real>` factor node;
- the Schur complement over `Dense`;
- a way to *express* the result as a factor.

**That last point is the blocker.** The factor is a Gaussian over the retired quantity's Markov
blanket. If the blanket holds more than one quantity, there is no way to express a joint over
several manifolds: `Normal<B, M>` is a single manifold, and a cross-manifold joint needs a
**product manifold**, which does not exist. Scope the first version to a **single-quantity
blanket** — which is exactly the chain structure in §2, where the blanket is the next state — and
implement it as `update_quantity` on the survivor. Record the product manifold as the
generalization; it is what a loop-closure or landmark model would need.

There is also a correctness precondition worth stating: the Laplace marginal is **exact** for a
linear-Gaussian model and an approximation otherwise. That is what makes it testable (§4.4).

**GAP 3 — `remove_closure` is not reachable from the client.** `Graph::remove_closure` exists;
`ModelClientOps::remove` is still leaf-only and all-or-nothing. Expose the closure form through
`ModelServer` and `ModelClientOps`. Small, and it unblocks the example's teardown.

### 4.2 The structural work

**4. Decouple compiled structure from the fold cache.** Move folded values off the nodes into
`Graph::folds`, drop `MappedNode::value` and `LogDensityNode::value`, and make
`is_differentiable_parent` a function of node kind alone. The read sites are
`compile_backward_from`, `check_cache`, `clear_cached_descendants`, the `observe` walk, and
`mark_likelihood_impl`.

Note that `constant_inputs` does double duty — it marks a step off the tape *and* short-circuits
the forward path — so the forward pass must consult the fold table at execution time or the
optimization is silently lost. "More steps on the tape, identical answers" is an assumption to
confirm against the finite-difference suite, not to assert.

Largest breaking change here; nothing after it is safe without it.

**5. Cache the `CompiledSequence`** behind a topology `revision` counter. The justification is
**enablement, not speed**: compilation is O(|graph|) bookkeeping against a pass doing
O(`D_out` × |graph|) of tensor arithmetic, so expect a few percent. It must land because item 6 is
impossible without it. `Graph` derives `Clone`, so a manual impl must drop the cache — a clone
sharing a revision space with its origin would serve a sequence against the wrong topology.

The invalidation surface has one trap worth naming: `observe`/`unobserve` keep the same node id
*and* the same parents, but flip the node between being a cotangent collection point and not. A
stale sequence therefore yields **right values and wrong derivatives**.

**6. Per-factor compiled subsequences, and block-sparse `J`.** Extend `FactorRecord` with the roots
its residual depends on and a cached subsequence for that subgraph.

This is the change that matters most, and it sidesteps the standing objection to storing adjacency.
That objection — that a `FactorGraph` would be a *derived projection* needing invalidation across
the whole mutation API — is correct about a derived global index. It does not apply to a
**captured** one: `ResidualCertificate::record` runs at the moment the builder creates the factor,
when its roots are already known. It is captured evidence, the same pattern that makes the
certificate unforgeable.

Cost to price in rather than assume away: per-factor blocks arrive as `n_factors` small tensors, so
host transfer becomes `n_factors` device reads instead of one — the exact trade `Dense::split_tensor`
was written to avoid. Measure before making block assembly the default, and keep the flat path
behind the same `NlsStructure` seam.

**7. `skoro-dense` zero-copy.** `to_faer` builds an owned `Mat` by a transposing scalar copy, and
every faer entry point then copies it again into its own storage (`ColPivQr::new` is `A.to_owned()`).
Use `MatRef::from_row_major_slice` for the input matrix of all five entry points to delete one of
the two copies. **Leave the rhs of `solve_lower`/`solve_upper` owned** — it is worked on in place,
and a row-major view has `row_stride == ncols`, which makes an in-place triangular solve march
against the stride. Also drop the double copy in `Dense::split_tensor`. Independent of everything
else and can land at any time.

### 4.3 The circularity to avoid re-creating

The six documents this replaces each deferred a piece of work to a consumer that did not exist:
the device-side batched solve, adjacency, and block assembly were all gated on "a batch smooth over
hundreds of poses" — which could not be built efficiently until they landed. One deferral counted
three times.

**Item 6 breaks the cycle**, and it is the right one to force, because it is the only one whose
*interface* the others depend on. A device-side batched solve is a swap behind names `skoro-dense`
already fixes; adjacency is needed only in the captured, per-factor form item 6 introduces.

### 4.4 Sequencing and verification

```
GAP 3  →  GAP 1  →  item 4  →  item 5  →  GAP 2  →  item 6
                                  item 7 — any time, independent
```

GAP 2 lands after the cache because a marginal likelihood is computed once per window and would
otherwise recompile the entire model each time.

How each is checked:

- `cargo test --workspace` after every item.
- **Item 4**: the JVP/VJP agreement tests and the `SE3` central-difference check through the
  retraction are the guard that decoupling changed no derivative.
- **Item 5**: assert cache hits across LM iterations *and* that mutation invalidates. The
  correctness test is `observe` mid-loop — derivatives must change.
- **Item 6**: per-factor blocks must reassemble to exactly the flat `jacobian_vjp` result.
- **GAP 2**: the check is statistical rather than structural. On a **linear-Gaussian** HMM the
  Laplace marginal is exact, so a windowed run must reproduce the full-batch smoother's estimate to
  solver tolerance. That is a real test of the mathematics, not of the plumbing, and the linear
  fixture in `skoro-levenberg-marquardt/tests/common/mod.rs` is the model for it.

### 4.5 Where the library stands

Model building is genuinely estimator-agnostic and well factored: nothing in `skoro-graph` knows
what an estimator is, the type gate is enforced by the compiler rather than by discipline, and
`TangentLayout` is the right abstraction — it survives a sparse rewrite untouched, as does
flattening for *vectors*, since `r` and `δ` are dense whether or not `J` is.

The library serves **filtering** well today. It does not yet serve **smoothing**, and the reason is
narrower than "it needs sparse linear algebra": it needs block-structured Jacobian assembly, which
needs captured per-factor adjacency and cached compiled subsequences, which needed the graph's
derived state to stop leaking. That last prerequisite is done. What remains is a chain of concrete
changes, not an architectural rewrite.
