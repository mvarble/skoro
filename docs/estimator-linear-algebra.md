# Estimator Linear Algebra

Where the dense linear algebra of an EKF or a Levenberg-Marquardt iteration actually runs: on `burn`
tensors through the backend, or on the host in a dedicated library. A decision document, and a
correction to a premise the other designs lean on.

Companion to `flattened-tangent-spaces.md`, whose §6 opens this question by naming a gap — both
consuming algorithms need a symmetric-positive-definite solve and the workspace has none — and
proposes filling it with hand-written `Tensor<B, 3>` kernels. `sequential-estimation.md` §11 makes
that solve a precondition of its filter phase — phase 2, behind only the graph teardown work — so no
estimator in the roadmap starts until this is settled.

**Conclusion, up front.** For the deterministic estimators — `skoro-ekf` and `skoro-levenberg-marquardt` —
the dense linear algebra runs on the **host, in `f64`, one matrix per solve**, in a new leaf crate
`skoro-dense` backed by `faer`. Everything else in the workspace stays on `burn` tensors, including
every graph operation, the Jacobian assembly that feeds the estimator, and the MCMC family, whose
workload is batching in the first place.

---

## 1. The boundary already exists

The organizing claim, and the reason this is not an ad-hoc dependency choice:

> `sequential-estimation.md` §2 and §10 already say it: **`P` lives in the estimator, keyed by a
> layout, and never in the graph.** That line is also the host/device boundary and the precision
> boundary. The only tensors that cross it are `J` on the way in and `δ` on the way out.

Read the EKF sketch in `flattened-tangent-spaces.md` §7 with that in mind and the split falls out
without being designed:

| quantity | lives where | why |
|---|---|---|
| the point (`x`, poses, biases) | `PointMap<B>`, on the manifold | points never flatten (`flattened-tangent-spaces.md` §2) |
| `F`, `H`, `J` | produced as `Tensor<B, 3>` | they come out of the autodiff engine |
| `P`, `S`, `K`, `JᵀJ`, the Joseph-form update | nothing in the graph refers to them | §2 and §10 of `sequential-estimation.md` |
| `δ` | consumed by `retract_flat` | retraction is manifold-aware and belongs to the graph |

So the estimator's inner arithmetic — every line between `H` and `δ` — already had no representation
in the graph, no `Id`, and no node. Moving it off the backend takes nothing away from anything else,
and it is a two-tensor interface rather than a pervasive one:

```
J = jacobian_vjp(client, points, &state, &meas)   // Tensor<B, 3>, the last tensor-typed step
  ↓ to_host_f64                                   // one conversion in
  … S, K, Joseph form, all host f64 …
  ↑ from_host_f64                                 // one conversion out
points.retract_flat(&state, δ)                    // Tensor<B, 2>, back on the manifold
```

**No signature in `flattened-tangent-spaces.md` §4 or §5 changes.** `flatten_vectors`,
`unflatten_frame`, `identity_frames`, `jacobian_jvp`, `jacobian_vjp` and `retract_flat` keep their
`Tensor<B, 2>` / `Tensor<B, 3>` types. They are the graph's boundary and they stay on the backend.
What this document relocates is only §6 — the dense solve that sits between them.

## 2. There is no free GPU runtime to lose

The premise worth correcting, because the whole tradeoff is usually stated as *"burn tensors give us
GPU execution for free; a CPU library does not."*

**`burn` 0.21 ships no Cholesky, no triangular solve, no general solve, and no QR.** The entire
contents of `burn-tensor-0.21.0/src/tensor/linalg/`:

```
cosine_similarity.rs  det.rs  diag.rs  lu.rs  matvec.rs  outer.rs  trace.rs  vector_norm.rs
```

`lu.rs` is a batched LU with partial pivoting, and its own documentation opens with "The current
implementation of LU decomposition is not fully optimized" (`lu.rs:37`). `solve_for_u_blocks`
(`lu.rs:498`) is private, so even the triangular solve inside it is unreachable.

So the comparison is not *free GPU* against *CPU dependency*. It is:

> **write and maintain a batched, blocked Cholesky and triangular solve** against
> **take a dependency that already has them.**

`flattened-tangent-spaces.md` §6 is already honest about this — its sequencing plan puts
"host-side implementations mirroring `invert_packed_lower_triangular`" as step 1 and the on-device
blocked version as step 2. The burn path *starts* on the host. The question this document answers is
whether step 2 is worth reaching, and for these two estimators the answer is no.

**Host round-trips are already the workspace's status quo for triangular algebra.** `TrilInvert`
(`crates/skoro-linalg/src/tril/invert.rs:89-92`) does exactly this dance:

```rust
let mat_data = mat_packed.clone().into_data().convert::<B::FloatElem>();
let mat_elems = mat_data.as_slice::<B::FloatElem>().unwrap();
let inv_mat_elems = invert_packed_lower_triangular(mat_elems, shape[0]);
let inv_mat_packed = Tensor::from_data(TensorData::new(inv_mat_elems, shape), &device);
```

around a scalar forward-substitution loop. That is cited here as precedent for the mechanics, not as
a target — §8 fences it explicitly.

## 3. QR is the argument no kernel work answers

The strongest reason, and it is about correctness rather than speed.

Levenberg-Marquardt as sketched in `flattened-tangent-spaces.md` §7 solves the normal equations:

```
δ = spd_solve(JᵀJ + λI, −Jᵀr)
```

Forming `JᵀJ` **squares the condition number**. If `J` has condition number `κ`, the SPD system has
`κ²`, and roughly half the available mantissa digits are gone before the solve begins. A QR
least-squares solve of `J δ = −r` never forms the product and works at `κ`. For a
sensor-fusion problem with a stiff mix of units — a rotation in radians next to a translation in
metres next to a bias in metres-per-second-squared — `κ` is routinely large enough for this to be
the difference between a converging optimizer and a diverging one.

**`burn` 0.21 has no QR at all.** Not a fast one; none. And a blocked Householder QR with column
pivoting is materially harder to write, test, and keep numerically sound than the blocked Cholesky
that §6 already scopes as step 2. This is the one requirement that no amount of kernel work answers
cheaply, and it is not a preference.

The same argument extends past QR, to cases the estimators will actually meet:

- **A covariance that has drifted indefinite.** `LDLᵀ` and Bunch-Kaufman (`LBLᵀ`) factor a symmetric
  matrix that is no longer positive definite. A Cholesky-only implementation has nothing to say
  about one, and a hand-written kernel most likely returns `NaN`s that propagate silently into `P`
  for the rest of the run.
- **Failure as a signal.** `faer`'s low-level Cholesky returns a `Result` — `LltError` and
  `cholesky_in_place` sit in the same module (`faer::linalg::cholesky::llt::factor`) — and carries an
  `LltRegularization` parameter for the near-singular case. A filter that learns "the innovation
  covariance stopped being positive definite at tick 41,203" can reset or inflate; one that gets
  `NaN` learns it several thousand ticks later.
- **Rank-deficient Levenberg-Marquardt.** Column-pivoted QR gives a usable answer and a numerical rank on a
  problem with an unobservable direction. Normal equations plus a fixed `λ` hides it.

## 4. Precision, argued from this repository

The precision case is usually made with claims about GPUs. It can be made from the test suite
instead, which is stronger because it is in-repo and already true.

The workspace's per-backend entry points, in every crate that has them:

| suite | backend | `delta` | tolerance |
|---|---|---|---|
| `ndarray_f64.rs` | `NdArrayBackend<f64, i32>` | `1.0e-6` | `1.0e-5` |
| `ndarray_f32.rs` | `NdArrayBackend<f32, i32>` | `1.0e-3` | `1.0e-2` |
| `wgpu_f32.rs` | `CubeBackend<WgpuRuntime, f32, i32>` | `1.0e-3` | `1.0e-2` |

Two facts, both plain from `crates/skoro-linalg/tests/`:

1. The tolerance tracks the **element type**, not the device: three orders of magnitude separate
   `f64` from `f32`, and the `f32` CPU and GPU suites agree with each other exactly.
2. **There is no `wgpu_f64` suite anywhere in the workspace.** So the tolerance the estimators want
   is one the device path has never been exercised at.

The workspace's own testing doctrine says the same thing from the other side:
`.claude/skills/skoro-diff-operation/references/testing.md` prescribes `(1e-7, 1e-4)` on `f64` and
`(1e-3, 1e-1)` on `f32`. Those are the tolerances for *first derivatives of a single operation*. An
EKF composes a Jacobian, a `κ²` solve, and a rank-1-ish covariance update, then feeds the result
back into itself, once per tick, forever.

**On whether a cube backend can do `f64` at all, this document declines to rule.**
`CubeBackend<R, F, I>` is generic over its float element, so nothing in *skoro* forbids it. `burn`'s
own dtype assertions (`burn-wgpu-0.21.0/src/lib.rs:142-178`) are inconsistent across feature arms —
`!supports_dtype(F64)` under `vulkan`, under `metal`, and on macOS; `supports_dtype(F64)` on the
default arm — so it is a runtime and driver question, not a design one. Nothing here is built on it
either way.

### The caveat that must not be buried

A host `f64` solve is **not** `f64` end to end, and pretending otherwise would be the most damaging
thing this document could do.

When `B::FloatElem = f32`, the Jacobians arrive from the autodiff engine already rounded to `f32`,
and `δ` is rounded back to `f32` on the way into `retract_flat`. What the host solve buys is `f64`
*inside the solve* — which is precisely where the `κ²` squaring of §3 lives, so it is the right place
to spend it — but the state, the Jacobians, and the correction are all still `f32`.

> **A filter that wants an `f64` state should run `NdArrayBackend<f64, i32>`.** And then the
> "host round-trip" this document proposes is a `memcpy` at worst, because `burn-ndarray` tensors are
> already host memory.

That is not a footnote; it is the recommended default configuration for filtering, and it makes the
cost argument of §5 nearly vacuous for the case the design actually targets.

To state the agreement with `sequential-estimation.md` §8 explicitly, since the two documents would
otherwise imply different defaults for the same first filter: **nothing in that actor shape
requires a cube backend.** Ingest, driver, and readers are a threading arrangement, not a device
one, and a retention-1 filter holding a tick or two of structure is not a workload the `ndarray`
backend struggles with. The performance point in §8 there is reusing one compiled sequence across a
tick's several evaluation passes — cross-tick hoisting is not available, since the graph is rebuilt
every tick — which is a backend-independent win either way.

## 5. What one-matrix-at-a-time costs

> **Amended.** This section was written as "what batch 1 costs", on the offered concession that an
> estimator may assume `Point<B>` is always `[1, _]`. The concession turned out to be stronger than
> the argument needs, and `skoro-levenberg-marquardt` now accepts a batch of **N independent
> problems** — multi-start optimization, where several starting points are run at once to see which
> local optima they find.
>
> **Nothing below changes.** The claim was never `N = 1` versus `N = 20`; it was *host* versus
> *device kernels*. Every premise survives: still host, still `f64`, still **one matrix per solve**
> — the batch axis is split around the factorizations rather than pushed into them — and still
> column-pivoted QR, which §3 makes load-bearing and which `burn` has no counterpart for at any batch
> size. What §5 rules out is 4,096 MCMC chains, where the batch axis *is* the workload. Twenty
> independent 6×20 QR solves in a host loop is not that, and is not the case §8 defers.
>
> The expensive half is shared rather than looped: one `jacobian_vjp` pass produces every row's
> Jacobian, because frames and batch are different axes.
>
> One real limit, which belongs here rather than in the crate: constants and observations are rank-1
> storage that the compiler expands to the running batch, so **every row sees the same data**. "One
> dataset, N starting points" works; "N datasets" does not.

Making a host library viable needs only that each *solve* is one matrix. That is not free, and the
costs are specific.

`Point<B>` is `[batch, storage]` (`crates/skoro-backend/src/lib.rs:77`) and `TangentFrame<B>` is
`[batch, dimension, frame]` (`:100`). Fixing `batch = 1` collapses exactly the axis a GPU is
good at. Whether that matters depends entirely on what else is in the axis:

**Where it costs nothing.** A filter over a handful of `SE3` poses and biases has `N` in the tens.
`sequential-estimation.md` §3 is emphatic that a retention-1 filter keeps *one tick* of structure in
the graph — two transiently, while the previous tick is still being linearized. A `[1, 20, 20]`
Cholesky dispatched to a GPU is launch overhead wearing a matrix
decomposition as a hat. The frame axis of the Jacobian pass is where the parallelism is, and that
pass stays on the backend.

**Where it costs something real, and why the recommendation is scoped.** Ensemble and particle
filters, and the batched-chain MCMC family — `skoro-hmc`, `skoro-mala`,
`skoro-metropolis-hastings`, all named as future `TangentLayout` consumers in
`flattened-tangent-spaces.md` §7 — have batching as their *entire* workload. Running 4,096 chains is
running `batch = 4096`, and a host loop over 4,096 solves is simply the wrong tool — which is a
statement about the *count*, not about the axis existing.

> **This is why the recommendation is scoped to the deterministic estimators.** `skoro-ekf` and
> `skoro-levenberg-marquardt` take the host `f64` path. The MCMC family stays on `burn` tensors, where the
> batch axis is the point. Two answers, because there are two workloads.

**Synchronization.** On a cube backend, reading `J` to the host fences a queue that would otherwise
stay asynchronous, so the estimator becomes the synchronization point of the driver loop
(`sequential-estimation.md` §8). On `ndarray` it is not a transfer at all. The honest statement:

> **The linear-algebra choice and the backend choice are not independent.** Host `f64` linear algebra
> wants the `ndarray` `f64` backend; a cube backend wants the batched workloads, which are the ones
> this document leaves on tensors.

**Two numeric worlds.** The conversion code is the new bug surface, and it is the part to test first
(§9). One consolation: the host path never routes through `burn`'s `DType`, so it is immune to the
specific silent-demotion trap this workspace has already been bitten by, where mixing `f32` and
`f64` `NdArray` backends in one test binary quietly demotes the dtype.

## 6. Which library

Three candidates, judged on what can be checked rather than on remembered benchmarks.

| | **faer** 0.24.4 | **nalgebra** 0.35.0 | **ndarray-linalg** |
|---|---|---|---|
| license | MIT | Apache-2.0 | MIT/Apache-2.0 |
| build | pure Rust | pure Rust | **links LAPACK** (OpenBLAS / Netlib / MKL) |
| Cholesky | `LLᵀ`, `LDLᵀ`, `LBLᵀ`, pivoted `LLᵀ` | `LLᵀ` | via LAPACK |
| QR | plain and column-pivoting | plain | via LAPACK |
| LU | partial and full pivoting | partial | via LAPACK |
| SVD / eigen | `Svd`, `Eigen`, `SelfAdjointEigen`, `GeneralizedEigen` | SVD, symmetric eigen | via LAPACK |
| least squares | `SolveLstsq` / `SolveLstsqCore` traits | manual | via LAPACK |
| threading | optional `rayon` | — | via BLAS |

`faer`'s surface above is read from its 0.24.4 documentation: `faer::linalg` exposes `cholesky`
(with `llt`, `ldlt`, `lblt`, `llt_pivoting`), `qr` (`no_pivoting`, `col_pivoting`), `lu`
(`partial_pivoting`, `full_pivoting`), `svd`, `evd`, `gevd`, `householder`, `jacobi`,
`triangular_solve`, `triangular_inverse`, and a `solvers` module with `Llt`, `Ldlt`, `Lblt`, `Qr`,
`ColPivQr`, `PartialPivLu`, `FullPivLu`, `Svd`, `Eigen`, `SelfAdjointEigen` and the `Solve` /
`SolveLstsq` traits.

**Recommend `faer`**, on two arguments, both defensible without a benchmark:

1. **It preserves a pure-Rust build.** `faer`'s non-optional dependencies are `bytemuck`,
   `dyn-stack`, `equator`, `faer-traits`, `gemm`, `generativity`, `libm`, `nano-gemm`,
   `num-complex`, `num-traits`, `pulp`, `reborrow` — no `build.rs` hunting for a system BLAS, no
   Fortran toolchain, no vendored C. This is what rules out `ndarray-linalg`, whose LAPACK link is a
   cross-compilation and licensing burden the workspace has so far entirely avoided (its own
   `Backend` trait over burn, its own autodiff engine, its own kernels).
2. **It has the decompositions §3 argues for.** Column-pivoted QR, `LDLᵀ`, and Bunch-Kaufman are
   precisely the three things that make the difference between an estimator that reports trouble and
   one that emits `NaN`. `nalgebra` covers Cholesky and plain QR but not the indefinite-symmetric
   cases.

Two arguments deliberately **not** made:

- **Not MSRV.** `edition = "2024"` already requires ≥ 1.85, so `nalgebra`'s 1.89 is not a constraint.
- **Not relative speed.** Any claim here about `nalgebra::DMatrix` versus `faer`'s blocked kernels
  would be recalled rather than measured, and would age badly in a design document. If it ever
  matters, settle it with a `diol` or `criterion` bench at the `N` the model actually has —
  somewhere in 10–200 for the filters this workspace targets — and record the number here.

Start from:

```toml
faer = { version = "0.24", default-features = false, features = ["std"] }
```

✅ **Verified against 0.24.4 as published.** There is no `linalg` feature — the published set is
`std`, `rayon`, `npy`, `perf-warn`, `nightly`, with `std` and `rayon` on by default. Dropping the
defaults and re-adding `std` therefore drops `rayon` and `npy`, and the whole of `faer::linalg` is
available unconditionally. The line above is what `crates/skoro-dense` builds against.

> **Write `skoro`-side signatures concretely; treat every `faer` call site as illustrative.** Its API
> has churned across versions, and a fabricated snippet in a design document is the likeliest way for
> this file to end up containing something false.

## 7. `skoro-dense`

A new leaf crate, depending on `skoro-backend` and `skoro-core` and **nothing else in the
workspace** — in particular not `skoro-graph` or `skoro-model`.

**Why not `crates/skoro-linalg/src/dense/`, as `flattened-tangent-spaces.md` §6 proposes.**
`skoro-linalg` depends on `skoro-graph` and `skoro-model`, and is pulled in by every user of
`Normal` and `Tril` — which is to say, by every model with a Gaussian in it. Putting `faer` there
taxes the whole workspace with a dependency that only two estimator crates use. A separate crate
also matches the shape §7 of that document already describes, where each algorithm is its own crate.

The surface, sketched:

```rust
/// A single dense matrix, host-resident, always `f64`.
///
/// Constructed from a `[1, m, n]` tensor; errors with `BatchNotSingleton`
/// on a leading dimension other than 1, rather than silently taking slice 0.
/// A batched caller reaches the same factorizations through `split_tensor` /
/// `split_vector` / `stack_vector`, which turn an `[N, ..]` tensor into N of
/// these in **one** host transfer — so the batch axis is handled around the
/// factorizations rather than inside them, and none of the five below changes.
pub struct Dense { rows: usize, cols: usize, data: Vec<f64> }

impl Dense {
    pub fn from_tensor<B: Backend>(t: Tensor<B, 3>) -> SkoroResult<Self>;
    pub fn to_tensor<B: Backend>(&self, device: &B::Device) -> Tensor<B, 3>;

    /// `[1, n]` <-> a column. **Tangent and residual vectors only** — `δ`, `r`, an
    /// innovation. A `Point<B>` is the same `Tensor<B, 2>` alias but must never come
    /// through here: points stay on the manifold, and `n` is a manifold *dimension*,
    /// not a storage size (`flattened-tangent-spaces.md` §2).
    pub fn from_vector<B: Backend>(t: Tensor<B, 2>) -> SkoroResult<Self>;
    pub fn to_vector<B: Backend>(&self, device: &B::Device) -> Tensor<B, 2>;
}
```

Every dimension crossing this boundary is a `TangentLayout::total()`, so `SO3` is 3 and `SE3` is 6,
never 9 and 12. The type system will not help here — `Point<B>`, `TangentVector<B>` and
`CotangentVector<B>` are all `Tensor<B, 2>` (`crates/skoro-backend/src/lib.rs:77-95`) — so this is a
convention the tests in §9 have to hold up.

Conversions go through `TensorData::convert::<f64>()` on the way in and
`TensorData::convert::<B::FloatElem>()` on the way out, the same mechanism `TrilInvert` already uses
(`tril/invert.rs:89-92`).

The four operations `flattened-tangent-spaces.md` §6 named keep their names, so that document's
consumers read unchanged and a device-side implementation stays addable behind the same seam. The
fifth does not — and the exception is the point:

```rust
pub fn cholesky(spd: &Dense) -> SkoroResult<Dense>;             // lower factor, or NotPositiveDefinite
pub fn solve_lower(l: &Dense, b: &Dense) -> SkoroResult<Dense>;
pub fn solve_upper(u: &Dense, b: &Dense) -> SkoroResult<Dense>;
pub fn spd_solve(a: &Dense, b: &Dense) -> SkoroResult<Dense>;

/// Least squares by column-pivoted QR: solves `min ‖J δ − r‖` without forming `JᵀJ`.
/// This is the Levenberg-Marquardt entry point, per §3. **No §6 counterpart, and no device
/// counterpart even in principle** — burn 0.21 has no QR.
pub fn lstsq(j: &Dense, r: &Dense) -> SkoroResult<Dense>;
```

> **`lstsq` gives the *undamped* step; damping is supplied by the caller, not by a second function.**
> Levenberg-Marquardt's `(JᵀJ + λI) δ = −Jᵀr` is exactly the least-squares problem
> `min ‖[J; √λ I] δ − [−r; 0]‖`, so an LM iteration calls `lstsq` on the row-augmented matrix and
> gets its damped step at condition `κ` instead of `κ²`. This is worth stating because it removes the
> one apparent reason to prefer `spd_solve` for `λ > 0`: the QR path is available at *every* damping
> value, and §3's argument therefore covers the whole algorithm rather than only its undamped limit.
> The augmentation costs `N` extra rows, which is nothing next to squaring the condition number.

> **`lstsq` is why the seam is not a full abstraction.** Four of these five could be reimplemented
> on `Tensor<B, 3>` behind the same names if a batched consumer ever appears (§8). The fifth is the
> one §3 makes load-bearing, and it has nothing to swap to. An estimator written against `lstsq` is
> committed to the host path; that is a deliberate consequence, not an oversight.

They return `SkoroResult` rather than bare values, which is the §3 argument made structural: a
non-positive-definite innovation covariance is a condition the estimator must be able to observe. A
new `SkoroError` variant — the enum is `#[non_exhaustive]` (`skoro-core/src/error.rs:24`), so this is
additive:

```rust
NotPositiveDefinite { .. },
```

Leave the payload open here. `faer` reports the failure through an `LltError`
(`faer::linalg::cholesky::llt::factor`), and this variant should carry whatever that actually
provides — the offending pivot index if it is available, nothing if it is not. Fix the shape when
the crate is written, against the real type, rather than guessing it now.

## 8. What stays on `burn` tensors

An explicit fence. Without it the natural reading of this document is "route everything through
`faer`", which would be badly wrong.

**`TrilInvert` stays a backend-generic graph operation.** It is a `DiffOperation` living on the tape,
reached from `Normal::log_density` through `std_dev.invert()`
(`crates/skoro-normal/src/distribution.rs:169-170`) — so it runs on **every log-density evaluation of
every Gaussian model on every backend**, inside the compiled sequence, with `vjp` and `hvp` states
memoized as tensors. Reimplementing it over `Dense` would drag a host synchronization and an `f64`
promotion into the hot autodiff path for models that never touch an estimator, and would break its
batching, which is genuine — a log-density is evaluated at `batch = n` routinely. Its host round-trip
is cited in §2 as *precedent for the mechanics*, not as a target for replacement. If it is worth
optimizing, the fix is a cube kernel alongside the existing per-backend implementations
(`skoro-linalg/src/vector/cross_product/{cube,ndarray}.rs` is the pattern), not `skoro-dense`.

Also staying:

- **Everything in `skoro-linalg`** — `Matmul`, `Transpose`, `CrossProduct`, `Tril` and its
  operations. These are graph operations with `forward` / `vjp` / `hvp`; they are not estimator
  arithmetic and this document says nothing about them.
- **The graph boundary of `flattened-tangent-spaces.md`** — `flatten_vectors`, `unflatten_frame`,
  `flatten_frames`, `identity_frames`, `jacobian_jvp`, `jacobian_vjp`, `retract_flat`. Unchanged
  signatures, unchanged types, still `Tensor<B, D>`.
- **The MCMC family** — `skoro-hmc`, `skoro-mala`, `skoro-metropolis-hastings`, per §5.
- **The on-device blocked Cholesky of `flattened-tangent-spaces.md` §6 step 2** — *deferred, not
  cancelled.* It becomes worth building when a batched estimator needs a batched dense solve, which
  is the ensemble-filter and batched-MCMC case. Building it now would be speculative work against a
  consumer that does not exist, and `skoro-dense`'s function names are kept identical so it can land
  behind the same seam later.

## 9. Verification

Per-backend entry points over a shared `common/` module, following
`skoro-linalg/tests/{ndarray_f32,ndarray_f64,wgpu_f32}.rs`. Heed
`.claude/skills/skoro-diff-operation/references/testing.md`: hand-computed expected constants in this
workspace are generated from the implementation and lock in whatever it already did. Prefer
identities, finite differences, and closed forms.

The conversions are the new code, so they are tested first.

1. **Round-trip, both dtypes.** `to_tensor(from_tensor(t)) == t` on `[1, m, n]` for
   `NdArrayBackend<f64, i32>` exactly, and for `NdArrayBackend<f32, i32>` to `f32` precision. A
   non-square `m ≠ n` fixture, since a transposed conversion is the classic silent failure — the same
   hazard `flattened-tangent-spaces.md` §5 flags for the JVP/VJP asymmetry.
2. **`batch > 1` errors.** A `[3, n, n]` input returns `LayoutDimensionMismatch`, never slice 0. The
   single-batch restriction is load-bearing and must fail loudly.
3. **Decomposition identities.** `spd_solve(A, A) ≈ I`; `cholesky(L Lᵀ) ≈ L`; and agreement with
   `TrilInvert` on a packed lower-triangular input converted to dense — carried over from
   `flattened-tangent-spaces.md` §8 item 4, and doubling as a check that §8's fence did not let the
   two implementations drift.
4. **The QR claim, made checkable.** An ill-conditioned least-squares problem — a Hilbert-like `J`,
   or a Jacobian with a stiff unit mix — on which `lstsq` and `spd_solve(JᵀJ, −Jᵀr)` disagree by more
   than the `f64` tolerance, with `lstsq` matching the analytic solution. **Without this test, §3 is
   an assertion**, and §3 is the reason for the whole recommendation.
5. **Indefinite input is an error, not a `NaN`.** `cholesky` on a symmetric matrix with a negative
   eigenvalue returns `NotPositiveDefinite`. Assert on the error variant, not on `is_nan` — and not
   on its payload, which §7 leaves to whatever `LltError` turns out to carry.
6. **End-to-end, unchanged.** The EKF and Levenberg-Marquardt checks already specified in
   `flattened-tangent-spaces.md` §8 items 5 and 6 — exact reproduction of the closed-form Kalman
   filter on a linear-Gaussian model, and convergence of Levenberg-Marquardt on a linear
   least-squares problem (in one iteration undamped, to the same point when damped). This document
   does not change what they assert; passing them is the
   evidence that the boundary conversions corrupted nothing.

Run with `cargo test --workspace`. Note that `skoro-dense` has no cube-backend-specific path of its
own — it is host code — so its own suites are `ndarray_f32` and `ndarray_f64`; the `wgpu_f32`
exposure comes through the estimator crates' end-to-end tests, which is also where the
synchronization cost of §5 will first be visible.
