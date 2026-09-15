# Defect Fix B — `burn-ndarray` scrambles `abs()` on a non-standard-layout tensor

**Branch:** `sequential` · **Date:** 2026-09-02

**Status (after merging the other machine's `sequential`):** the defect is real and confirmed, but
the fix that shipped is `skoro_backend::ndarray_simd_workaround` (`86c70a8`), which covers `abs`
and `recip` at their call sites in `PointwiseAbs` and `LiteralScalarDivisionLeft` — not the
`diag.rs` normalization described below, which is no longer in the tree. The end-to-end
regression test `diag_survives_abs` was kept. `github-issue-abs-simd.md` is the upstream report.

---

## Step 1 — Repro confirmed firsthand

Wrote a scratch test (`crates/skoro-linalg/tests/scratch_abs_repro.rs`, since deleted) against the
workspace's actual pinned `burn-ndarray` 0.21.0, on `NdArrayBackend<f64, i32>`:

```
n=  8 elements= 16 strided_abs_ok=true  contiguous_abs_ok=true
n= 10 elements= 20 strided_abs_ok=true  contiguous_abs_ok=true
n= 12 elements= 24 strided_abs_ok=true  contiguous_abs_ok=true
n= 14 elements= 28 strided_abs_ok=true  contiguous_abs_ok=true
n= 15 elements= 30 strided_abs_ok=true  contiguous_abs_ok=true
n= 16 elements= 32 strided_abs_ok=false contiguous_abs_ok=true
   want [1.0, 3.0, 4.0, 6.0, 7.0, 9.0, 10.0, 12.0, 13.0, 15.0, 16.0, 18.0]
   got  [1.0, 4.0, 7.0, 10.0, 13.0, 16.0, 19.0, 22.0, 25.0, 28.0, 31.0, 34.0]
n= 17 elements= 34 strided_abs_ok=false contiguous_abs_ok=true
n= 20 elements= 40 strided_abs_ok=false contiguous_abs_ok=true
n= 40 elements= 80 strided_abs_ok=false contiguous_abs_ok=true
```

Exactly the reported behavior: correct below 32 elements, transposed at and above it, and always
correct once the tensor is rebuilt contiguous.

### Root cause (primary sources, `burn-ndarray` 0.21.0)

Three source facts fully explain it:

1. `src/ops/simd/base.rs:17` — `pub fn should_use_simd(len: usize) -> bool { len >= 32 }`.
   This is the 32-element threshold, exactly.
2. `src/ops/base.rs:974` — `abs` opens with
   `dispatch_unary_simd!(E, VecAbs, tensor, i8, i16, i32, f32, f64)`, so both f32 and f64 take
   the SIMD fast path.
3. `src/ops/simd/unary.rs:85` — the fast path's only layout guard is
   `input.as_slice_memory_order().is_none()`. That returns `Some` for a **column-major** array,
   which is contiguous *in memory* but whose memory order is not its logical order.
   `unary_scalar_simd_owned` then reads the input via `as_slice_memory_order()` and writes into a
   freshly allocated **standard-layout** output — transposing the elements silently, with no
   shape error.

The producer side: `NdArrayOps::select` (`src/ops/base.rs:922`) delegates to
`ndarray::ArrayBase::select`, which for an axis other than 0 goes through `ndarray::concatenate`
and returns a column-major array.

`burn-ndarray` is *aware* of this hazard — its own `concatenate` helper (`src/ops/base.rs:370`)
ends with:

```rust
// Transform column-major layout into row-major (standard) layout. (fix #1053)
let shape = array.shape().into_shape();
Self::reshape(array, shape)
```

`select` simply never got the same treatment.

### Impact confirmed in this codebase

`Normal::factor` (`crates/skoro-normal/src/distribution.rs:210`) computes
`let log_det = w.diag()?.abs()?.log()?.sum()?;`, and `Tril::diag` gathers with `select(1, ..)`.
So every `Normal` over a 2+ dimensional quantity at a batch large enough to give ≥32 diagonal
elements returned a wrong log density on the ndarray backend.

---

## Step 2 — Upgrade investigation, and why I chose a local workaround

**`burn` 0.21.0 is the current latest stable release.** Read directly from the crates.io sparse
index (`https://index.crates.io/bu/rn/burn`): after `0.21.0` (published 2026-05-07) the only newer
entries are `0.22.0-pre.1`, `-pre.2` and `-pre.3` — all prereleases, the newest published
2026-08-25. There is no newer stable version to bump to, so the "clean drop-in version bump" path
does not exist.

The 0.22 prereleases are additionally **not** a drop-in for this workspace:

- This workspace depends directly on `burn-cubecl` and `burn-wgpu`. In `0.22.0-pre.3`, `burn`'s
  dependency list no longer contains `burn-cubecl`, `burn-ndarray`, `burn-wgpu` or `burn-autodiff`
  at all — the graph was restructured to route everything through `burn-core`.
- The feature sets were renamed wholesale between 0.21 and 0.22-pre (`burn-ndarray/simd` became
  `burn-core/simd`, `optim` was split out, `capture`/`rayon`/`extension` added, etc.).
- MSRV moved 1.92 → 1.95.

Per the task's own guidance ("if it looks like more than a version-number bump… stop that path"),
this is well past a version bump — and it would mean shipping a prerelease. **Stopped, fell back
to Step 3.**

I also spent one web search looking for an upstream issue to cite. Nothing matching this specific
`abs`-on-column-major defect surfaced; the closest related upstream artifacts are the `#1053`
reference already embedded in burn's own `concatenate` comment. The root cause above is therefore
established from the crates.io sparse index plus the `burn-ndarray` 0.21.0 source, not from a
changelog entry.

### Considered and rejected: disabling `burn-ndarray`'s `simd` feature

`simd` is a default feature of `burn-ndarray` 0.21, and the whole defect is gated behind it, so
`default-features = false` would kill the bug class at the root. I rejected it:

- `simd` also backs the binary and comparison dispatches (`add`/`sub`/`mul`/`div`/`equal`/…), so
  disabling it is a broad CPU-backend performance regression to fix one op.
- It is fragile under Cargo feature unification: any future dependency pulling `burn-ndarray`'s
  defaults silently restores the defect, with nothing to catch it.

---

## Step 3 — The fix

**One line, producer-side, in `Tril::diag`'s gather helper.**

`crates/skoro-linalg/src/tril/diag.rs`, `fn diagonal`:

```rust
let diag = tril.select(1, indices);
// … comment explaining the mechanism …
let dims = diag.dims();
diag.reshape(dims)
```

Reasoning:

- **Why reshape-to-the-same-shape.** This is `burn-ndarray`'s own normalization idiom, not a
  guess: the `reshape!` macro (`src/tensor.rs:531`) branches on `is_standard_layout()` and calls
  `.as_standard_layout()` when the input is not standard — and burn uses exactly this trick in
  `concatenate` for the identical problem. It is a metadata no-op when the tensor is already
  standard, so it costs nothing on the cubecl/wgpu backends. Critically, it is *not* an
  `into_data()`/`from_data()` round trip, which would have forced a GPU→CPU→GPU sync in the hot
  path of every `Normal::factor` evaluation.
- **Why the producer (`diagonal`) rather than the consumer (`PointwiseAbs`).** The deciding
  argument is coverage: fixing inside `diagonal` corrects the layout for *every* consumer rather
  than just `abs`, and covers all three of its call sites in `diag.rs` at once (forward, the JVP
  tangent, and the HVP tangent). It also sits in `skoro-linalg`, whose three-backend harness gives
  the regression test ndarray f32, ndarray f64 and wgpu f32 coverage for free — though that
  convenience did not decide it, since I ended up adding a `skoro-pointwise` dev-dependency anyway
  so the test could drive the real `PointwiseOps::abs`.

**Plus a permanent regression test** (this is the part that was missing — the existing suite
provably could not catch this, since the only test that would have was `#[ignore]`d):

- `crates/skoro-linalg/tests/common/tril.rs` — new `diag_survives_abs_impl`, exercising the real
  `diag().abs()` chain from `Normal::factor` at `BATCH * N == 48` diagonal elements (past the
  32-element threshold), with distinct alternating-sign entries so any reordering is detectable.
- `crates/skoro-linalg/tests/common/mod.rs` — wired into the `tests!` macro as `diag_survives_abs`.
- `crates/skoro-linalg/Cargo.toml` — added `skoro-pointwise` as a dev-dependency so the test can
  use the real `PointwiseOps::abs`. No dependency cycle: `skoro-pointwise` does not depend on
  `skoro-linalg`.

I verified the new test genuinely catches the defect by temporarily reverting the one-line fix:

```
test tril::diag_survives_abs ... FAILED
  left  = [1.0, 22.0, 43.0, 64.0, 85.0, 106.0, 127.0, 148.0, 3.0, 24.0, …]
  right = [1.0, 3.0, 6.0, 10.0, 15.0, 21.0, 22.0, 24.0, 27.0, 31.0, …]
```

— the left-hand side is the column-major read, exactly the predicted failure mode. The fix was
then restored and the test passes.

---

## Step 4 — Verification

**`cargo test -p skoro-linalg` — PASS, all three backends:**

```
Running tests/ndarray_f32.rs   test result: ok. 13 passed; 0 failed; 0 ignored
Running tests/ndarray_f64.rs   test result: ok. 13 passed; 0 failed; 0 ignored
Running tests/wgpu_f32.rs      test result: ok. 13 passed; 0 failed; 0 ignored
Doc-tests skoro_linalg         test result: ok. 0 passed; 0 failed; 0 ignored
```

`tril::diag_survives_abs ... ok` on each of the three.

**`cargo test --workspace --no-fail-fast` — PASS.** Every `test result:` line reports `ok`; a
filter for `FAILED` / `failures:` / `N failed` across the whole run returned nothing. The one
`#[ignore]`d test still shows as `1 ignored` in `skoro-normal`, as expected.

**`cargo clippy --workspace --all-targets -- -D warnings` — PASS, clean:**

```
Finished `dev` profile [unoptimized + debuginfo] target(s) in 1.00s
```

**The still-ignored test — it now PASSES, and this fix is what unblocks it:**

```
$ cargo test -p skoro-normal --release --test information -- --ignored covariance_information_matches_the_score_covariance
test covariance_information_matches_the_score_covariance ... ok
test result: ok. 1 passed; 0 failed; 0 ignored; 0 measured; 8 filtered out; finished in 0.06s
```

It also passes in the default debug profile (0.82 s), so it does not need `--release`.

I established causation rather than correlation by temporarily reverting the one-line fix and
re-running it:

```
failures:
    covariance_information_matches_the_score_covariance
test result: FAILED. 0 passed; 1 failed
```

The fix was restored immediately afterwards. So `covariance_information_matches_the_score_covariance`
is now green purely on this fix plus the earlier `TrilInvert` transpose fix. **Per instructions I
left the `#[ignore]` in place** — un-ignoring it is separate follow-up work.

---

## Other `.abs()` and layout-sensitive call sites assessed

I grepped `\.abs()` across `crates/` and assessed every hit.

**Not at risk — scalar `f64::abs`, not a tensor op** (these are plain Rust float arithmetic in test
assertions and host-side code, with no burn tensor involved):
`skoro-so3/tests/common/coefficients.rs:113,126`; `skoro-so3/src/ops/exp/ndarray.rs:542,563`;
`skoro-so3/src/ops/log/ndarray.rs:589,609`; `skoro-normal/tests/information.rs:259`;
`skoro-dense/tests/factor.rs:152,249,263,266`; `skoro-dense/tests/common/mod.rs:36,50,127,143`;
`skoro-dense/src/factor.rs:158`.

**Tensor `.abs()` — all reached through the one op, and all now covered:**

- `skoro-pointwise/src/abs.rs:22,26,36,38,51` — `PointwiseAbs`, the single generic tensor-abs
  chokepoint in the workspace. **Not touched.** Its only realistic non-standard-layout input in
  this codebase was `Tril::diag`'s output, now normalized at the producer. Adding a second
  normalization here would be redundant work on every abs on every backend.
- `skoro-normal/src/distribution.rs:210` and `skoro-normal/tests/structure.rs:85` — both are
  `diag().abs()`, i.e. the fixed path. **Fixed by this change**, not touched directly.

**Layout producers assessed:**

- `skoro-linalg/src/tril/diag.rs:120` (`TrilDiag::vjp`'s `omega.select(1, ..)`) — **not touched.**
  Its output feeds straight into `* mask_tensor`, and I empirically confirmed binary `mul` on a
  column-major operand is correct (it goes through `Zip`, which respects logical order).
- `skoro-linalg/src/tril/utils.rs:69` — already `select(..).reshape(new_shape)`, so already
  normalized.
- `skoro-linalg/src/tril/utils.rs:108` (`pack`) — returns an un-normalized `select` output.
  **Not touched.** Its consumers are `invert`, `vector_product` and `TrilDiag`; none applies `abs`
  to it directly — the packed form always passes through `diag()` first, which is now normalized.
- All ~60 `.transpose()` / `.swap_dims()` sites across `skoro-so3`, `skoro-se3`, `skoro-linalg`
  and `skoro-model` — every one feeds `matmul` or an explicit `reshape`, never `abs`. Not touched.

### Empirical blast radius (f64 and f32, 40-element column-major input)

Only `abs` is affected on f64. Every other op I probed returned correct results:
`neg`, `recip`, `mul_scalar`, `add_scalar`, `log`, `sqrt`, `exp`, `powf_scalar`, `mul`/`add`/`div`/
`sub` by tensor, strided×strided, `sum`, `greater_elem`.

---

## Concerns

**1. On the f32 ndarray backend, `recip` has the same layout defect — plus a precision loss.**
The advisor flagged that `dispatch_unary_simd!` for `recip` lists only `f32`, which is why `recip`
looked clean on my f64 probe. Re-running on `NdArrayBackend<f32, i32>`:

```
recip  ok=false
    want [1.0, 0.33333334, 0.25, 0.16666667, …]
    got  [0.99975586, 0.24993896, 0.14282227, 0.099975586, …]
```

Two distinct problems there. The reordering is the same defect as `abs`. But separately, even on a
**correctly laid out, contiguous** f32 tensor, burn's SIMD `recip` is a low-precision reciprocal
approximation once past 32 elements:

```
contiguous f32 recip n=  8 exact=true  max_rel_err=0e0
contiguous f32 recip n= 16 exact=true  max_rel_err=0e0
contiguous f32 recip n= 31 exact=true  max_rel_err=0e0
contiguous f32 recip n= 32 exact=false max_rel_err=2.7e-4
contiguous f32 recip n= 64 exact=false max_rel_err=2.8e-4
```

~2.7e-4 relative error is far looser than the 1e-6 / 1e-5 tolerances the `ndarray_f32` harness uses
elsewhere. **I did not fix this**, per the instruction not to touch call sites that aren't
genuinely at risk: the workspace's only `recip` uses are in
`skoro-operation/src/ops.rs:798,802,810,822` (`LiteralScalarDivisionLeft`), whose input is a
`Q<B, Real>` of shape `[b, 1]`. Nothing in the workspace produces a `Real` via `select`, so the
*ordering* defect cannot be reached there today. The *precision* issue, however, is live for any
`LiteralScalarDivisionLeft` on the f32 ndarray backend at batch ≥ 32 — worth a separate look, and
not something this task should have silently absorbed.

**2. The fix depends on a burn-internal behavior that is not part of burn's public contract.**
That `reshape` normalizes layout is implementation detail of the `reshape!` macro, not a documented
`burn-tensor` guarantee. It is the same detail burn itself relies on in `concatenate`, so it is
unlikely to change casually — and the new `diag_survives_abs` test will fail loudly on all three
backends if it ever does. That is precisely why the regression test matters more than usual here.

**3. This is a workaround, not an upstream fix.** The underlying `burn-ndarray` defect is still
present for any other consumer of `abs`/`recip` on a column-major tensor. Worth filing upstream
against `tracel-ai/burn` — the fix there is one line, applying the same `#1053` normalization to
`NdArrayOps::select` (or tightening the `as_slice_memory_order` guard in `try_unary_simd` to
`is_standard_layout`).

---

## Files changed

| File | Change |
| --- | --- |
| `crates/skoro-linalg/src/tril/diag.rs` | The fix: normalize `select`'s output to standard layout in `diagonal`, with a comment recording the mechanism |
| `crates/skoro-linalg/tests/common/tril.rs` | New `diag_survives_abs_impl` regression test |
| `crates/skoro-linalg/tests/common/mod.rs` | Wire `diag_survives_abs` into the three-backend `tests!` macro |
| `crates/skoro-linalg/Cargo.toml` | Add `skoro-pointwise` dev-dependency for the test |
| `Cargo.lock` | Records the new dev-dependency edge |

Scratch repro file created for Steps 1/4 was deleted (Step 5). No `#[ignore]` was removed.
