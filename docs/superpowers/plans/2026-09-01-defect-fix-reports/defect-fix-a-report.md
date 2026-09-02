# Defect fix A — `TrilInvert::vjp` and `TrilInvert::hvp` missing transposes (and a sign)

**Status: DONE.** Both `vjp` and `hvp` were wrong. `hvp` needed **three** changes, not the one
transpose the brief anticipated.

---

## 1. The `vjp` fix

### Derivation

`f(A) = A⁻¹ = Y`, restricted to lower-triangular `A`. The JVP (already correct in `forward`) is the
linear map

```
dY = −Y dA Y
```

Pulling a Frobenius-inner-product cotangent `ω` back through that linear map:

```
⟨ω, dY⟩ = tr(ωᵀ · (−Y dA Y))
        = −tr(ωᵀ Y dA Y)
        = −tr(Y ωᵀ Y dA)          (cyclic permutation)
        = ⟨−Yᵀ ω Yᵀ, dA⟩
```

so `VJP(ω) = −Yᵀ ω Yᵀ`, i.e. the *conjugating matrix is `Yᵀ`, not `Y`*. The two coincide only when
`Y` is symmetric, which a non-diagonal triangle never is. `pack` then restricts to the lower
triangle, which is correct because `dA` is itself constrained to the lower triangle — the packed
coordinates are an isometry onto lower-triangular matrices with the Frobenius product, so no
scaling correction is needed.

### Applied change

`crates/skoro-linalg/src/tril/invert.rs`:

```rust
// before
let inv_mat: Tensor<B, 3> = unpack(inv_mat_packed.clone());
let omega_tril: Tensor<B, 4> = unpack(omega);
vec![Some(-pack(conjugate(inv_mat, omega_tril)))]

// after
let inv_mat_t: Tensor<B, 3> = unpack::<B, 2, 3>(inv_mat_packed.clone()).transpose();
let omega_tril: Tensor<B, 4> = unpack(omega);
vec![Some(-pack(conjugate(inv_mat_t, omega_tril)))]
```

(plus a derivation comment in the source).

---

## 2. `hvp` — investigated independently, and it was worse than expected

I did **not** assume the same one-line fix. I differentiated the *corrected* `vjp` expression a
second time and then checked the result against the current code term by term.

### Derivation

Let `Z = Yᵀ = (Aᵀ)⁻¹`. The corrected VJP is `G(A) = −Z ω Z` with `ω` held fixed. Then

```
dZ = (dY)ᵀ = (−Y ΔA Y)ᵀ = −Z ΔAᵀ Z
dG = −(dZ ω Z + Z ω dZ)
   = −((−Z ΔAᵀ Z) ω Z + Z ω (−Z ΔAᵀ Z))
   = (Z ΔAᵀ Z) ω Z + Z ω (Z ΔAᵀ Z)
```

Note there is no leftover minus: each of the two minus signs from `dZ` cancels the outer one.

### What the old code actually computed

Reading the old body with `P = Y ΔA Y`:

```
conj_omega = conjugate(Y, ω)   = Y ω Y
conj_dmat  = Y ΔA Y            = P
s0 = −(P ω Y)
s1 = −(Y ω Y · ΔA Y) = −(Y ω P)
result = s0 + s1 = −(P ω Y + Y ω P)
```

But `P ω Y + Y ω P` is exactly `d/dt[−Y ω Y]`, the derivative of the *old, wrong* `vjp`. So the old
`hvp` was **the negative of the derivative of the (already wrong) vjp**. It carried:

1. the wrong conjugator (`Y` instead of `Z = Yᵀ`), **and**
2. the tangent entering untransposed (`ΔA` instead of `ΔAᵀ`), **and**
3. an inverted overall sign.

The framework does not negate: `crates/skoro-operation/src/diff/step.rs:64-66` composes
`ord2 = hvp(α) + vjp(β)` with `β = 0` for a terminal all-ones seed, and I confirmed the old raw
`hvp` output reproduced the old hard-coded constant exactly (see §4), so the sign is genuinely in
the op, not in the plumbing. The `vector_product` fixture in the same file independently pins the
convention as `hvp = +d/dt vjp` with `β = 0` (`hd_L = ω Δvᵀ`, positive).

### Applied change

```rust
let inv_mat_t: Tensor<B, 3> = unpack::<B, 2, 3>(inv_mat_packed.clone()).transpose();  // Z
let omega_mat: Tensor<B, 4> = unpack(omega);
let conj_omega = conjugate(inv_mat_t.clone(), omega_mat.clone());                     // Z ω Z

let dmat_t: Tensor<B, 3> = unpack::<B, 2, 3>(dmat_packed.clone()).transpose();        // ΔAᵀ
let conj_dmat = inv_mat_t.clone().matmul(dmat_t.clone()).matmul(inv_mat_t.clone());   // Z ΔAᵀ Z

let s0 = right_broadcast_matmul(                                                     // (Z ΔAᵀ Z) ω Z
    left_broadcast_matmul(conj_dmat, omega_mat.clone()),
    inv_mat_t.clone(),
);
let s1 = right_broadcast_matmul(conj_omega, dmat_t.matmul(inv_mat_t));               // Z ω (Z ΔAᵀ Z)
vec![Some(pack(s0 + s1))]
```

Structure is unchanged; conjugator, tangent transpose, and sign all changed.

---

## 3. Step 4 — finite-difference verification (before touching the fixture)

Scratch file `crates/skoro-linalg/tests/scratch_invert_fd.rs` (ndarray-f64 only, per the
"one backend per test executable" rule), since deleted.

### 3a. `Tril(3)` against the FD harness

Fixture chosen per `references/testing.md` — far from the identity, no zeros, no repeats:

```
packed [1.3, −2.1, 0.7, 0.4, −1.6, 2.9]
  =  [[ 1.3,  0  ,  0  ],
      [−2.1,  0.7,  0  ],
      [ 0.4, −1.6,  2.9]]
```

Ran all three harness entry points at `delta = 1e-7`, `epsilon = 1e-4`:

```
test_jvp_numerically(&l, &linv, point, 1e-7, 1e-4)   ok
test_vjp_numerically(&l, &linv, point, 1e-7, 1e-4)   ok   <- anchors vjp to the primal
test_hvp_numerically(&l, &linv, point, 1e-7, 1e-4)   ok   <- ord2_cotangents path
```

`test_vjp_numerically` seeds an identity cotangent frame on `linv` and compares each row of the
VJP against `(f(x ⊕ δeᵢ) − f(x))/δ`, so it covers the full 6×6 packed Jacobian, not just a
contraction. `test_hvp_numerically` compares `ord2_cotangents` against
`(vjp(x ⊕ δeᵢ) − vjp(x))/δ` for every `i` — the `ord1_cotangents`/`ord2_cotangents` path the shared
fixture exercises. Single-input op, so there are no mixed blocks and no
`test_hvp_cross_numerically` to run.

```
running 2 tests
test print_tril2_fixture_values ... ok
test invert_fd_tril3 ... ok
test result: ok. 2 passed; 0 failed
```

### 3b. Mutation testing — the FD test was seen failing

Per `references/testing.md`, a green numerical test proves nothing until it has been seen red. I
re-broke each of the three corrections *separately* and re-ran:

| Mutation                                       | `invert_fd_tril3` |
| ---------------------------------------------- | ----------------- |
| M1 — `vjp` transpose removed                    | **FAILED**        |
| M2 — `hvp` transposes removed (sign kept `+`)   | **FAILED**        |
| M3 — `hvp` sign flipped back to `−` (transposes kept) | **FAILED**  |
| restored                                        | ok                |

M3 is the important one: flipping only the sign back leaves the FD test red, which is independent
confirmation that the sign flip is a real correction and not a compensating pair.

---

## 4. Step 5 — new expected values in `tests/common/tril.rs`

Fixture: `l` packed `[1, 2, 3]` → `L = [[1,0],[2,3]]`, `Y = L⁻¹ = [[1,0],[−2/3,1/3]]`, tangent
`Δl = [0.2, 0.1, 0.4]`.

**Seeding convention (read, not assumed).** `evaluate_ord1`/`evaluate_ord2` seed an all-ones
cotangent on the terminal node, i.e. `ω = unpack([1,1,1]) = [[1,0],[1,1]]`, with `β = 0`. Confirmed
two ways: (a) `−Y ω Y` with that `ω` reproduces the *old* constant `[−1, 5/9, −1/9]` exactly, which
would not happen for an identity frame (a 3-element assert rules that out anyway); (b) the
`vector_product` fixture in the same file is consistent with all-ones and `β = 0`.

With an all-ones cotangent, the whole thing reduces to a scalar objective — the sum of the packed
inverse:

```
S(l) = 1/l₀ − l₁/(l₀ l₂) + 1/l₂
```

### VJP / `ord1_cotangents`: `[−1/3, −1/3, 1/9]`

**Hand derivation, route 1 (scalar calculus):**

```
∂S/∂l₀ = −1/l₀² + l₁/(l₀² l₂) = −1 + 2/3 = −1/3
∂S/∂l₁ = −1/(l₀ l₂)           = −1/3
∂S/∂l₂ = l₁/(l₀ l₂²) − 1/l₂²  = 2/9 − 1/9 = 1/9
```

**Hand derivation, route 2 (matrix):** `Yᵀ = [[1,−2/3],[0,1/3]]`,
`Yᵀ ω Yᵀ = [[1/3, −4/9],[1/3, −1/9]]`, negated and packed → `[−1/3, −1/3, 1/9]`. Agrees.

**Numerically (central differences on `S`, h = 1e-5):**
`[−0.33333333336882, −0.33333333333552, 0.11111111111739]`.

**Graph output after the fix:** `[−0.33333333333333337, −0.3333333333333333, 0.1111111111111111]`.

The old constant `[−1, 5/9, −1/9]` is `pack(−Y ω Y)` — the transposed Jacobian.

### HVP / `ord2_cotangents`: `[7/90, 1/9, −17/270]`

**Hand derivation, route 1 (Hessian of `S` at `l = (1,2,3)`):**

```
        [  2/3   1/3  −2/9  ]
  H  =  [  1/3   0     1/9  ]
        [ −2/9   1/9  −2/27 ]

H · Δl,  Δl = (1/5, 1/10, 2/5):
  row0 = 2/15 + 1/30 − 4/45   = (12 + 3 − 8)/90   =  7/90
  row1 = 1/15 + 2/45          = (3 + 2)/45        =  1/9
  row2 = −2/45 + 1/90 − 4/135 = (−12 + 3 − 8)/270 = −17/270
```

**Hand derivation, route 2 (matrix, from §2's formula):** with `Z = Yᵀ`,
`Z Δlᵀ Z = [[1/5, −17/90],[0, 2/45]]`,

```
(Z Δlᵀ Z) ω Z = [[1/90, −19/270], [2/45, −2/135]]
Z ω (Z Δlᵀ Z) = [[1/15, −5/54  ], [1/15, −13/270]]
sum           = [[7/90,  ...   ], [1/9,  −17/270]]
packed        = [7/90, 1/9, −17/270]
```

Agrees entry for entry with the Hessian route.

**Numerically (central differences of the analytic gradient along `Δl`, h = 1e-5):**
`[0.07777777776830, 0.11111111111184, −0.06296296296421]`.

**Graph output after the fix:**
`[0.0777777777777778, 0.1111111111111111, −0.06296296296296296]`
= `[7/90, 1/9, −17/270]` to machine precision.

The old constant `[−0.4, 41/135, −4/135]` is `−(P ω Y + Y ω P)`, exactly what §2 says the old code
computed — which is how I confirmed I had read the old implementation correctly rather than
guessed at it.

### Edits made

`crates/skoro-linalg/tests/common/tril.rs`, `invert_impl`:

| assertion                        | old                        | new                        |
| -------------------------------- | -------------------------- | -------------------------- |
| `vjp.cotangents[l]`              | `[-1.0, 5/9, -1/9]`        | `[-1/3, -1/3, 1/9]`        |
| `hvp.ord1_cotangents[l]`         | `[-1.0, 5/9, -1/9]`        | `[-1/3, -1/3, 1/9]`        |
| `hvp.ord2_cotangents[l]`         | `[-0.4, 41/135, -4/135]`   | `[7/90, 1/9, -17/270]`     |

Each is now accompanied by an in-file comment recording the derivation, so the constants are
auditable without re-deriving from scratch. Tolerance left at `1e-6`; **not loosened** — all three
backends pass at it, including `wgpu_f32`.

The `forward`, `JVP`, and `hvp.tangents` assertions were untouched and still pass — the forward and
forward-mode paths were never wrong.

---

## 5. Step 7 — full test output

### `cargo test -p skoro-linalg` — all three backends

```
Running tests/ndarray_f32.rs
running 12 tests
test tril::inverse_retraction_round_trip ... ok
test vector::cross_product ... ok
test tril::invert_agrees_with_dense ... ok
test vector::sum ... ok
test vector::norm_squared ... ok
test tril::diag ... ok
test vector::norm ... ok
test matrix::right_broadcast_matmul ... ok
test matrix::left_broadcast_matmul ... ok
test tril::vector_product ... ok
test tril::invert ... ok
test matrix::matmul ... ok
test result: ok. 12 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.16s

Running tests/ndarray_f64.rs
running 12 tests
... test tril::invert ... ok ...
test result: ok. 12 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.09s

Running tests/wgpu_f32.rs
running 12 tests
... test tril::invert ... ok ...
test result: ok. 12 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 22.80s

Doc-tests skoro_linalg
test result: ok. 0 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out
```

`tril::invert` — the `invert_impl` fixture — passes under **ndarray-f32, ndarray-f64, and
wgpu-f32**.

### `cargo test --workspace`

All green. 80 test binaries; `grep -c "^test result: FAILED"` over the full output → **0**.
No downstream test changed behaviour: nothing outside `skoro-linalg` had pinned a reverse-mode
number through `Tril::invert` (see Concerns).

### `cargo clippy --workspace --all-targets -- -D warnings`

```
    Checking skoro-linalg v0.0.0
    Checking skoro-so3 v0.0.0
    Checking skoro-normal v0.0.0
    Checking skoro-se3 v0.0.0
    Checking skoro v0.0.0
    Finished `dev` profile [unoptimized + debuginfo] target(s) in 0.98s
```

Clean; no new warnings.

---

## 6. Files changed

- `crates/skoro-linalg/src/tril/invert.rs` — the `vjp` transpose fix and the three-part `hvp` fix,
  each with a derivation comment.
- `crates/skoro-linalg/tests/common/tril.rs` — three rewritten expected-value assertions in
  `invert_impl`, each with its derivation recorded inline.
- `crates/skoro-normal/tests/information.rs` — the `#[ignore]` string on
  `covariance_information_matches_the_score_covariance` no longer names the (now fixed)
  `TrilInvert::vjp` defect; the test stays ignored on the `burn-ndarray abs()` bug alone.
  See Concern 5.

Created and then deleted per Step 6: `crates/skoro-linalg/tests/scratch_invert_fd.rs`.

---

## 7. Concerns

1. **`hvp` needed three changes, not one.** The brief framed the question as "same transpose bug, or
   already correct". Neither: it also had an inverted sign. A reader skimming the commit will
   assume a transpose was applied uniformly; §2 is the record that it was not. The sign was
   verified three independent ways — the derivation, the FD harness, and mutation M3.

2. **Step 6 leaves the fixture back on hard-coded constants — the exact configuration that hid this
   bug.** `crates/skoro-linalg/tests/common/tril.rs` has *no* FD coverage of the tril ops; only
   `common/matrix.rs` and `linear_map.rs` use the harness. `testing.md` is explicit that "every real
   bug in this workspace survived a suite of such constants", and this is a fourth instance.
   I followed the brief and deleted the scratch file, and I mitigated by recording the derivation
   for each constant inline — but **I recommend, as separate follow-up work, adding
   `test_jvp_numerically`/`test_vjp_numerically`/`test_hvp_numerically` calls on a `Tril(3)` fixture
   to `invert_impl`** (and to `vector_product_impl`/`diag_impl`), which would have caught this on
   the day it was written. I did not do this unilaterally, as it is outside the brief's scope.

3. **Nothing downstream was covering this.** `Normal::factor` (`crates/skoro-normal/src/distribution.rs:208`)
   and `whiten` (line 171) both differentiate through `Tril::invert`, and `skoro-normal` has seven
   test files — yet the full workspace suite did not move by a single assertion under a change that
   inverts the sign *and* transposes every second-order cotangent through the op. That is a coverage
   gap, not a reassurance: every reverse-mode gradient any estimator has ever taken through a
   covariance factor was the transpose of the truth, and every HVP through it had the wrong sign
   too. Any tuned step sizes, convergence thresholds, or recorded convergence behaviour from before
   this commit should be treated as unverified.

4. **Blast radius on the in-progress Fisher-roots work.** The Fisher information root plan builds on
   `Normal::factor`, which is on the `invert()` path. Second-order quantities computed against the
   old behaviour differ by more than a transpose — they differ by sign — so any Fisher/Hessian
   numbers already recorded in that plan's briefs or reports are stale.

5. **The test that named this bug is still `#[ignore]`d, and I left it that way.**
   `crates/skoro-normal/tests/information.rs:209`,
   `covariance_information_matches_the_score_covariance`, carries
   `#[ignore = "blocked on TrilInvert::vjp missing a transpose ... and a burn-ndarray abs()
   strided-tensor ordering bug at >=32 elements"]`. I ran it explicitly with `--ignored` after the
   fix: it **still fails**, now at `information.rs:259` with `left = 3.1250000000000004`,
   `right = 3.811676712072403` (tolerance `0.05 · max(|g|, 1)`). That is the *second* blocker in its
   own ignore reason — the `burn-ndarray abs()` bug — which is outside this task's scope, so
   un-ignoring it here would just turn a documented skip into a red suite.

   The attribution is confirmed by `progress.md`, which the attribute cites: lines 78-79 record the
   two blockers as **independent** defects (Defect 1 = this transpose, Defect 2 = the `abs()` bug,
   with its own minimal repro in `task-23-report.md`), and line 93 states the plan's own intent —
   "**Then**: un-ignore `covariance_information_matches_the_score_covariance` ... (not just that the
   two defects are individually fixed)" — i.e. un-ignoring was always gated on Defect fix **B** as
   well, which is a separate task not yet done. Nothing in `progress.md` expected the transpose fix
   alone to green this test.

   I did edit the ignore string (`crates/skoro-normal/tests/information.rs:209`) to drop the
   now-stale `TrilInvert::vjp` clause, so the next reader is not sent chasing a fixed defect; the
   test remains `#[ignore]`d on the `abs()` bug alone.

6. **`wgpu_f32` at `epsilon = 1e-6`.** The new HVP constants (`≈0.078`, `0.111`, `−0.063`) are
   comparable in magnitude to the old ones, and the GPU backend passes at the unchanged tolerance,
   so no loosening was needed. Flagging only because `1e-6` is ~1e-5 relative on values that size,
   which is tight for f32 — if this fixture ever becomes flaky on a different GPU, tolerance is the
   first thing to look at, not the math.
