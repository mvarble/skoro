# Phase 1 — remaining record

**Part B is complete.** `skoro-dense`, `skoro-model/structure.rs`, `skoro-normal/structure.rs` plus
the facade cache, and `skoro-levenberg-marquardt` have all landed, and every Part D item below is
covered. 197 tests pass across the workspace, on `ndarray_f64`, `ndarray_f32` and `wgpu_f32`.

Item 1 needed extending rather than inheriting: the round-trip in `skoro-graph`'s own suite reaches
only `Real` and `EuclideanD`, since that crate cannot see the group manifolds, so the mixed
`Real`/`EuclideanD`/`Tril`/`SO3`/`SE3` case now lives in `skoro-se3/tests/common/mod.rs`.

Items 2, 5 and 9 — the ones flagged as failing silently — were **mutation-tested**: removing
`swap_dims(1, 2)` from `jacobian_vjp` fails four tests, dropping the residual-vector negation in
`step` fails exactly the undamped ones, and dropping the recorded residual handle fails the
`cleanup` regression. Each gate is discriminating rather than merely green.

Three things were built differently from the interfaces this document specified. All three are now
recorded in `model-structure-bounds.md` §6 and §8.2, which is the durable record:

- `ModelClientOps` gained **`tangent_layout(&[Id])`**, and `NlsStructure::residual_layout` is a
  default body over it. Without it a facade holder could name the rows of `J` and not the columns.
- **`add_gaussian_quantity` takes `&mut self`**, and `GaussianModelClient` is no longer `Clone`.
  Recording mutates the cache, and two clones sharing one graph would each certify half the model.
- **`GaussianElem` never existed.** The bound is restated longhand, as everywhere else.

What follows is the verification record and the choices deliberately not taken.

---

# Part C — What writing the interfaces changed

The exercise earned its keep here. One design in `model-structure-bounds.md` does not survive contact
with the crate graph.

### The sealed-trait design cannot work across crates — ✅ **folded into the docs**

`model-structure-bounds.md` used to specify `NlsStructure` as a **sealed** trait, on the argument
that a trait anyone can implement cannot carry an invariant. The argument was right and the mechanism
was not: sealing works by a private `Sealed` supertrait, so only the defining crate can implement the
trait — and `NlsStructure` must live in `skoro-model` while `GaussianModelClient` lives in
`skoro-normal`. Sealing would have forbidden exactly the implementation the trait exists to admit.

The fix moves the guarantee from the trait to `ResidualCertificate`'s constructor: `record` demands
`D: WhitenedResidual<B>`, so a hand-rolled implementor can only produce an empty or honest
certificate. The trait is forgeable; the evidence is not.

> **This is now recorded in `model-structure-bounds.md` §2**, with the naming convention in §3 and the
> verification in §9 rebuilt around it. That document is the durable record; this section is a
> pointer, kept only because the finding originated here.

### Two smaller notes

- **`GaussianStructure` ships without a consumer.** Phase 1 has no use for `blocks()`; only the EKF
  does. It is declared now because the refinement `GaussianStructure: NlsStructure` is the thing
  being validated, and declaring it later would mean revisiting the facade a second time.
- **Encapsulating `Graph` turned out to be the load-bearing change, not `Q.client`.** The interfaces
  in B.3 and B.5 assert that a certificate cannot be bypassed. Writing them made clear that `Graph`'s
  five `pub` fields plus a `pub` `ModelServer.graph` made every such assertion provisional — a
  `&mut Graph` reached through `get_server` could append to `log_likelihoods` directly. Privatizing
  `Q.client` closes the route from a _quantity_; encapsulating `Graph` closes the route from a
  _client_, and without it the rest is a convention. This only became visible once the certificate
  had a written signature to defend. Both are now applied.

---

# Part D — Verification

Per-backend entry points over a shared `common/`, following the existing layout exactly. Each entry
point is three lines and the bodies are generic over `B`:

```rust
// crates/skoro-levenberg-marquardt/tests/ndarray_f64.rs — complete file
mod common;
type B = skoro_backend::NdArrayBackend<f64, i32>;
tests!(crate::B, 1.0e-6, 1.0e-5);
```

Heed the workspace doctrine (`.claude/skills/skoro-diff-operation/references/testing.md`):
hand-computed constants generated from the implementation only lock in whatever the implementation
already did. Prefer finite differences and closed forms.

Items 2 and 5 are the gates — the two that fail silently otherwise. Write them first.

| #   | test                                                                                                                                                                                                                                                                                                                                                                 | from                               |
| --- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------- |
| 1   | **Layout round-trip.** `unflatten(flatten(map)) == map` for a mixed layout of `Real`, `EuclideanD`, `Tril`, `SO3`, `SE3`. Confirms dimension-not-storage: the `SO3` block is 3 wide, not 9.                                                                                                                                                                          | `flattened-tangent-spaces.md` §8.1 |
| 2   | **JVP/VJP agreement.** `jacobian_jvp` and `jacobian_vjp` produce identical `[b, D_out, D_in]` tensors on the same model. **This is the test that catches the transpose asymmetry**, and it must include a non-square case or it proves nothing.                                                                                                                      | §8.2                               |
| 3   | **Finite differences.** Both Jacobian paths against a central difference taken _through the retraction_ (`retract_flat` with `±ε eᵢ`), on an `SE3` model so the 6-versus-12 split is exercised.                                                                                                                                                                      | §8.3                               |
| 4   | **Dense linalg.** `spd_solve(A, A) ≈ I`; `cholesky(L Lᵀ) ≈ L`; agreement with `TrilInvert` on a packed lower-triangular input converted to dense; `lstsq` on an ill-conditioned `J` beating `spd_solve(JᵀJ, ..)`, which is the QR argument made checkable. Plus the batch-1 guard: a `[2, m, n]` tensor returns `BatchNotSingleton`.                                 | `estimator-linear-algebra.md` §9   |
| 5   | **The normalizer precondition, both directions.** A model with a free `Tril` covariance **in** the `free` layout is rejected with `NormalizerNotConstant`; the _same model_ with that covariance outside `free` is accepted. A naive "covariances must be constant" implementation passes the first half and fails the second.                                       | `model-structure-bounds.md` §9.2   |
| 6   | **Certificate honesty.** A `ResidualCertificate` built by hand is empty, and `record` cannot be called without a `WhitenedResidual` distribution — a `trybuild` compile-fail test.                                                                                                                                                                                   | Part C                             |
| 7   | **Residual and density agree.** `‖whitened_residual(x)‖²` reconstructs `Normal::log_density(x)` up to `constant + log_det`, on `Real`, `EuclideanD` and `SE3` means.                                                                                                                                                                                                 | `model-structure-bounds.md` §9.1   |
| 8   | **LM end-to-end.** A linear-Gaussian least-squares problem. At `λ = 0` the analytic solution is reached in **one** iteration — the undamped step is exact on a linear problem, so anything else is a bug in `J` or in the solve. With `λ > 0` it converges to the **same** point in several, which checks that damping rescales the step without moving the optimum. | `flattened-tangent-spaces.md` §8.6 |
| 9   | **Cotangent defaulting.** A model with a free parameter that does not reach the residuals produces a zero column in `J` rather than panicking — the regression for the `vjp_forward` zero-defaulting already applied in `compiler.rs`.                                                                                                                               | `flattened-tangent-spaces.md` §5   |

Run with `cargo test --workspace`. The wgpu suites additionally exercise the device path, and will
surface the `TrilInvert` host round-trip as a sync point.

---

## Summary of files touched

Part A is applied; only the remaining work is listed.

|     | file                                                | change                                                                           |     |
| --- | --------------------------------------------------- | -------------------------------------------------------------------------------- | --- |

### Breaking changes not taken

Recorded so the choice is visible rather than forgotten.

- **`Distribution::parent_ids()` stays positional and untyped** (`skoro-normal/src/distribution.rs:164`
  returns `&[mean.id, std_dev.id]`). A richer parent description — names, or roles — would be a real
  ergonomic gain, and `model-structure-bounds.md` §8.4 flags the weakness. It is not taken because
  `normalizer_parents` (B.3) answers the only question phase 1 actually asks, and the right shape for
  the general fix is not yet known: designing it now, from one consumer, would likely produce
  something a second consumer breaks.
- **`IdGenerator: Clone` still copies the counter** (`skoro-core/src/id.rs:11-15`), so two clones of a
  graph issue colliding ids forever. Removing the impl would forbid `Graph: Clone` — which
  `sequential-estimation.md` §7 wants for snapshot/restore, where counter preservation is a _feature_.
  The right resolution belongs to the phase that builds retirement rollback, not to this one.
- **`Q<B, T>`'s type parameter stays unbounded.** Bounding `T: QuantityType<B, K>` would catch
  nonsense instantiations at construction rather than at first use, but `K` cannot be inferred at the
  struct definition without a second parameter, which is the `Q<B, T, Fam>` shape
  `model-structure-bounds.md` §6 rejects for unrelated reasons.

