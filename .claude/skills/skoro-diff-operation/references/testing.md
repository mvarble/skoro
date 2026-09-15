# Verifying a `DiffOperation` numerically

Hand-computed expected constants are near-worthless for derivatives: they are almost always
generated _from the implementation_, so they lock in whatever the implementation already did. Every
real bug in this workspace survived a suite of such constants. Compare against finite differences
instead.

## The harness

`crates/skoro-model/src/test_utils.rs`, behind the `testing` feature. Add to the crate's
`[dev-dependencies]`:

```toml
skoro-model = { workspace = true, features = ["testing"] }
```

Four entry points, all with signature `(.., point, delta, epsilon)` where `delta` is the
finite-difference **step** and `epsilon` the **tolerance**:

| Function                                                   | Checks                                                                                                |
| ---------------------------------------------------------- | ----------------------------------------------------------------------------------------------------- |
| `test_jvp_numerically(inqty, outqty, ..)`                  | `Rinv(f(x), f(R(x, δeᵢ)))/δ` against the JVP column                                                   |
| `test_vjp_numerically(inqty, outqty, ..)`                  | the same difference against the VJP row                                                               |
| `test_hvp_numerically(inqty, outqty, ..)`                  | `(vjp(x ⊕ δeᵢ) − vjp(x))/δ` against the HVP — **diagonal block only**                                 |
| `test_hvp_cross_numerically(seedqty, readqty, outqty, ..)` | the same, but seeds/perturbs `seedqty` and reads `readqty`'s order-2 cotangent — the **mixed** blocks |

## Why the cross test is not optional

`test_hvp_numerically` seeds the tangent on, and reads the HVP from, the _same_ quantity. That only
ever probes `∂²f/∂x²`. For a two-input op whose VJP happens not to depend on one of its inputs, the
entire diagonal HVP block is identically zero and the test passes against a completely wrong
implementation.

This is not hypothetical: the SE(3) retraction's VJP depends only on `(ω, v)` and never on the pose,
so `test_hvp_numerically` on either input was vacuously satisfied. Deleting a whole HVP term changed
nothing. Only `test_hvp_cross_numerically` caught it.

**Rule: for an `n`-input op, cover all `n²` blocks.**

```rust
test_jvp_numerically(&a, &out, point.clone(), delta, epsilon);
test_vjp_numerically(&a, &out, point.clone(), delta, epsilon);
test_hvp_numerically(&a, &out, point.clone(), delta, epsilon);
test_jvp_numerically(&b, &out, point.clone(), delta, epsilon);
test_vjp_numerically(&b, &out, point.clone(), delta, epsilon);
test_hvp_numerically(&b, &out, point.clone(), delta, epsilon);
test_hvp_cross_numerically(&a, &b, &out, point.clone(), delta, epsilon);
test_hvp_cross_numerically(&b, &a, &out, point.clone(), delta, epsilon);
```

## Choosing fixtures

- **Far from the identity.** Rotations ~1 rad apart. Near `I`, `E ≈ Eᵀ`, `Δr ≈ Δrᵀ` and
  `J⁻¹(ω) ≈ I`, so transposes and missing Jacobian factors all vanish into the tolerance.
- **Asymmetric magnitudes.** A large translation next to a small rotation makes a missing
  lever-arm coupling (`a × v`) show up as an error of order `|v|` instead of order `ε`.
- **No zeros or repeated values** in the operands — they mask index and transpose errors.

Generate rotation-matrix fixtures with a scratch script and paste the literals in with a comment
recording the generating `ω`, e.g. `// Exp((0.3, -0.2, 0.5))`.

## Tolerances

On `f64` the four ops in this workspace agree to `1e-4` at `delta` of `1e-6`, `1e-7`, and `1e-8`.
Use `(1e-7, 1e-4)`. If you cannot reach that, suspect the implementation before loosening the
tolerance — and sweep several `delta` values, since a genuine error is roughly `delta`-independent
while truncation error scales with it.

On `f32`, `(1e-3, 1e-1)` is realistic.

## Cover every backend

Each op has independent `flex` and `cube` impls. A test on one compiles the other but never runs
it. Wire up `flex_f64`, `flex_f32`, and `wgpu_f32` entry points. The GPU path found a real
kernel bug (`left_broadcast_cross_product` ignoring the frame stride) that no CPU test could.

To localise a cube/flex disagreement, call `DiffOperation::forward` directly on both backends
with identical inputs and diff the flattened outputs mode by mode. That pinpoints which mode and
which block diverges far faster than reading kernels.

## Mutation-test the tests

A passing numerical test proves nothing until you have seen it fail. After a fix, re-break the
specific term and confirm the _expected_ test fails:

```bash
cp op.rs /tmp/op.bak
python3 - <<'EOF'
s = open('op.rs').read()
old = "<the corrected term>"
assert old in s
open('op.rs','w').write(s.replace(old, "<the buggy term>", 1))
EOF
cargo test -p <crate> 2>&1 | grep -E "^test .* FAILED"
cp /tmp/op.bak op.rs
```

Watch out when counting failures: `grep -cE "^test .* FAILED"` also matches the
`test result: FAILED.` summary line, inflating the count by one. Match `^test \w+ \.\.\. FAILED`
or just read the names.

Doing this caught that an early "fix" was untested: dropping a whole HVP term left the suite green,
which is what exposed the missing cross-block coverage.

## Framework-contract tests

Some invariants are not observable through returned cotangents. Cotangents wrongly computed for a
constant are discarded by the collection pass
(`crates/skoro-graph/src/compiler.rs:467` only surfaces `Parameter`/`Sampler` slots), so a missing
short-circuit costs work but produces identical output.

Test those with a spy `DiffOperation` that records what the framework handed it — see
`hvp_mode_reports_per_input_differentiability` in `crates/skoro-model/tests/hvp_diff_gating.rs`.
