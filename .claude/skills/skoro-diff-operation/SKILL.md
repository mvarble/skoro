---
name: skoro-diff-operation
description: How to write, review, or debug a `DiffOperation` implementation in the skoro workspace — the forward/vjp/hvp contract, the two independent gates in `DiffModeState::Hvp`, tensor shape conventions, and how to verify derivatives numerically. Load this before adding a new op, editing an existing `forward`/`vjp`/`hvp`, or diagnosing wrong gradients, dropped cotangents, `unreachable!()` panics, or shape errors in an autodiff pass.
---

# Authoring `DiffOperation` implementations

`DiffOperation` (`crates/skoro-operation/src/diff/base.rs:123`) is skoro's stateless mathematical
description of one operation. It knows nothing about the graph; the graph hands it inputs and a
mode, and it returns a value plus whatever it must memoize for the backward passes.

You do **not** need to understand skoro's manifold machinery to write one correctly. You need the
contract below. If your op is manifold-valued (its input or output point is a rotation, a pose,
etc.), also read `references/manifold-conventions.md` — that is where the chart and Jacobian
conventions live.

## Shapes

Aliases are in `crates/skoro-backend/src/lib.rs:70-105`.

| Alias                                     | Rank | Shape                                                                        |
| ----------------------------------------- | ---- | ---------------------------------------------------------------------------- |
| `Point<B>`                                | 2    | `[batch, storage]` — the packed representation (e.g. a rotation is 9 floats) |
| `TangentVector<B>` / `CotangentVector<B>` | 2    | `[batch, dim]` — **one** vector per batch                                    |
| `TangentFrame<B>` / `CotangentFrame<B>`   | 3    | `[batch, dim, frame]` — a _stack_ of vectors                                 |

`storage` and `dim` differ whenever the quantity is manifold-valued: `SO3` is `storage = 9`,
`dim = 3`; `SE3` is `storage = 12`, `dim = 6`.

For rank-3 frames, **axis 1 is the component axis and axis 2 indexes the columns of the frame**.
So `slice_dim(1, 0..3)` takes the first three components of every frame column, and
`slice_dim(2, i)` takes the `i`-th column.

A frame is how skoro propagates many gradient signals at once. Most of the time you get it for
free: a frame-shaped `matmul` against a `[b, n, n]` operator does the right thing on all columns
simultaneously. When you need a per-column cross product against a fixed `[b, 3]` vector, use
`left_broadcast_cross_product(u, alpha)` = `u × alpha_col`
(`crates/skoro-linalg/src/vector/cross_product/flex.rs`), **not** `cross_product`, which is
`[b,3] × [b,3]` (line 127).

## The four modes

`forward` receives a `DiffModeState` and **must** return the matching `DiffOperationForward`
variant; the caller panics otherwise (`crates/skoro-operation/src/diff/operation.rs:115-186`).
Always compute the primal eagerly — every variant carries it.

| `DiffModeState`                  | You are given                              | You must return                                                   |
| -------------------------------- | ------------------------------------------ | ----------------------------------------------------------------- |
| `ForwardOnly`                    | nothing                                    | `output`                                                          |
| `Jvp(Vec<Option<TangentFrame>>)` | input tangent **frames** (rank 3)          | `output`, pushed-forward tangent frame                            |
| `Vjp(Vec<bool>)`                 | per-input differentiability                | `output`, `VjpState`                                              |
| `Hvp { diffs, tangents }`        | **both** (tangents are rank-2 **vectors**) | `output`, pushed-forward tangent _vector_, `VjpState`, `HvpState` |

Note the rank change: `Jvp` propagates a frame, `Hvp` propagates a single vector. Ops that share
code between the two typically `unsqueeze_dim(2)` / `squeeze_dim(2)` around a frame-shaped helper.

`vjp` and `hvp` always operate on **cotangent frames** (rank 3).

## The single most important rule: `Hvp` carries two independent gates

This is where essentially every bug in this workspace has lived. `DiffModeState::Hvp` carries
`diffs` _and_ `tangents`, and they mean different things:

- **`diffs[j]`** — does input `j` trace back to a free variable? Comes from
  `is_differentiable_parent` (`crates/skoro-graph/src/graph.rs:71`): constants and observed nodes
  are `false`. **This gates `VjpState`.**
- **`tangents[j]`** — was input `j` seeded with a tangent for _this_ HVP pass? **This gates
  `HvpState`.**

`tangents[j].is_some()` **implies** `diffs[j]`, because `get_tangents`
(`crates/skoro-operation/src/diff/operation.rs:63`) returns `None` for non-differentiable inputs.
**The converse is false**: a differentiable input with no seeded tangent arrives as `None` too,
and that is completely normal — an HVP is usually seeded on one variable at a time.

The HVP backward pass calls `vjp` as well as `hvp` (`crates/skoro-operation/src/diff/step.rs:64-66`),
which is why `VjpState` must still be built correctly during `Hvp` mode. Getting the gate wrong
breaks in opposite directions:

```rust
// WRONG — too tight. Silently drops the order-1 cotangent of any differentiable
// input that happened not to be seeded.
let vjp_state = (tangents[0].is_some().then(|| ...), ...);

// WRONG — too loose. `vjp` stops short-circuiting subtrees rooted only in constants,
// so work is done and tensors allocated for them.
let vjp_state = (Some(...), Some(...));

// RIGHT — VjpState on `diffs`, HvpState on `tangents`.
let [d0, d1] = diffs.try_into().unwrap();
let vjp_state = (d0.then(|| ...), d1.then(|| ...));
let hvp_state = [tangent0_derived, tangent1_derived];
```

The invariant to preserve: **if `∂f/∂xⱼ` is `None`, then `∂²f/∂y∂xⱼ` must be `None` too.** So gate
each `hvp` output slot on the corresponding `VjpState` slot, not merely on the tangents.

If `hvp` needs state that `VjpState` legitimately lacks (because that input is constant), put it in
`HvpState` — don't widen `VjpState`.

`crates/skoro-model/tests/hvp_diff_gating.rs` guards both directions with a spy op. Note the first
test asserts on the flags the framework delivers, not on returned cotangents: the collection passes
in `crates/skoro-graph/src/compiler.rs` — one at the end of `vjp_forward`, one at the end of
`hvp_forward` — surface only `Parameter`, `Given` and `Sampler` slots, so a cotangent wrongly
computed for a constant is discarded before it is observable. The cost of the "too loose" bug is
wasted work, not wrong output — it cannot be caught downstream.

## Designing `VjpState` / `HvpState`

- Prefer `Option<(A, B)>` **per input slot** over a flat tuple of parallel `Option`s plus a `bool`.
  It makes "this input is differentiable, here is everything its VJP needs" a single unrepresentable-
  when-absent fact, and it eliminates the exhaustive match arms that end in `unreachable!()`.
- **Never** use `Option<[T; 2]>` for a pair of per-input tangents. That shape cannot represent
  "both inputs differentiable, only one seeded" — a state that is reachable as soon as `VjpState`
  is gated on `diffs`. Use `[Option<T>; 2]`. `Matmul` hit `unreachable!()` this way and
  `CrossProduct` silently returned `[None, None]` when it owed a non-zero answer.
- Build outputs by composing `Option`s (`.map` / `.zip` / `.and` / `sum_opt_tensors`) rather than
  matching every combination. Exhaustive tuple matches over `(Option, Option, bool)` are how the
  `unreachable!()` arms got there.
- Tensor clones are refcounted — duplicating a shared operand into both state slots is cheap and
  usually clearer than reaching across slots.

## Leibniz is not your job

`hvp` computes only `∇_v dF*_p`. The tape applies the product rule itself
(`crates/skoro-operation/src/diff/step.rs:64-66`):

```
ord2_out = hvp(vjp_state, hvp_state, α) + vjp(vjp_state, β)
```

Do not add a `vjp(β)` term inside `hvp`.

A useful consequence when deriving: **`hvp` is the derivative of the `vjp` expression with respect
to the base point, along the seeded tangent, with the cotangent frame held fixed.** If your `vjp`
formula does not mention an input, that input contributes nothing to the HVP. This is worth
checking first — for the SE(3) retraction it collapses the whole derivation, because the VJP
depends only on the tangent argument and never on the pose.

## Checklist

1. Returned `DiffOperationForward` variant matches the input mode.
2. `VjpState` gated on `diffs` — never on `tangents.is_some()`, never ungated.
3. `HvpState` gated on `tangents`.
4. Every `hvp` output slot gated on its `VjpState` slot.
5. No `unreachable!()` left in `vjp`/`hvp`.
6. Paired tangent state is `[Option<T>; N]`, not `Option<[T; N]>`.
7. Blocks of a composite output are `cat`'d along axis 1, not summed. (`sum_opt_tensors` combines
   _contributions to the same block_; `Tensor::cat` assembles _different blocks_. Confusing them
   gives a shape error at best and a silently wrong answer at worst.)
8. Rank-3 frame paths and rank-2 vector paths both handled.
9. Both the `flex` and `cube` impls updated — they are separate `impl` blocks and drift silently,
   since a test on one backend compiles but never exercises the other.
10. Verified numerically — see `references/testing.md`.

## Known traps, all found the hard way

- The `flex` and `cube` impls of the same op are independent and _did_ diverge: the SE(3)
  retraction's cube branch `cat`'d its output blocks correctly while the CPU branch (then on ndarray) summed them.
- A near-identity fixture hides transposes. `E` vs `Eᵀ` and `Δr` vs `Δrᵀ` are both invisible when
  the rotation is close to `I`, and `J⁻¹(ω) ≈ I` hides a missing Jacobian factor. Use fixtures far
  from the identity (~1 rad).
- A test that seeds and reads the _same_ quantity only ever exercises diagonal Hessian blocks. Mixed
  blocks need `test_hvp_cross_numerically`.
- Cube kernels that write a frame must apply the frame stride on **every** read, not just the write.
  `left_broadcast_cross_product` omitted `f * v_sf` on its `v` reads, making every output column a
  copy of column 0 — invisible on CPU, wrong on GPU.
