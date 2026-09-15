# Manifold-valued ops: charts, conventions, and how to derive the three passes

Read this when the input or output point of your op is a rotation, a pose, or anything else whose
`storage` size differs from its `dim`. Everything here has been verified against finite differences
on `f64`, `f32`, and wgpu.

## Tangent coordinates are defined by the retraction

A quantity type supplies `retract(x, δ)`. That map *is* the definition of what a tangent vector's
coordinates mean; there is no other convention in play.

| Type | `retract` | `dim` |
|---|---|---|
| Euclidean | `x + δ` | n |
| `SO3` | `r Exp(a)` — right/body frame | 3 |
| `SE3` | `(r Exp(a), p + r b)` for `δ = (a, b)` — both body-frame | 6 |

Both `SO3` and `SE3` use **right** (body-frame) charts. A tangent's translation block `b` is
expressed in the body frame, so it is `r b` that gets added to the world-frame position.

## The derivation recipe

For `y = f(x)`, with input tangent `a` and output tangent coordinates read in the chart *at `y₀`*:

1. Perturb: `x_t = retract(x, t·a)`.
2. Push through: `y_t = f(x_t)`.
3. **Solve** `y_t = retract(y₀, c(t))` for `c(t)`. This step is the whole game — it is where the
   `Eᵀ` factors come from, and skipping it is how you end up with `E` where you needed `Eᵀ`.
4. JVP = `c'(0)`.
5. VJP = the adjoint: from `⟨α, Jv⟩ = ⟨Jᵀα, v⟩`, read off the coefficient of each input tangent.
6. HVP = `d/dt` of the **VJP expression** evaluated along `x_t`, with the cotangent frame `α` held
   fixed in coordinates.

**Check step 6 first.** If your VJP formula does not mention an input, that input contributes
nothing to the HVP. For the SE(3) retraction the VJP depends only on `(ω, v)` and never on the pose,
which collapses the derivation and — importantly — means the diagonal HVP blocks are identically
zero, so only cross-block tests can validate it.

## `exp` / `log` primitive conventions — the asymmetry that causes bugs

`J(ω)` is the right Jacobian of `SO(3)`, defined by `Exp(ω + δ) = Exp(ω) Exp(J(ω)δ + o(δ))`.
Coefficients: `exp_jacobian_coeffs` in `crates/skoro-so3/src/ops/exp/flex.rs`, transcribed into
`ops/exp/cube.rs` for the GPU. `Log` has its own, unrelated pair in `log_jacobian_coeffs`
(`ops/log/{flex,cube}.rs`) — same idea, different Jacobian, different series.

All four branch between a truncated Taylor series and the closed trig forms at a cutoff that
**depends on the float width** — sharing one cutoff across `f32` and `f64` leaves `f32` evaluating
expressions that have cancelled away entirely. Both maps shipped that bug. Guarding a change here
takes two things: the unit tests beside each function (which see the cutoff directly) and the
frozen `f64` tables in `crates/skoro-so3/tests/common/coefficients.rs` (which see the hand-written
`cube` transcription). Finite-difference tests cannot see either.

Two spaces are in play: **ω-space** (the 3-vector coordinates of a `Log`) and **group-side**
(a body-frame perturbation of a rotation matrix). They are related by `J(ω)`.

| Function | First arg | Direction / cotangent arg | Returns |
|---|---|---|---|
| `exp_jvp(ω, v)` | ω | `v` in **ω-space** | group-side, `= J(ω)v` |
| `exp_vjp(ω, α)` | ω | `α` group-side cotangent | ω-space, `= J(ω)ᵀα` |
| `exp_hvp(ω, u, α)` | ω | **`u` in ω-space** | `∇_u[J(ω)ᵀα]` |
| `log_jvp(ω, dr)` | ω | `dr` **group-side** | ω-space, `= J⁻¹(ω)dr` |
| `log_vjp(ω, α)` | ω | `α` ω-space cotangent | group-side, `= J⁻ᵀ(ω)α` |
| `log_hvp(ω, u, α)` | ω | **`u` in ω-space** | `∇_u[J⁻ᵀ(ω)α]` |

> **The rule:** the direction argument of *both* `*_hvp` functions is `dω/dt` — the velocity of the
> ω that the function's own first argument holds. It is never a group-side quantity.

The asymmetry follows:

- For `exp`, `ω` is the **input**, so the seeded tangent is already `dω/dt`. Pass it straight
  through: `exp_hvp(ω, dω, α)`.
- For `log`, `ω` is the **output**, so a group-side velocity `h` must be converted first:
  `log_hvp(ω, log_jvp(ω, h), α)`. **Passing `h` directly is wrong.**

The reference implementations state the convention: `SO3Exp::forward` stores the raw tangent as its
`HvpState`, while `SO3Log::forward` stores `eta = log_jvp(ω, dr)`
(`crates/skoro-so3/src/ops/log/flex.rs`, `Hvp` branch).

This bug is invisible near `ω = 0`, where `J⁻¹(ω) ≈ I`. It shipped in both `SO3InverseRetraction`
and `SE3InverseRetraction` behind a fixture whose two attitudes were 0.01 rad apart.

## Identities worth having at hand

- `d/dt Exp(ω + t·dω) = Exp(ω)·[J(ω)dω]ₓ`
- For `Δr_t = Exp(−t·a₀) Δr Exp(t·a₁)`: `dΔr/dt = −[a₀]ₓΔr + Δr[a₁]ₓ`, and the **body-frame**
  perturbation is `h = a₁ − Δrᵀa₀`.
- `(Ru) × (Rw) = R(u × w)` for `R ∈ SO(3)`. This folds `Δrᵀ(a₀ × α) − a₁ × (Δrᵀα)` into the single
  cross product `−h × (Δrᵀα)`, and it is why a `Δrᵀ` appears where `dΔr/dt` had a `Δr`.
- `a × v = −v × a`. With `a` a rank-3 frame and `v` a rank-2 vector, `a_col × v` is
  `-left_broadcast_cross_product(v, a)`.
- Adjoint of `a ↦ a × v` is `u ↦ v × u`. Adjoint of `a ↦ q × a` is `u ↦ −q × u`.

## Reference formulas

All verified numerically. `E = Exp(ω)`, `J = J(ω)`, `vjp = J⁻ᵀ(ω)α`, `g = J(ω)dω`,
`h = a₁ − Δrᵀa₀`, `∇ₕvjp = log_hvp(ω, J⁻¹(ω)h, α)`.

### `SO3Retraction` — `R(r, ω) = r Exp(ω)`

```
JVP  c   = Eᵀa + J dω
VJP  ∂r  = E α                     ∂ω = Jᵀ α
HVP  ∂r  → E (g × α)               ∂ω → exp_hvp(ω, dω, α)
```

### `SO3InverseRetraction` — `Rinv(r₀, r₁) = Log(r₀ᵀ r₁)`

```
JVP  dω  = J⁻¹ h
VJP  ∂r₀ = −Δr vjp                 ∂r₁ = vjp
HVP  ∂r₀ → a₀ × (Δr vjp) − Δr(a₁ × vjp) − Δr ∇ₕvjp
     ∂r₁ → ∇ₕvjp
```

### `SE3Retraction` — `R((r,p),(ω,v)) = (r Exp(ω), p + r v)`

```
JVP  c = Eᵀa + J dω                d = Eᵀ(b + a × v + dv)
VJP  ∂(r,p) = (E α_r + v × (E α_p),  E α_p)
     ∂(ω,v) = (Jᵀ α_r,               E α_p)
HVP  ∂(r,p) → (E(g × α_r) + dv × (E α_p) + v × (E(g × α_p)),  E(g × α_p))
     ∂(ω,v) → (exp_hvp(ω, dω, α_r),                           E(g × α_p))
```

The `a × v` term in `d` is the lever arm: rotating `r` swings the offset `r v`. Both translation
blocks carry `Eᵀ`, not `E`, because `d` is read in the *output* frame `rE`. Dropping the lever arm
produces an error of order `|v|`, so a fixture with a large translation makes it unmissable.

### `SE3InverseRetraction` — `Rinv((r₀,p₀),(r₁,p₁)) = (Log(Δr), q)`, `q = r₀ᵀ(p₁−p₀)`

```
JVP  dω = J⁻¹ h                    dq = q × a₀ − b₀ + Δr b₁
VJP  ∂(r₀,p₀) = (−Δr vjp − q × α_q,  −α_q)
     ∂(r₁,p₁) = (vjp,                 Δrᵀ α_q)
HVP  ∂(r₀,p₀) → (a₀ × (Δr vjp) − Δr(a₁ × vjp) − Δr ∇ₕvjp
                  − (q × a₀) × α_q + b₀ × α_q − (Δr b₁) × α_q,   0)
     ∂(r₁,p₁) → (∇ₕvjp,                          −h × (Δrᵀ α_q))
```

Note the `Δrᵀ` in the final term and the `Δr` two lines above it — they are *not* the same factor,
and near-identity fixtures cannot tell them apart.
