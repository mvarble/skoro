# Model Structure Bounds

How an inference algorithm states the structure it requires of a model, where that claim is checked,
and why it can never be recovered by looking at the graph.

Fifth companion to `flattened-tangent-spaces.md`, `sequential-estimation.md`,
`graph-marginalization.md` and `estimator-linear-algebra.md`. Those own, respectively, the algebra,
the architecture, the surgery, and the numerics. This one owns **preconditions**: an EKF needs
Gaussian conditionals, Levenberg-Marquardt needs a nonlinear-least-squares objective, and something
has to carry those facts from where they are known to where they are needed.

## 1. Motivation

Three claims, in increasing strength:

| algorithm | needs |
|---|---|
| Levenberg-Marquardt | `−log p` is `½ Σ ‖rᵢ(x)‖²` — a sum of squared residuals |
| hierarchical Gaussian | additionally, each `rᵢ` is a whitening composed with a nonlinear mean |
| EKF | additionally, the model is a Markov chain of ticks (§7) |

The middle row is a *minor* generalization of the first, and saying exactly how minor is most of the
design: LM needs only that residual vectors exist; the Gaussian case needs the mean and the whitening
reachable *separately*, because `H = ∂μ/∂x` and `R = L Lᵀ` are different objects.

The concrete pressure is **how LM populates `J`**. Its rows are the residual nodes
(`flattened-tangent-spaces.md` §7), so something must name them. The tempting answer is to look:
walk down from each `LogDensity` and recognize the residual. §4.1 shows why that fails. The deeper
answer is that for distribution *family* it cannot work at all.

### Reflection is not merely brittle — for family identity it is impossible

`Distribution` is consumed **by value** in `ModelClient::add_quantity` (`model.rs:533-563`) and
dropped; what it contributed is expanded into ordinary graph operations. What remains is
`LogDensityNode` (`node.rs:131-137`), which records `op`, `parent_ids`, `stochastic_id` and `value` —
nothing identifying the distribution. `Node::quantity_type()` returns `None` for it (`node.rs:48`).

There is exactly one surviving trace: `ModelServer::add_stochastic` (`model.rs:98-105`) stores
`distr.sampler()` on the `StochasticNode`, so a `NormalSampler<B, M>` is still in memory, reachable
as `log_density_to_stochastic[density] → StochasticNode.sampler`. But it is stored as
`Arc<dyn GraphSampler<B>>`, and `GraphSampler` (`skoro-graph/src/sampler.rs:8-18`) exposes only
`lower` and `clone_box` — **no `as_any`**, unlike `DynQuantityType` (`quantity.rs:174-187`), which
has one and is downcast in six places. The concrete type is present and unreachable.

> **"Was this factor a `Normal`?" has no typed answer after the model is built.** Not an expensive
> one — none. Any design that needs the answer must arrange to know it *before* the graph exists.

That is not a defect to fix. Adding `as_any` to `GraphSampler` would make family identity
recoverable, and it would still be the wrong mechanism: LM does not care whether a factor is Gaussian,
only whether it has a residual, and "has a residual" is not a family.

## 2. Two kinds of structure claim

The organizing distinction, and everything below follows from it:

> A **factor-local** claim — *this factor's negative log-density is `½‖r‖²` plus a normalizer* — is a
> bound on the **`Distribution` type**, checked where the factor is built.
>
> A **model-global** claim — *every factor in this model is like that* — is the **accumulation** of
> factor-local claims, carried as data by the builder that enforced them.
>
> Neither is a predicate over the graph.

And the property that makes this pay rather than merely typecheck:

> **The certificate and the algorithm's input are the same object.** What a restricted builder
> accumulates is a list of residual nodes — which is exactly the `residuals` `TangentLayout` that
> `flattened-tangent-spaces.md` §7 already hands to LM. There is no separate artifact to keep in
> sync. `residual_layout()` returns the rows of `J`.

### Where trait bounds belong

Bounds are the right tool, in two places, for two different jobs. The distinction matters because
the workspace already has one of each and treats them as the same thing.

**A trait anyone can implement proves nothing about a model.** `GaussianModelClientOps`
(`skoro-normal/src/facade.rs:71`) is declared as the Gaussian capability and is **also implemented
for the unrestricted `ModelClient<B>`** at `:89`. So `C: GaussianModelClientOps<B>` says *can add
Gaussians*, never *only ever did*. `skoro-normal/tests/facade.rs:23` exists specifically to show it
accepts both clients — the openness is deliberate and correct for what the trait is.

**The repair is to guard the data, not the trait.** The obvious move is to *seal* the structure
trait, so only builders that enforced the bound can implement it — the workspace seals three traits
already (`Kind<B>`, `skoro-core/src/kind.rs:25-27`; `Backend`, `skoro-backend/src/lib.rs:17-37`;
`DynManifold`, `skoro-riemann/src/base.rs:32-49`). It does not work here, and the reason is
structural rather than incidental:

> **Sealing and cross-crate variance are mutually exclusive.** A sealed trait can only be
> implemented in the crate that defines it. `NlsStructure` must live in `skoro-model`, since it
> bounds `Distribution` and `ModelClientOps`; `GaussianModelClient` lives in `skoro-normal`. Sealing
> would forbid exactly the implementation the trait exists to admit.

So the trait stays open, and the guarantee moves into the **value it returns**:

```rust
pub struct ResidualCertificate { /* private */ }

impl ResidualCertificate {
    /// The `D` bound is the whole guarantee.
    pub fn record<B, D>(&mut self, distr: &D, density: Id, residual: &Q<B, EuclideanD>)
        -> SkoroResult<()>
    where B: Backend, D: WhitenedResidual<B>;
}
```

A hand-rolled implementor of `NlsStructure` must produce a `ResidualCertificate`, and the only
certificate it can produce is either **empty** or one populated through `record` — which demands a
`WhitenedResidual` distribution. There is no constructor that accepts a fabricated list of ids. The
trait is forgeable; the evidence is not, and the evidence is what the algorithm consumes.

This is the better design independent of the crate-boundary problem, because it puts the guarantee on
the object that crosses the boundary rather than on a marker sitting beside it — the same move as
"the certificate and the algorithm's input are the same object".

Trait refinement still states the implication directly:

```rust
pub trait GaussianStructure<B: Backend>: NlsStructure<B> { /* … */ }
```

**is** "hierarchical Gaussian ⇒ LM-capable". An algorithm bounded on `NlsStructure` accepts a
Gaussian model for free, and a third builder added later implements whichever trait it qualifies for
with no algorithm changes.

To summarize the two negatives and the two positives, since they are easy to conflate:

> Bounds on the **`Distribution`** establish factor-local structure, at the one moment it is
> knowable. Bounds on a **certificate's constructor** carry that structure onward, and the open trait
> over it gives variance. What does not work is an *open* bound on a client used as if it were a
> guarantee. What cannot work at all is a predicate over the built graph.

## 3. Naming: capability versus structure

The two roles must be distinguishable at a glance, and **half the convention already exists and is
perfectly consistent.** The workspace has thirteen `*Ops` traits — `ModelClientOps`
(`client_ops.rs:46`), `GaussianModelClientOps`, `TrilOps`, `MatmulOps`, `TransposeOps`,
`LinalgVectorOps`, `CrossProductOps`, `PointwiseOps`, and five `SO3*Ops` — and *every one* is a
capability trait, openly implemented (usually blanket over `Q<B, M>`), none sealed, none asserting
anything about a whole model. The rule below is mostly a matter of writing down what the codebase
already does, and giving the counterpart a suffix of its own.

| | capability | structure |
|---|---|---|
| answers | "can this client **add** X?" | "is **every** factor in this model X?" |
| suffix | **`…Ops`** | **`…Structure`** |
| guarantee lives in | nothing — it is a capability | the **returned certificate's constructor** |
| implemented by | anything that can perform the action | any builder — the evidence is what is guarded |
| methods | verbs — `add_gaussian_quantity` | nouns returning evidence — `certificate`, `blocks` |
| used during | **building** | **inference** |
| example | `GaussianModelClientOps` | `GaussianStructure` |

Both trait kinds are open. That is not a weakness in the `Structure` side — §2 shows sealing is
unavailable across a crate boundary anyway — so the rule cannot be "one is sealed". It has to be
about what the trait hands back:

> **An `Ops` trait returns whatever the action produced. A `Structure` trait returns a value whose
> constructor is bounded.** If a trait's methods return something anyone can build — a `bool`, a
> `Vec<Id>`, a count — it proves nothing and is an `Ops` trait no matter what it is called. The
> suffix is a claim about the *return type*, not about the trait.

That is mechanically checkable, which is the point: `ResidualCertificate` qualifies because `record`
demands `D: WhitenedResidual<B>` and no other constructor populates it; a `bool` does not qualify,
and neither would a `Vec<Id>` assembled by the implementor.

Two corollaries worth stating, because each names a mistake that is easy to make and invisible
afterwards:

- **An `Ops` trait's documentation must never claim a model-wide invariant.** This is a live defect
  (§8.1): `facade.rs:11-15` and `:196-199` describe `GaussianModelClient` as "the structural
  guarantee that every conditional in this model is Gaussian, used by inference algorithms". That is
  a `Structure` claim written next to an `Ops` trait. The trait is fine; the prose is the defect.
- **A `Structure` trait returns the cached artifact, not a verdict.** This is now the mechanical test
  above rather than a matter of taste: `fn is_gaussian(&self) -> bool` returns a value any
  implementor can fabricate, so it guards nothing and the trait carrying it is an `Ops` trait
  mislabelled. The methods must return the evidence itself, and the evidence's constructor is what
  makes it trustworthy.

So the three names around one concept, each with a distinct job:

| name | kind | says |
|---|---|---|
| `GaussianModelClient` | struct | builds a Gaussian model, and caches what it built |
| `GaussianModelClientOps` | capability | "this client allows you to add a conditional Gaussian" |
| `GaussianStructure` | structure | "this model looks like a hierarchical Gaussian" |

Both traits are wanted. Nothing is removed; they are told apart by their suffix.

## 4. The factor-local bounds

Adopt the name `flattened-tangent-spaces.md` §7 already chose — it proposes a `WhitenedResidual<B>`
trait for exactly this purpose, "so `skoro-levenberg-marquardt` is distribution-generic rather than
`Normal`-specific". This document supplies the rest of it.

```rust
/// A continuous distribution whose negative log-density is `½‖r(x)‖² + n(θ)`.
pub trait WhitenedResidual<B: Backend>: Distribution<B, Continuous> {
    /// The residual node `r(x)`. The sum of squares of these is the LM objective.
    fn whitened_residual(&self, x: &Q<B, Self::Qty>) -> SkoroResult<Q<B, EuclideanD>>;

    /// The subset of `parent_ids()` on which the normalizer `n(θ)` depends. See §5.
    fn normalizer_parents(&self) -> &[Id];
}

/// Additionally: `r(x) = L⁻¹ (x ⊖ μ)`, with the mean and the whitening separately reachable.
pub trait GaussianFactor<B: Backend>: WhitenedResidual<B> {
    fn mean(&self) -> &Q<B, Self::Qty>;
    fn whitening(&self) -> &Q<B, Tril>;
}
```

**This is the "minor generalization" of §1 made precise.** LM needs `WhitenedResidual` and nothing
more. The hierarchical-Gaussian case needs `GaussianFactor`, whose entire additional content is that
`μ` and `L` are reachable separately rather than only through their composition.

And the nonlinearity comes for free: `mean()` returns a `Q`, which may be the output of arbitrarily
many operations. The type does not have to express "nonlinear mean" because the graph already holds
it. That is why the refinement is two accessors rather than a new algebra.

Notes on the bounds:

- **Bound on `Distribution<B, Continuous>`, not on `Distribution<B, K>`.** A `Discrete` factor has no
  tangent space, `inverse_retract` is undefined for it, and no residual vector exists at all. The
  `Kind<B>` split (`skoro-core/src/kind.rs:26-60`) already draws this line; reuse it rather than
  discovering it at runtime.
- **The residual type is forced, not chosen.** For a generic mean manifold `M`, `inverse_retract`
  resolves only through the blanket `AssociatedTangentVectorBuilder<EuclideanD, B>`
  (`skoro-model/src/ops/base.rs:18`). So `whitened_residual` returns `Q<B, EuclideanD>` and needs no
  bound on `M` beyond the one `Normal` already carries.
- **Reuse `GaussianElem`.** The six `B::FloatElem` operator bounds these traits need are already
  bundled at `skoro-normal/src/facade.rs:46-65` — and never used, with every impl restating them
  longhand (§8.2). This is the chance to make the bundle earn its place.

### `Normal` implements both nearly for free

`Normal<B, M>` (`skoro-normal/src/distribution.rs:79-86`) already has `mean() -> &Q<B, M>` (`:131`)
and `std_dev() -> &Q<B, Tril>` (`:135`), so `GaussianFactor` is two forwarding methods.
`whitened_residual` is an extraction of three lines that already exist (`:168-178`):

```rust
let colored_tangent = self.mean.inverse_retract(x);
let std_dev_inv     = self.std_dev.invert();
let white_tangent   = std_dev_inv.vector_product(&colored_tangent)?;   // ← this node
```

`white_tangent` is a local binding today, consumed immediately by `.norm_squared()`; its `Id` is
never returned to anyone. Exposing it is the whole change.

> **One fidelity note.** The current line uses `.expect("internal: Normal covariance factor and
> tangent dimensions should match")` (`:173-175`), not `?`. Returning `SkoroResult` from
> `whitened_residual` is therefore a **change in error posture**, not a pure extraction. Keep
> `log_density` calling `whitened_residual` and re-applying the `expect`, so behaviour on the one
> path that cannot fail by construction is unchanged, and the fallible signature exists only for the
> new caller that can meaningfully handle it.

### 4.1 Rejected: recovering the residual by graph position

Worth refuting concretely, because it is what one reaches for when wiring up `J`: *the residual is
the `Mapped` node just below the `LogDensity`.*

It is not. `Normal::log_density` returns

```rust
self.constant + -0.5 * white_tangent.norm_squared() + log_det
```

so the node directly beneath the density is the **outermost `+` from operator sugar**. The residual
is three operations further down and hidden behind `norm_squared`, and the precise shape is whatever
`std::ops::Add` on `Q` happened to lower to. A rule keyed on that position is pinned to the sugar's
lowering, and it breaks the first time a distribution reassociates its own expression — silently,
producing a wrong `J` rather than an error.

The residual has to be captured when it is built, by the only party that still knows the distribution
type. That is the argument for §6.

## 5. The normalizer is the hole

**`Normal::log_density` is not `½‖r‖²`.** It is

```rust
constant + -0.5 * white_tangent.norm_squared() + log_det
```

where `log_det = std_dev.invert().diag().abs().log().sum()` (`distribution.rs:176`) is a **live graph
node depending on `std_dev`**. If that dependence reaches a variable being optimized, minimizing
`½ Σ ‖rᵢ‖²` minimizes the wrong function, silently and with no error anywhere.

This is not hypothetical. `Tril` is a `Manifold` and therefore a `ContinuousQuantity`, and
`GaussianModelClient` delegates both `add_parameter` and `demote_to_parameter` — so **a free
covariance factor is constructible through the Gaussian-only facade.** "Every factor is Gaussian" is
not sufficient for LM.

The precondition, stated so it is checkable without inspecting a single operation:

> **LM's objective is valid iff no `normalizer_parents()` id of any factor is a column of the `free`
> layout.**

Note it is *not* "the covariance must be constant". The correct and weaker condition is that the
normalizer does not vary with the variables being optimized — a covariance that is a `Parameter` but
is not in `free` is fine, and that is a real modelling case (a fixed but not compile-time-known
noise). The check is O(#factors × arity), needs no reflection, and produces a nameable error.

`normalizer_parents` also repairs a smaller defect for the one purpose that needs it.
`Distribution::parent_ids()` is **positional and untyped** — `Normal` returns `&[mean.id,
std_dev.id]` (`:122`, `:164`) — so there is no way to ask which parent is the covariance. Rather than
add a general "which parent is which" facility, the trait asks the question the algorithm actually
has. For `Normal` the implementation is `&self.ids[1..]`.

## 6. The certificate

A restricted builder records what it built, at the moment it built it — the only moment the
distribution type is still in hand. The certificate belongs on the **existing facade struct**, since
`GaussianModelClient` (`facade.rs:139-142`) is already the builder:

```rust
pub struct GaussianModelClient<B: Backend> {
    client: ModelClient<B>,
    factors: Vec<GaussianRecord>,   // { density, residual, mean, whitening, normalizer_parents }
}
```

`NlsModelClient<B>` is the weaker sibling, bounded on `WhitenedResidual` rather than
`GaussianFactor`, recording `{ density, residual, normalizer_parents }` only.

The traits of §2 sit over them. Both are **open** — the guarantee is in `ResidualCertificate`, which
`certificate()` hands back and which nothing outside `record` can populate:

```rust
pub trait NlsStructure<B: Backend>: ModelClientOps<B> {
    /// The evidence. Bounded constructor; see §2.
    fn certificate(&self) -> &ResidualCertificate;

    /// The rows of `J` — `flattened-tangent-spaces.md` §7's `residuals` layout.
    fn residual_layout(&self) -> SkoroResult<TangentLayout>;
    fn normalizer_parents(&self) -> impl Iterator<Item = Id> + '_;
}

pub trait GaussianStructure<B: Backend>: NlsStructure<B> {
    fn blocks(&self) -> &[GaussianBlock];   // { mean, whitening, residual }
}
```

`NlsModelClient: NlsStructure`. `GaussianModelClient: NlsStructure + GaussianStructure`. Then
`fn solve<M: NlsStructure<B>>(..)` is Levenberg-Marquardt, `fn step<M: GaussianStructure<B>>(..)` is
the EKF, and a Gaussian model is accepted by both without a cast, a check, or a walk.

> **`into_client()` dissolves the certificate.** `facade.rs:162` hands back the raw `ModelClient`, and
> a caller who takes it can add factors the cache will never see. It should either be removed or
> documented as exactly that — an exit from the guarantee, not a view onto it.

### Making it unforgeable — ✅ **done**

The certificate is only as good as the facade's exclusivity, and the facade used to leak four ways.
Three are now closed; the fourth is deliberate.

| leak | status |
|---|---|
| `Q<B, Real>::mark_likelihood` attaching a factor to any real-valued node | **removed**; the method is on `ModelBuilderOps`, which a facade may decline to implement |
| `Q.client` a `pub` field, reaching the unrestricted surface from any quantity | **private**; `Q.id` too |
| `ModelClient::get_server` handing out `Arc<RwLock<ModelServer>>` over a `pub graph` | **removed**; `Graph`'s five fields and `ModelServer.graph` are private behind read accessors, so `server_call` is now read-only in effect |
| `into_client()` (`facade.rs:162`) | **kept**, and documented as dissolving the certificate — an explicit exit, not a hole |

What made the difference was not the `Q` work but the **graph encapsulation**. Privatizing `Q.client`
closes the route from a *quantity*; making `Graph`'s fields and `ModelServer.graph` private closes the
route from a *client*, and without that second one a `&mut Graph` could append to `log_likelihoods`
directly and every certificate would be provisional.

> **Do not put a family tag on `Q`.** A naive reading of the `Kind<B>` precedent suggests
> `Q<B, T, Fam>`, so that quantities from a Gaussian builder cannot be re-widened. It does not work:
> means are built by ops — `μ(θ)` is a mapped node — so every operator would have to return an
> untagged quantity, and the tag would be gone exactly where it was needed. What needs restricting is
> the **capability**, not the quantity, which is what the closures above did.

### Lifecycle

The factor set is not write-once. Retirement installs joint marginal factors and drops old ones
(`graph-marginalization.md` §7, steps 4–5), so the certificate must be maintained, not merely built.

That has a clean owner: `sequential-estimation.md` §2 already establishes the **Timeline** as the
single writer of graph structure. It is also where the privileged `mark_likelihood` lives after
Tier 1 — the marginalization machinery sits *below* the facade rather than being a user of it, so
withholding the user-facing method costs it nothing.

## 7. What this does not cover

An EKF needs more than Gaussian factors. It needs the *shape* of the model: a designated state, a
process factor relating consecutive states, measurements attached to the current one. No bound on a
distribution expresses that, and no accumulation of factor-local claims implies it.

That structure comes from a different mechanism, already specified: the `Transition` / `TickCtx`
builder of `sequential-estimation.md` §5, which gives the Markov property *by construction* because a
tick's only inputs are the previous tick's state and this tick's exogenous data.

> **Factor-local structure is a bound on the `Distribution`. Model-shape structure is a property of
> the builder that assembled the ticks.** Two mechanisms, two kinds of claim, and conflating them is
> how one ends up asking the graph a question it cannot answer.

So a full EKF precondition is the conjunction: `M: GaussianStructure<B>` for the numerics, built
through a `Transition` for the shape. The first is checkable by the compiler; the second is
guaranteed by construction. Neither is checkable by inspection, and that is the point.

## 8. Defects to fix along the way

Following the sibling documents' idiom. Each is independently reachable and none should ship as-is.

1. **A `Structure` claim is attached to an `Ops` trait.** `facade.rs:11-15` and `:196-199` describe
   `GaussianModelClient` as the structural guarantee that every conditional is Gaussian, "used by
   inference algorithms (e.g. EKF / Levenberg–Marquardt) that need the nonlinear-least-squares
   structure". The trait `GaussianModelClientOps` is blanket-implemented for the general
   `ModelClient` (`:89`) and proves nothing; the type intends the invariant but leaks it four ways
   (§6). Per §3, **the trait is correct and the prose is the defect** — it should claim capability
   only, and the invariant should move to `GaussianStructure`.
2. **`GaussianElem` is dead.** `facade.rs:46-65` bundles the six `FloatElem` operator bounds and is
   never used as a bound anywhere; every impl restates them longhand. The traits of §4 need the same
   bounds — use it.
3. **`GaussianModelServer` is orphaned.** `facade.rs:113-130` is constructed nowhere;
   `GaussianModelClient::new` builds a `ModelClient` directly. It is the natural home for the
   certificate cache if the facade ever grows a server half.
4. **`parent_ids()` is positional and untyped** (`distribution.rs:122`, `:164`). §5's
   `normalizer_parents` repairs this for the one purpose that needs it rather than adding a general
   facility, but the underlying weakness stays and will resurface for any algorithm that needs to
   distinguish parents by role.

## 9. Verification

Per-backend entry points over a shared `common/` module, as in
`skoro-linalg/tests/{ndarray_f32,ndarray_f64,wgpu_f32}.rs`. Heed the workspace doctrine
(`.claude/skills/skoro-diff-operation/references/testing.md`): hand-computed constants generated from
the implementation lock in whatever the implementation already did.

Items 2 and 6 are the gates — they are the ones that fail silently otherwise. Write them first.

1. **Residual and density agree.** `‖whitened_residual(x)‖²` reconstructs `log_density(x)` up to
   `constant + log_det`, on `Real`, `EuclideanD` and `SE3` means — the last so the
   dimension-versus-storage split (`flattened-tangent-spaces.md` §2) is exercised.
2. **The normalizer precondition, both directions.** A model with a free `Tril` covariance in the
   `free` layout is **rejected**; the *same* model with that covariance outside `free` is
   **accepted**. This is the gate on §5, and a naive "covariances must be constant" implementation
   fails the second half.
3. **LM over the certificate.** `solve` against `residual_layout()` reaches the analytic solution on
   a linear-Gaussian least-squares problem — in one iteration at `λ = 0`, and to the same point when
   damped. Shared with `flattened-tangent-spaces.md` §8 item 6.
4. **A forged certificate is empty.** Construct a `ResidualCertificate` by hand and assert it holds
   no factors; then assert `record` cannot be reached without a `WhitenedResidual` distribution — a
   `trybuild` compile-fail case. This is §2's guarantee made executable, and it is the gate on the
   whole design: if `record` ever grows an unbounded sibling, every claim in this document lapses.
5. **The certificate survives retirement.** After a tick retires and installs a joint marginal, the
   recorded factor set still equals `graph.log_likelihoods()` (§6, Lifecycle).
6. **The convention holds mechanically.** `GaussianStructure` **is** implementable downstream — that
   is the cross-crate variance §2 exists to admit, and `skoro-normal` implementing it is the proof.
   Pair that with the existing `skoro-normal/tests/facade.rs:23`, which shows `GaussianModelClientOps`
   accepting both a restricted and an unrestricted client. Together they pin §3: openness is not the
   distinguishing property, since both traits have it — the guarded return type is.

> Do **not** write a test asserting `GaussianStructure` cannot be implemented externally. An earlier
> draft of this document specified exactly that, and §2 now explains why it would be asserting the
> opposite of what the design needs.

Run with `cargo test --workspace`. The wgpu suites additionally exercise the device path.
