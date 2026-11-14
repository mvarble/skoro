---
summary: Every continuous `Space` gets a log-volume composite and a chart domain, an embedded declaration takes its chart's volume with `with_volume`, the core refuses a density, a proposal or an integration over a type without one, and `testing::check_volume` holds a declaration to the area formula.
depends_on: [archive/measures/math, chart-volume, archive/core/implementation]
---

## Overview

When this is built, every continuous space of `skoro-core` can say what its chart's log-volume is,
as a composite like its retraction:

- `Space::log_volume()` is the composite of $\lambda_\mu(\tangent)$, of inputs the space and
  `Vector(d)`, to `Real`: zero on `Real`, `Vector` and `Matrix`, $\sum_i (\log \mu_i + \tangent_i)$
  on `Positive` [](eq:archive/measures/math/core-volumes), the factors' sum on `Product`
  [](eq:archive/measures/math/product-volume), and $\Lambda(\iota(\mu), \tangent)$ on an embedded
  type that declares one [](eq:archive/measures/math/declared);
- `Space::domain()` says whether the chart is global, and otherwise gives the composite of
  $\tangent \in U_\mu$, to `Boolean`;
- `Space::has_measure()` says whether the space has a reference measure at all: every space but an
  embedded type that declares no volume, and the products with one among their factors.

An embedded type declares its chart's volume after its retraction, with one more call:

```rust
// In a feature crate, beside the sphere's R, Q and P.
let log_volume = Composite::define(
    "sphere.log_volume",
    Vec::new(),
    &[Space::Vector(3), Space::Vector(2)],
    |q| ((q[1] * q[1]).sum() * 0.25 + 1.0).log() * -2.0, // -2 log(1 + |t|²/4)
)?;
let sphere = Embedded::declare("sphere", Vec::new(), 3, 2, r, q, Some(p))?
    .with_volume(Volume::global(log_volume))?;
```

and its tests run `testing::check_volume` on it, as they run `check_embedded`. A distribution with a
density through a chart, the normal entry's tangent normal first, writes its term as
[](eq:archive/measures/math/chart-density):

```rust
// log f(x ⊖ μ) - λ_μ(x ⊖ μ), on any space that has a measure.
let t = x.local(mu);
let term = gaussian(t, scale) - mu.log_volume(t);
scope::score(x, term);
```

The core enforces [Types without a volume](math.svx#types-without-a-volume): `scope::score`,
`scope::propose` and `Writer::retire` refuse, with `Error::NoVolume`, a quantity whose space has no
measure, scored, proposed or integrated out. Nothing else changes for such a type: it is conditioned
on, observed and optimized as before.

## Constraints

Read at commit `1a489a4`.

### An embedded declaration

`crates/skoro-core/src/embedded.rs`:

```rust
/// A curved type of storage `n` and dimension `d`, declared by its retraction and its inverse.
pub struct Embedded {
    name: &'static str,
    params: Vec<Param>,
    storage: usize,
    dimension: usize,
    retraction: Arc<Composite>,
    inverse: Arc<Composite>,
    projection: Option<Arc<Composite>>,
    flat: bool,
}

impl Embedded {
    pub fn declare(
        name: &'static str,
        params: Vec<Param>,
        storage: usize,
        dimension: usize,
        retraction: Arc<Composite>,
        inverse: Arc<Composite>,
        projection: Option<Arc<Composite>>,
    ) -> Result<Arc<Embedded>, Error>;

    pub fn declare_flat(name: &'static str, params: Vec<Param>, n: usize)
        -> Result<Arc<Embedded>, Error>;
}

/// Two declarations are one type when their names and parameters are.
impl PartialEq for Embedded { /* name and params */ }
```

`declare` takes seven arguments, clippy's default limit for `too_many_arguments`, which the
workspace runs with `-D warnings`; an eighth for the volume would fail the lint. A declaration is
returned in an `Arc`, and the crate that writes it keeps it in a `LazyLock` and reaches it again
through the registry's `MakeEmbedded`, `fn(&[Param]) -> Result<Arc<Embedded>, Error>`. Equality is
by name and parameters alone, so the volume must be part of what a name and its parameters always
make, never something one copy of a declaration has and another lacks.

Every declared map is an `Arc<Composite>`, of Euclidean inputs, and a fused map is declared as the
composite of one call to it, so the boundary kernels compile every declaration the same way. The
log-volume and the domain follow the same rule.

### The spaces' composites

`crates/skoro-core/src/retraction.rs` gives every continuous space its retraction and inverse as
composites, defined once per space and cached:

```rust
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
enum Map {
    Retraction,
    Inverse,
}

type Cache = Mutex<HashMap<(Space, Map), Arc<Composite>>>;

impl Space {
    pub fn retraction(&self) -> Result<Arc<Composite>, Error> { cached(self, Map::Retraction) }
    pub fn inverse_retraction(&self) -> Result<Arc<Composite>, Error> { cached(self, Map::Inverse) }
}
```

A product's composite is built from its factors' on the slices `blocks()` gives, and an embedded
type's from its boundary operations. The log-volume and the domain of a space are built the same
way, in the same cache, so that a product with a curved factor gets them with no code of its own.
`QDyn::retract` and `QDyn::local` in `ops.rs` apply these composites to handles, and
`Q<T>::retract` and `Q<T>::local` wrap them; `y.local(x)` is $y \ominus x$.

### The discretizations

`crates/skoro-core/src/primitive/discrete.rs` has the core's operations to `Boolean`, all of order
zero and passing no cotangent back:

```rust
/// `1[x ≥ 0]`, a boolean: a discretization, passing no cotangent back.
IsNonnegative, "skoro.is_nonnegative", [Space::Real] -> Space::BOOLEAN
/// A discrete value read as a real: an embedding, with no derivative.
ToReal, "skoro.to_real", [Space::Integer | Space::Categorical(_)] -> Space::Real
```

There is no conjunction of booleans. A domain written as a composite is a discretization of a real
by its sign, `is_nonnegative`, and the conjunction of several is read off the sum of their
`to_real`, as below.

### Terms, proposals and retirement

`crates/skoro-core/src/term.rs`, `Draft::add_term`, after the scored node is found to be a
quantity not yet scored:

```rust
let scores = match scores {
    Some(q) => {
        let quantity = self.model.nodes.get(&q.id()).ok_or(Error::UnknownNode { id: q.id() })?;
        if !quantity.is_quantity() {
            return Err(Error::NotAQuantity { id: q.id() });
        }
        if self.model.scoring.contains_key(&q.id()) {
            return Err(Error::AlreadyScored { quantity: q.id() });
        }
        Some(q.id())
    }
    None => None,
};
```

and `Draft::add_proposal`, after `node.is_quantity()` is checked. Both record their error with the
caller's location as a `Failure`, and the handle they return is failed. `Writer::retire` in
`retire.rs` reads the plan, with `integrated` the quantities of $E$ not given a value, and returns
`Result<Retirement<'_>, Error>` before anything changes. `generate.rs` already refuses a free
quantity with neither sampler nor proposal (`CannotGenerate`).

### Errors and checks

`Error` in `error.rs` is `#[non_exhaustive]`, one variant per refusal, each carrying what names the
fault (`NoProjection { space }`, `AlreadyScored { quantity }`). `testing::check_embedded` is the
check a crate runs on its declaration, at points of the manifold, on a target; it compiles `R` and
`Q`, computes $B(x)$ by a tangent pass of $\mathcal R$ at $(x, 0)$ with an identity frame at its
second root, checks a Gram matrix with `positive_definite`, and ends with `check_operation` on each
boundary operation. Every check returns `Result<(), String>`, the message naming the map and the
pass.

### The archive

The `archive` branch had no reference measures and no chart volumes: its normal was the Gaussian of
the tangent residual on every manifold, with no correction (`git grep -i` for Haar, measure and
volume finds nothing but a normal's $\log \lvert \det W \rvert$). Nothing is kept from it here.

## Design

### The declaration: `Volume` and `Domain`

In `skoro-core`, module `embedded`, re-exported from the crate root:

```rust
/// A chart's declared volume: its log-volume Λ and its domain, [the definition](math.svx#what-an-embedded-type-declares).
#[derive(Clone, Debug)]
pub struct Volume {
    log_volume: Arc<Composite>,
    domain: Domain,
}

/// Where a chart is a diffeomorphism onto its full image.
#[derive(Clone, Debug)]
pub enum Domain {
    /// `U_μ = ℝ^d` at every point.
    Global,
    /// The composite of `t ∈ U_μ`, to `Boolean`. On a declaration it takes `Vector(n)` and
    /// `Vector(d)`, the point's storage and the tangent; from `Space::domain()`, the point itself.
    Within(Arc<Composite>),
}

impl Volume {
    /// Λ, `Vector(n)` and `Vector(d)` to `Real`, on a global chart.
    pub fn global(log_volume: Arc<Composite>) -> Volume;
    /// Λ on the chart whose domain is where `domain`, `Vector(n)` and `Vector(d)` to `Boolean`, is true.
    pub fn within(log_volume: Arc<Composite>, domain: Arc<Composite>) -> Volume;
    pub fn log_volume(&self) -> &Arc<Composite>;
    pub fn domain(&self) -> &Domain;
}

impl Embedded {
    /// This declaration, with its chart's volume against the measure its retraction normalizes.
    /// Refuses a log-volume not of `Vector(n)` and `Vector(d)` to `Real`, a domain not of the
    /// same to `Boolean`, a declaration that already has a volume, and a flat one, whose volume is
    /// the core's (`InvalidDeclaration`).
    pub fn with_volume(self: Arc<Self>, volume: Volume) -> Result<Arc<Embedded>, Error>;

    /// The chart's volume, when the type declares one.
    pub fn volume(&self) -> Option<&Volume>;
}
```

`Embedded` gains a field `volume: Option<Volume>` and derives `Clone`, so that `with_volume` is
`Arc::unwrap_or_clone(self)` with the field set. `declare` sets it to `None`; `declare_flat` sets
it to the global volume of the composite `skoro.flat.log_volume`, the constant `0`, since a flat
type's measure is Lebesgue measure, normalized by $B = I$.

**Why a method and not an argument.** An eighth argument of `declare` would trip
`too_many_arguments`, and every existing call would change for a declaration most types will make.
A builder, `Embedded::builder(...).projection(p).volume(v).build()`, would read better, but it
changes every existing call too, and the projection has no need of it. `with_volume` changes no
existing call, reads in the order the declarations are written, and is refused once made, so a type
has one volume or none. The cost is the equality hazard of [the constraints](#an-embedded-declaration):
`declare(...)` and `declare(...).with_volume(...)` under one name are equal spaces with different
measures. The registry's `MakeEmbedded` and the crate's `LazyLock` make a declaration in one
expression, so a crate meets this only by declaring a name twice, and the declaration's
documentation says that a name and its parameters name the declaration whole, its volume included.

**Why the domain is a composite.** It is the one consumer-facing form of $U_\mu$ that a test and a
sampler can both evaluate on a batch, and the core already has the discretizations it is written
with. A flag alone would leave a test unable to choose tangent vectors inside a domain that is not
global. A radius, $\lVert \tangent \rVert < r$, would fit the exponential charts of the sphere but
not those of $SE(3)$, whose domain bounds only the rotation's coordinates. A discretization by sign is
closed, $\Omega = 1[\pi^2 - \lVert \tangent \rVert^2 \ge 0]$ for the exponential chart of the
sphere, where the domain is open; the difference is a boundary of measure zero, which no draw from a
density reaches.

**Fullness is not declared.** [The definition](math.svx#what-an-embedded-type-declares) asks every
declared chart to be full, so there is nothing to record: `with_volume` is a declaration that the
chart is full, and `check_volume` checks what it can of it (below).

### The spaces: log-volume, domain, measure

In `retraction.rs`, whose module documentation gains a column for the log-volume and the domain,
the cache's `Map` gains two variants:

```rust
enum Map {
    Retraction,
    Inverse,
    LogVolume,
    Domain,
}

impl Space {
    /// The composite of `λ_x(t)`, of inputs this space and `Vector(dimension)`, to `Real`
    /// (`NotContinuous` for a discrete space, `NoVolume` for one without a measure).
    pub fn log_volume(&self) -> Result<Arc<Composite>, Error>;

    /// Whether every chart of this space is global, and otherwise the composite of `t ∈ U_x`, of
    /// inputs this space and `Vector(dimension)`, to `Boolean` (errors as `log_volume`'s).
    pub fn domain(&self) -> Result<Domain, Error>;

    /// Whether the space has a reference measure: false only for an embedded type that declares
    /// no volume, and a product with such a factor.
    pub fn has_measure(&self) -> bool;
}
```

`has_measure` is true on the discrete spaces, whose measure is counting measure, though they have
no chart; it is what the refusals below ask. The composites, each named after its map with the
space as its parameter, like `skoro.retraction`:

| space | `skoro.log_volume`, $\lambda_x(\tangent)$ | `skoro.domain` |
| --- | --- | --- |
| `Real`, `Vector`, `Matrix` | `constant(Real, 0.0)` | global |
| `Positive(n)` | `(positive.log(x) + t).sum()` | global |
| `Product` | the factors' `log_volume`, on the slices of `x` and `t` by `blocks()`, summed | global if every factor's is; otherwise `is_nonnegative(Σ to_real(Ωᵢ) − k)` over the $k$ factors that are not |
| `Embedded` with a volume | `Λ(include(x), t)` | the declaration's: global, or `Ω(include(x), t)` |
| `Embedded` without | `NoVolume` | `NoVolume` |

A discrete space gives `NotContinuous` for both, as for its retraction. The positive orthant's
composite is [](eq:archive/measures/math/core-volumes) as written; read at
$\tangent = \point \ominus \mu$ it is $\sum_i \log \point_i$ only to within rounding, which the
normal entry may avoid by writing $\sum_i \log \point_i$ directly on a positive space, and need not.
The product's domain is [](eq:archive/measures/math/product-volume)'s conjunction: the sum of $k$
booleans read as reals is $k$ exactly when all are true, so its difference with $k$ is nonnegative
exactly then, with no rounding, the summands being $0$ and $1$. Since `Space::domain()` returns
`Domain`, a caller asks `matches!(space.domain()?, Domain::Global)` to know whether a sampler
through the chart is exact.

On the handles, beside `retract` and `local` in `ops.rs`:

```rust
impl<T: Typed> Q<T> {
    /// `λ_self(t)`, the log-volume of the chart at `self`, with `t` in its tangent coordinates
    /// (`NoVolume` when the space has no measure).
    #[track_caller]
    pub fn log_volume(self, t: impl Into<QDyn>) -> Q<Real>;
}

impl QDyn {
    #[track_caller]
    pub fn log_volume(self, t: QDyn) -> QDyn;
}
```

applying `Space::log_volume()` as `QDyn::retract` applies the retraction. No boundary operation is
added: $\Lambda$ is a composite on Euclidean values, read at $\iota(\mu)$, so its derivatives in
charts are its parts', the inclusion's among them [](eq:archive/measures/math/declared), and its
order is what its program admits, which bounds the order of every term that contains it.

### Refusals

One error:

```rust
/// A quantity whose space has no reference measure, scored by a term, given a proposal or
/// integrated out by a retirement; or a log-volume or a domain asked of such a space.
NoVolume { space: Space, quantity: Option<NodeId> },
```

with `quantity` set when a quantity is at fault. Three checks, each where the existing check of the
same quantity is made:

1. `Draft::add_term`, after `AlreadyScored`: a scored quantity whose space has no measure gives
   `NoVolume`. A likelihood factor scores nothing and is never refused, whatever it is conditioned
   on.
2. `Draft::add_proposal`, after `NotAQuantity`: likewise. A retirement's replacement proposals go
   through `add_proposal` and are refused there.
3. `Writer::retire`, after the plan's `integrated` is read and before the `Retirement` is returned:
   the first integrated quantity whose space has no measure gives `NoVolume`, and nothing changes.
   A quantity of $E_O$ is not refused.

Generation needs no new check: a quantity no term may score and no proposal may draw is refused by
`CannotGenerate` already, naming it. Queries and passes need none either: a free quantity without a
measure is differentiated and optimized as before.

### `testing::check_volume`

In `testing.rs`, beside `check_embedded`:

```rust
/// The check a crate runs on its declaration's volume, at the points given, one per row, on the
/// manifold, after `check_embedded`:
///
/// - Λ admits every pass the retraction admits, and each is finite at `(x, 0)`;
/// - `Λ(x, 0) = 0`, and `Ω(x, 0)` is true when a domain is declared;
/// - at tangents `t` of lengths 0.1, 1, 3 and 10 in random directions, those `Ω` admits when a
///   domain is declared and at least one of them per row: `Q(x, R(x, t)) = t`, and `Λ(x, t)`
///   agrees with the area formula;
/// - `check_operation` holds on Λ, and on `Space::Embedded(declaration).log_volume()`, at `t = 0`
///   and away from it.
///
/// A failure names the check, the row and the tangent.
pub fn check_volume(
    declaration: &Arc<Embedded>,
    points: &HostValue,
    target: &Target,
) -> Result<(), String>;
```

The area formula is [](eq:archive/measures/math/area-check), computed as `check_embedded` computes
$B$: a tangent pass of $\mathcal R$ at $(x, \tangent)$ with an identity frame at its second root
gives $\partial_2 R(x, \tangent)$, `[N, n, d]`; a tangent pass at $(\otherPoint, 0)$, with
$\otherPoint = R(x, \tangent)$ from the forward pass, gives $B(\otherPoint)$; and on the host, each
row's Gram matrices and their log-determinants, by a Cholesky factorization beside
`positive_definite`, which also reports a Gram matrix that is not positive definite. The agreement
is to the target's `Tolerance`, absolute and relative, since $\Lambda$ is a logarithm and crosses
zero. The tangents of length $10$ are there for a global chart, whose inverse must hold on the whole
chart, not only near $0$; on a domain, a tangent $\Omega$ refuses is skipped, and a row where every
tangent but $0.1$ is refused still checks that one.

What it cannot check is the total mass [](eq:archive/measures/math/total-mass), an integral over
$\R^\dimension$ that needs the type's own knowledge of where its volume lives; nor that $\Omega$ is
false beyond the domain, which is not a property of the declaration's maps. Both are left to the
type's own tests, and the sphere's below do both. A check of fullness is the total mass: a chart
that is not full has less.

### Where things live

| piece | crate, module | public |
| --- | --- | --- |
| `Volume`, `Domain`, `Embedded::with_volume`, `Embedded::volume` | `skoro-core`, `embedded` | yes, at the root |
| `Space::log_volume`, `Space::domain`, `Space::has_measure` | `skoro-core`, `retraction` | yes |
| `Q::log_volume`, `QDyn::log_volume` | `skoro-core`, `ops` | yes |
| `Error::NoVolume` | `skoro-core`, `error` | yes |
| the three refusals | `skoro-core`, `term` and `retire` | no |
| `testing::check_volume` | `skoro-core`, `testing` | yes |
| the sphere's log-volume, the exponential sphere's volume | `skoro`, `tests/common/sphere.rs` and `tests/sphere.rs` | test code |

`skoro` re-exports `Volume` and `Domain` with the rest of the core, and its `public_interface` test
names them. The core's `REVISIT.md` has no entry on measures, and gains none.

## Tests

Each test is named as it will be in the code, with the file it goes in.

**The core's spaces** (`crates/skoro-core/src/retraction.rs`, unit tests):

- `euclidean_log_volume_is_zero`: on `Real`, `Vector(3)` and `Matrix(2, 3)`, the log-volume is $0$
  at random points and tangents, and its tangent and cotangent maps are zero; their domain is
  `Global` and `has_measure` is true.
- `positive_log_volume_is_the_log_of_the_moved_point`: on `Positive(3)`, it equals
  $\sum_i \log(\mu_i e^{\tangent_i})$ on the host, which is the change of variables of
  [%full](statement:archive/measures/math/core-types); at $\tangent = \point \ominus \mu$ it is
  $\sum_i \log \point_i$ to within rounding; `check_operation` holds on it.
- `product_log_volume_is_the_sum_of_its_factors`: on a nested product of `Real`, `Positive(2)` and
  `Vector(2)`, its value is the sum of the factors' on the slices, and `check_agreement` holds
  between it and a composite that sums the factors' composites by hand.
- `discrete_space_has_a_measure_and_no_log_volume`: `has_measure` is true on `Integer` and
  `Categorical(3)`, and `log_volume` and `domain` give `NotContinuous`.

**Declarations** (`crates/skoro-core/tests/embedded.rs`):

- `with_volume_refuses_a_wrong_signature`: a log-volume to `Vector(1)`, one of the wrong storage,
  and a domain to `Real`, each `InvalidDeclaration`.
- `with_volume_refuses_a_second_volume_and_a_flat_declaration`.
- `flat_declaration_has_a_global_zero_volume`: `declare_flat` gives `volume()` with `Domain::Global`
  and a log-volume of $0$, and its space `has_measure`.
- `declaration_without_volume_has_no_measure`: `has_measure` is false on it and on a product with
  it, and `log_volume` and `domain` give `NoVolume`.
- `product_domain_is_the_conjunction_of_its_factors`: with two factors declared `Volume::within`
  balls of different radii and one global factor, the product's domain is true exactly when both
  balls hold, on tangents inside, outside and across.

**The refusals** (`crates/skoro-core/tests/terms.rs` and `crates/skoro/tests/retirement.rs`):

- `score_refuses_a_quantity_without_a_measure`: `scope::score` of a quantity of a type with no
  volume fails with `NoVolume`, naming it; so does a product quantity with such a factor.
- `factor_over_a_quantity_without_a_measure_is_allowed`: a likelihood factor conditioned on it, and
  a term scoring a real quantity conditioned on it, are added, and a query observing it and one
  leaving it free compile and run their passes.
- `propose_refuses_a_quantity_without_a_measure`.
- `generation_refuses_a_free_quantity_without_a_measure`: a query leaving it free cannot generate,
  with `CannotGenerate` naming it.
- `retire_refuses_to_integrate_a_quantity_without_a_measure`: retiring it free gives `NoVolume`
  and leaves the model unchanged; retiring it with a given value succeeds.

**The checker** (`crates/skoro/tests/sphere.rs`, against the test sphere and its variants):

- `sphere_passes_check_volume(target)`, at both precisions and on wgpu, with
  [](eq:archive/measures/math/sphere-volume) declared on the acceptance test's sphere in
  `tests/common/sphere.rs`.
- `exponential_sphere_passes_check_volume(target)`: the guarded exponential sphere of the existing
  `removable_singularity_is_reported`, declared with $\Lambda = \log(\sin \lVert \tangent \rVert /
  \lVert \tangent \rVert)$, guarded as its retraction is, below the same cutoff, by the series
  $-r^2/6 - r^4/180$ in $r^2 = \lVert \tangent \rVert^2$, and the domain
  $\Omega = \text{is\_nonnegative}(\pi^2 - \lVert \tangent \rVert^2)$: a chart that is not global,
  full, with a removable singularity at $0$.
- `check_volume_reports_a_wrong_declaration`: each of these fails, with a message naming its check:
  the sphere's log-volume plus $0.1$ (at $\Lambda(x, 0) = 0$); $-\tfrac32 \log(1 + \lVert \tangent
  \rVert^2)$, the normalization chart's, on the stereographic sphere (the area formula); the
  exponential sphere declared global (the inverse beyond $\pi$); its log-volume unguarded (finiteness
  at $(x, 0)$).
- `sphere_log_volume_in_storage`: at random pairs of points, $\Lambda(x, y \ominus x) = 2
  \log((1 + x\transpose y)/2)$, the form in storage of [What an embedded type declares](math.svx#what-an-embedded-type-declares).
- `sphere_charts_have_the_mass_of_the_sphere`: $2\pi \int_0^\infty r\, e^{\Lambda(x, r e_1)}\, \dd r$
  by the trapezoidal rule in $u = \arctan(r/2)$, and over $[0, \pi)$ for the exponential sphere,
  is $4\pi$ to $10^{-6}$ at double precision, at several $x$, evaluating the declared composites
  along a ray (both volumes depend on $\lVert \tangent \rVert$ alone): the total mass
  [](eq:archive/measures/math/total-mass), and with it the fullness of both charts.
- `exponential_domain_ends_at_the_antipode`: $\Omega$ is true below $\lVert \tangent \rVert = \pi$
  and false above it.
- `pushed_density_integrates_to_one_on_the_sphere`: with $\varepsilon$ standard normal on
  $\R^2$, the density [](eq:archive/measures/math/chart-density) at $\mu$, built in a model from
  `local` and `log_volume`, integrates to $1$ over the sphere by a product rule in latitude and
  longitude, to $10^{-4}$, at two $\mu$; without the log-volume it does not.

**Products with a curved factor** (`crates/skoro/tests/acceptance.rs`):

- `product_log_volume_agrees_with_its_factors(target)`: on the acceptance test's product of two
  reals, a positive factor and the sphere, every pass over the product's `log_volume` agrees with a
  composite that sums the factors' by hand, by `check_agreement`, and its domain is `Global`.

## Open questions

None remain.
