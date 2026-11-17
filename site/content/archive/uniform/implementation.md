---
summary: A type of finite mass declares its log-mass and its uniform draw in the core, `Rotation` declares both, and a new crate `skoro-uniform` holds the distribution `Uniform`, a constant term with an exact sampler and an empty residual, on curved and categorical types alike; the normal gains only documentation and a test of what a large scale does.
depends_on: [archive/uniform/math, uniform, archive/normal/implementation, archive/measures/implementation]
---

## Overview

When this is built, a quantity of a type of finite mass can be given the uniform distribution:

```rust
use skoro_core::scope::sample;
use skoro_core::Boolean;
use skoro_normal::{Normal, Scale};
use skoro_spatial::Rotation;
use skoro_uniform::Uniform;

let heading = sample("heading", Rotation, Uniform::on(Rotation));
let outlier = sample("outlier", Boolean, Uniform::on(Boolean));
let compass = sample("compass", Rotation, Normal::on(Rotation, heading, Scale::sd(0.05)));
```

`Uniform::on(Rotation)` scores `heading` with the constant term [](eq:archive/uniform/math/term),
$-\log 8\pi^2$, and draws it by [](eq:archive/uniform/math/rotation-draw) from three variates of
its key. `Uniform::on(Boolean)` scores with $-\log 2$ and draws by
[](eq:archive/uniform/math/categorical-draw) from one. Generation, importance weights and
retirement treat the term as any other with an exact sampler. A linearization reads it as a factor
with no rows, so the model above is linearized and minimized with `heading` free, and the estimate
of `heading` is what `compass` alone says.

Three things are added, in three crates.

- **`skoro-core`**: an embedded type may declare its mass, `Mass`, a log-mass and a draw, beside
  its volume; a `Space` answers for its log-mass and draws a uniform point, a product by its
  factors; and `testing::check_mass` checks a declaration.
- **`skoro-spatial`**: `Rotation` declares $\log 8\pi^2$ and the draw
  [](eq:archive/uniform/math/rotation-draw). `Pose` and `ExtendedPose` declare nothing.
- **`skoro-uniform`**, a new crate: the distribution `Uniform` and its sampler `UniformSample`.

The normal's code does not change. What
[the mathematics decided](math.md#a-normal-with-a-large-scale) of it is documentation: a
paragraph in the doc comments of `skoro-normal`, written at the build, the pages, written at
publish, and a test that draws the table of median angles.

## Constraints

Read at commit `ec23df3`.

**A term's log-density is a mapped node, one per term.** `crates/skoro-core/src/term.rs`:

```rust
        let node = self.model.nodes.get(&id).ok_or(Error::UnknownNode { id })?;
        if node.operation().is_none()
            || *node.space() != crate::Space::Real
            || self.model.terms.contains_key(&id)
        {
            return Err(Error::NotALogDensity { id });
        }
```

A constant node has no operation, and a term is identified by its log-density node, so the
uniform's term cannot be a constant, nor one node shared by two terms. Each `score` must add a
mapped `Real` node of its own.

**Nothing public gives a handle's space.** `Draft::spaces` is `pub(crate)`. The normal learns that
`x` is of its space because `x.local(mean)` is an operation whose `output` checks its inputs. The
uniform has no such operation to lean on, and must still refuse a quantity of another space.

**A distribution is `Distribution`, and its sampler a `Sample`.** `crates/skoro-core/src/term.rs`:

```rust
pub trait Sample: Debug + Send + Sync + 'static {
    fn name(&self) -> &'static str;
    fn params(&self) -> Vec<Param>;
    /// One draw per row, from the values of the sampler's inputs and one key per row.
    fn draw(&self, inputs: &[Value], keys: &[Key], target: &Target) -> Value;
}

pub struct Sampler {
    pub sample: Arc<dyn Sample>,
    pub inputs: Vec<QDyn>,
    /// `None` for an exact sampler; otherwise `log q_t`, a `Real` node reading `s_t` and `C_t`.
    pub log_density: Option<Q<Real>>,
}

pub trait Distribution {
    #[track_caller]
    fn score(&self, x: QDyn) -> TermHandle;
}
```

The uniform's sampler has no inputs and is exact. Its draw must return a `Value::Continuous` of
the space's storage for a continuous space and a `Value::Discrete` for a categorical one, on
`target`, as `NormalSample::draw` returns `Points::from_host(.., rows, n, target)`.

**Variates come from the row's key.** `crates/skoro-core/src/key.rs`:

```rust
    /// `n` variates in the open interval (0, 1), from the extendable output keyed by this key
    /// under tag `0x04`: each takes 8 bytes `u`, little-endian, as `((u >> 12) + 0.5) * 2^-52`.
    pub fn uniforms(&self, n: usize) -> Vec<f64> {
```

These are the grid of [](eq:archive/uniform/math/categorical-draw). A variate is never $0$ or $1$.

**An embedded type is declared in steps, and is its name.** `crates/skoro-core/src/embedded.rs`:

```rust
/// A name and its parameters name the declaration whole: two declarations under one name and the
/// same parameters are one space, so they must not differ in any part, the volume included.
pub struct Embedded { /* name, params, storage, dimension, retraction, inverse, projection, */
    volume: Option<Volume>,
    /* flat, identification */
}

    pub fn with_volume(self: Arc<Self>, volume: Volume) -> Result<Arc<Embedded>, Error>
    pub fn stored_up_to(self: Arc<Self>, generators: Vec<Linear>) -> Result<Arc<Embedded>, Error>
```

and `crates/skoro-spatial/src/types.rs` declares the three spatial types by one function:

```rust
    Embedded::declare(name, Vec::new(), n, d, r, q, Some(p))?
        .stored_up_to(vec![sign])?
        .with_volume(Volume::global(lambda))
```

The mass is one more step of the same kind, and part of the declaration a name stands for.
`Rotation` is stored as `(w, x, y, z)`.

**The linearizer wants a residual of every factor with a free quantity.**
`crates/skoro-linearize/src/linearizer.rs`:

```rust
            if blocks.is_empty() {
                continue;
            }
            blocks.sort_unstable();
            let residual = term
                .bound(&roles::RESIDUAL)
                .map_err(|_| Error::NoResidual { term: id })?;
            let components = snapshot
                .space(residual)
                .and_then(|space| space.dimension())
                .unwrap_or(0);
```

The role is `skoro_normal::roles::RESIDUAL`, "of any number of components". A residual of none
is already a factor: `Stacked::improper(0, ..)` makes one, and
`values_of_no_entries_through_every_pass` (`crates/skoro-core/tests/passes.rs`) runs a
`constant(VectorDyn(0), ..)` through every pass. A target of a tangent pass that no seed reaches
is given a zero frame (`Program::run`, `Frame::zeros(rows, dimension, columns, ..)`), so a constant
residual has a Jacobian of no rows. The linearizer therefore needs no change, provided the
uniform binds such a residual. That reading is of the code and not of a run, and
`a_uniform_prior_adds_no_rows` is the test that confirms it. A term reached from no free quantity is skipped before the role is
read, which is the case of an observed categorical quantity; a free one is refused by
`refuse_free_discrete` before any term is looked at.

**The mass is checked today by two tests that state it.** `volume_has_the_stated_mass`
(`crates/skoro-spatial/tests/types.rs`) integrates the Cayley volume along a ray and compares it
with the literal `8.0 * PI * PI`, and `sphere_charts_have_the_mass_of_the_sphere`
(`crates/skoro/tests/sphere.rs`) does the same for the two test spheres with `4.0 * PI`. These are
the check [](eq:archive/measures/math/total-mass), with the test supplying the mass. The test
spheres are declared in `crates/skoro/tests/common/sphere.rs`, outside the library.

**Refusals are recorded at the caller's line.** `Normal::score` never returns an error: a refusal
is a failed handle made by `scope::apply` of an operation whose `output` gives the error
(`crates/skoro-normal/src/refuse.rs`, private to that crate, named `skoro.normal`). The uniform
refuses the same way, under its own name.

The archive branch has no uniform distribution to compare with.

## Design

### The core: a declared mass

`crates/skoro-core/src/embedded.rs` gains the declaration, exported from the crate root beside
`Volume`.

```rust
/// A uniform point of a type from variates: `variates`, as many as the type has dimensions, each
/// in (0, 1), to `point`, the storage of one point. It carries Lebesgue measure on the cube to
/// the type's reference measure divided by its mass.
pub type UniformDraw = fn(variates: &[f64], point: &mut [f64]);

/// The mass of a type's reference measure, finite, and how a uniform point is drawn: what a type
/// declares to have a uniform distribution.
#[derive(Clone, Copy, Debug)]
pub struct Mass {
    log_mass: f64,
    draw: UniformDraw,
}

impl Mass {
    pub fn new(log_mass: f64, draw: UniformDraw) -> Mass;
    /// `log ν(𝕏)`.
    pub fn log_mass(&self) -> f64;
    pub fn draw(&self) -> UniformDraw;
}

impl Embedded {
    /// This declaration, with the mass of the measure its volume is against. Refuses a
    /// declaration without a volume, whose measure is not fixed, a flat one, one that already
    /// declares a mass, and a log-mass that is not finite (`InvalidDeclaration`).
    pub fn with_mass(self: Arc<Self>, mass: Mass) -> Result<Arc<Embedded>, Error>;
    /// The mass, when the type declares one.
    pub fn mass(&self) -> Option<&Mass>;
}
```

`Embedded` gains the field `mass: Option<Mass>`, `None` from `declare` and from the flat
declaration.

> [!DECISION] The draw is a host function, not a composite
> Everything else a type declares is a composite, because it is compiled for a device and
> differentiated. The uniform draw is neither: no parameter reaches it, so no pass ever crosses
> it, and every sampler runs on the host in double precision. A composite would be compiled into
> a program to compute four numbers, and would need the core to have every function a draw may
> use. A function pointer is `Copy`, `Debug` and comparable by nothing, which suits a declaration
> that is its name: `Embedded`'s equality stays that of the name and the parameters.

`Space` answers for the mass in `crates/skoro-core/src/retraction.rs`, beside `log_volume` and
`domain`.

```rust
impl Space {
    /// `log ν(𝕏)`: `log n` for `Categorical(n)`, the declared log-mass of an embedded type, and
    /// the sum of the factors' for a product. Every other space has infinite mass or declares
    /// none (`NoMass`).
    pub fn log_mass(&self) -> Result<f64, Error>;

    /// A uniform point of a continuous space of finite mass, from `dimension` variates: an
    /// embedded type by its declared draw, a product factor by factor, each from the variates of
    /// its tangent range into its storage range (`NoMass` otherwise, a categorical space
    /// included, whose value is not a point of storage).
    pub fn draw_uniform(&self, variates: &[f64], point: &mut [f64]) -> Result<(), Error>;
}
```

`crates/skoro-core/src/error.rs` gains one variant.

```rust
    /// A space asked for its mass or a uniform point whose reference measure has infinite mass,
    /// or which declares none.
    NoMass { space: Space },
```

Its message is "`{space:?}` has no finite mass: it has no uniform distribution".

`testing::check_mass` is the check a crate runs on its declaration, after `check_volume`.

```rust
/// The check a crate runs on its declaration's mass, at the points given, one per row:
///
/// - a draw from each of 65 536 keys is finite and on the type: `R(x, 0) = x` up to the group
///   the type is stored up to;
/// - at each point `μ` given, with `t = Q(μ, x)` over the draws `x` and `φ` the standard normal
///   density of `ℝ^d`, the mean of `φ(t) e^{-Λ(μ, t)}` times the mass is 1 to within four of its
///   standard errors, estimated from the same draws. A domain, when one is declared, is where
///   every `t` must lie, and the mean is then compared with the normal's mass of the domain, which
///   the caller gives.
pub fn check_mass(
    declaration: &Arc<Embedded>,
    points: &HostValue,
    normal_mass_of_domain: f64,
) -> Result<(), String>;
```

The second check is one identity that binds the draw, the volume and the mass together. By
[%full](statement:math/uniform/tangent) the coordinate $t$ of a uniform point has the density
$e^{\Lambda} / \nu(\manifold)$ on the domain $U_\mu$, so the mean of $\varphi(t)\, e^{-\Lambda(\mu,
t)}$ is $\int_{U_\mu} \varphi / \nu(\manifold)$: one over the mass on a global chart, and the
normal's mass of the domain over it otherwise. A draw that is not uniform, a volume wrong away
from the centre and a mass wrong by a factor each fail it, in any dimension, where the quadrature
of the existing tests needs the chart's symmetry. The summand is bounded for the charts at hand
(on the Cayley chart it is a Gaussian times $(1 + r^2/4)^2$), so its mean has a standard error. The
keys are fixed, so the test is deterministic. It runs on the host, as the draws do.

### The rotations declare theirs

`crates/skoro-spatial/src/types.rs` adds the draw and one step to `declare`, for `Kind::Rotation`
alone.

```rust
/// A unit quaternion `(w, x, y, z)` of a uniform rotation, from three variates: Shoemake's map.
fn uniform_rotation(u: &[f64], q: &mut [f64]) {
    let (a, b) = ((1.0 - u[0]).sqrt(), u[0].sqrt());
    let (phi, psi) = (TAU * u[1], TAU * u[2]);
    q[0] = a * phi.cos();
    q[1] = a * phi.sin();
    q[2] = b * psi.cos();
    q[3] = b * psi.sin();
}

/// `log 8π²`: the mass of Haar measure on `SO(3)`, normalized at the chart's centre.
const LOG_MASS: f64 = /* (8.0 * PI * PI).ln() */;
```

This is [](eq:archive/uniform/math/rotation-draw) with the parts in storage order. `declare` ends,
for the rotation, with `.with_volume(Volume::global(lambda))?.with_mass(Mass::new(LOG_MASS,
uniform_rotation))`. A pose built from a uniform rotation and a position of its own is a model's
to write, with the operations the crate already has; the type `Pose` has no uniform distribution,
and asking for one is refused with `NoMass`.

The test spheres of `crates/skoro/tests/common/sphere.rs` declare `log 4π` and the first map of
[%full](statement:math/uniform/draws), `(√(1 - c²) cos φ, √(1 - c²) sin φ, c)` with `c = 1 - 2u₁`
and `φ = 2πu₂`, on both `SPHERE` and `EXPONENTIAL`.

### The distribution

A new crate, `crates/skoro-uniform`, a member of the workspace with `skoro-uniform` among the
workspace's dependencies, depending on `skoro-tensor`, `skoro-core` and `skoro-normal`. The last
is for one constant, `skoro_normal::roles::RESIDUAL`, which is named without the normal's prefix
because it means the same of any term.

> [!DECISION] A crate of its own
> The alternative was a module of `skoro-normal`, which already holds the improper factor. The
> uniform is not a normal in any reading, and a crate named for the normal that holds every
> distribution would have to be renamed sooner or later. A crate per distribution keeps each with
> its tests and its page. Its dependency on `skoro-normal` is for the role alone; moving the
> shared roles into the core would remove it, and is a chore for when a second reader needs it.

`crates/skoro-uniform/src/lib.rs` exports `Uniform` and `UniformSample`.

```rust
/// The uniform distribution on a space of finite mass: the law `ν / ν(𝕏)`, with no parameters.
#[derive(Clone, Debug)]
pub struct Uniform {
    space: Space,
}

impl Uniform {
    /// The uniform distribution on the space `marker` declares.
    pub fn on<T: Declares>(marker: T) -> Uniform;
    /// The uniform distribution on `space`, for a product or any space without a marker.
    pub fn in_space(space: Space) -> Uniform;
    pub fn space(&self) -> &Space;
    /// `log ν(𝕏)` (`NoMass` for a space without a finite mass).
    pub fn log_mass(&self) -> Result<f64, Error>;

    /// Scores `x`: the constant term `-log ν(𝕏)`, its exact sampler, and a residual of no
    /// components. Refuses a space without a finite mass (`NoMass`) and an `x` of another space
    /// (`SpaceMismatch` of `skoro.uniform.log_density`), recording the refusal at the caller's
    /// line.
    #[track_caller]
    pub fn score(&self, x: impl Into<QDyn>) -> TermHandle;
}

impl Distribution for Uniform { /* by Uniform::score */ }
```

`score` is short enough to give whole, since how the term is made is the design.

```rust
    pub fn score(&self, x: impl Into<QDyn>) -> TermHandle {
        let x = x.into();
        let term = self.log_mass().and_then(|log_mass| {
            let space = self.space.clone();
            Composite::define(
                "skoro.uniform.log_density",
                vec![Param::Space(space.clone())],
                &[space],
                |_| (-constant(Real, log_mass)).untyped(),
            )
            .map_err(|failure| failure.error)
        });
        let log_density = match term {
            Ok(composite) => composite.call(&[x]),
            Err(error) => refuse(error, &[x]),
        };
        score(x.narrow::<Scored>(), log_density.narrow())
            .sampler(Sampler {
                sample: Arc::new(UniformSample::new(self.space.clone())),
                inputs: Vec::new(),
                log_density: None,
            })
            .bind(&roles::RESIDUAL, constant(VectorDyn(0), &[] as &[f64]))
    }
```

- **The term is a composite of `x` that does not read it.** A call is a mapped node, so it is a
  log-density; each call is a node of its own, so two quantities have two terms; the composite's
  input is the distribution's space, so a call on a quantity of another space is refused by the
  core's own check of an operation's inputs; and its body is the constant, so every derivative
  map of it is zero. The core's flat log-volume, `|_| constant(Real, 0.0)`, is a composite of the
  same shape, over a continuous input; that a composite takes a categorical input as well is
  what `term_is_minus_the_log_mass` confirms on `Boolean`. The alternatives were a new primitive operation, which would need a kernel on every
  backend to compute a constant, and a public way to read a handle's space, which the core has
  chosen not to give.
- **The composite is defined at each `score`.** Its name and parameters make it the same
  operation each time, and a definition of one constant is cheap; the normal caches its composites
  because they are not.
- **`Scored`** is the marker that admits any space, as in `skoro-normal`, private to the crate.
  **`refuse`** is that crate's too, copied under the name `skoro.uniform`: an operation whose
  `output` gives its error.
- **The residual is bound on every space**, a constant `Vector(0)`. On a continuous quantity it is
  what a linearization reads: [a factor with no rows](math.md#in-skoro-s-terms). On a categorical
  one no reader reaches it.
- **No other role is bound.** There is no mean, no scale, no tangent and no log-volume, and with
  no parameter there is no information to give a root of. A Fisher assembly asks for a root only
  of a factor whose scale is free, and this factor binds no scale.

The sampler, in `crates/skoro-uniform/src/sample.rs`:

```rust
/// Draws a uniform point on the host, from the variates of each row's key: as many as the space
/// has dimensions, or one for a categorical space.
#[derive(Debug)]
pub struct UniformSample {
    space: Space,
}

impl Sample for UniformSample {
    fn name(&self) -> &'static str { "skoro.uniform" }
    fn params(&self) -> Vec<Param> { vec![Param::Space(self.space.clone())] }

    fn draw(&self, _: &[Value], keys: &[Key], target: &Target) -> Value {
        match self.space.cardinality() {
            Some(n) => /* Value::Discrete of category(n, key.uniforms(1)[0]), a row per key */,
            None => /* Value::Continuous of self.space.draw_uniform(&key.uniforms(d), row) */,
        }
    }
}

/// `⌊n u⌋`, never `n`: the category of a variate `u` in (0, 1).
pub(crate) fn category(n: usize, u: f64) -> i64 {
    ((n as f64 * u) as usize).min(n - 1) as i64
}
```

`category` is [](eq:archive/uniform/math/categorical-draw). The product `n as f64 * u` is rounded
to a double, and for `u` just below $1$ it may round to `n`, so the minimum is what the
mathematics asks when it says the rounding cannot return $n$. `Uniform::score` has already
refused a space without a mass, so the draw of a term that was added cannot fail, and
`draw_uniform`'s error is unreachable there. A categorical value fits every target's integers,
being below `n`.

### The normal

No code of `skoro-normal` changes. Its crate documentation and the doc comment of `Normal` gain a
paragraph, written at the build with the code it describes: on a curved type the scale is in the
chart's coordinates; the mean is a mode only while every eigenvalue of the covariance is below
one; beyond that the mass goes towards the chart's cut, the half-turns from the mean on
`Rotation`; and a quantity of which nothing is known is given `skoro_uniform::Uniform`. The pages
say the same, with the table, at publish.

## Tests

### `skoro-core`

- `space::tests::log_mass_of_the_spaces` (unit): `Categorical(5)` gives `ln 5` and `BOOLEAN`
  `ln 2`; `Real`, `Vector(3)`, `Positive(2)`, `Integer` and a product of flat spaces give `NoMass`
  naming the space. It shows the rule of which spaces have a mass.
- `tests/embedded.rs::with_mass_is_refused_without_a_volume_or_twice`: `with_mass` on a
  declaration with no volume, on a flat one, a second time, and with a log-mass of infinity or
  NaN, each `InvalidDeclaration`; and after a volume it succeeds and `mass()` returns what was
  declared.
- `tests/embedded.rs::a_product_has_the_mass_of_its_factors`: with a small declared type of the
  test's own, `Space::product` of two of them has twice the log-mass, `draw_uniform` fills each
  factor's storage from its own variates, and a product with a `Real` factor gives `NoMass`.

### `skoro-spatial`

- `tests/types.rs::volume_has_the_stated_mass`, amended: the quadrature is compared with
  `ROTATION.mass().unwrap().log_mass().exp()` in place of the literal. This is
  [](eq:archive/measures/math/total-mass) against the declaration.
- `tests/types.rs::rotation_mass_checks`: `check_mass(&ROTATION, points, 1.0)` at the points the
  volume is checked at. It shows that the draw, the Cayley volume and the mass are one law.
- `tests/types.rs::uniform_rotation_is_shoemakes_map`: at the variates `(0.25, 0.5, 0.75)` the
  draw is `(-√0.75, 0, 0, -0.5)` to rounding, and at a grid of variates it has unit length. It
  pins the formula and the storage order.
- `tests/types.rs::uniform_rotations_have_the_haar_angle`: 65 536 draws; the angle `θ = 2
  atan(|x ⊖ μ| / 2)` from a fixed `μ` has an empirical distribution function within `1.63 / √N`
  of `(θ - sin θ) / π`, the Kolmogorov bound at the 1 % level, and the mean of the unit direction
  of `x ⊖ μ` is within `4 / √(3N)` of zero in each component. It is the mathematics' test of the
  draw, by the chart's route, at a `μ` that is not the identity.
- `tests/types.rs::poses_have_no_mass`: `Space::log_mass` of `POSE` and `EXTENDED_POSE` is
  `NoMass`.
- `tests/normal.rs::broad_normals_turn_towards_the_half_turns`: 16 384 draws of
  `Normal::on(Rotation, μ, Scale::sd(σ))` for `σ` in 0.5, 1, 3, 10 and 100; the fraction of draws
  whose angle from `μ` is below `2 atan(σ m / 2)`, `m = 1.5382`, is within `1.5 / √N` of one
  half; and at `σ = 100` the fraction within 0.1 rad of a half-turn is within 0.01 of 0.984. It
  draws the table of [the mathematics](math.md#a-normal-with-a-large-scale) from the normal's own
  sampler, so that what the pages will say is what the library does.

### `skoro-uniform`

- `tests/density.rs::term_is_minus_the_log_mass`: on `Rotation`, `Categorical::<5>`, `Boolean`
  and the product of two rotations, the term's value at any point is `-ln 8π²`, `-ln 5`, `-ln 2`
  and `-2 ln 8π²`, on the host and in single precision.
- `tests/density.rs::derivatives_are_zero`: for a free rotation scored by `Uniform` alone, the
  gradient of the log-density query is a zero frame and its Hessian products are zero, at several
  points. It shows the term changes no mode and no step.
- `tests/refusals.rs::a_space_of_infinite_mass_is_refused`: `Uniform::on` of `Real`,
  `Vector::<3>`, `Positive::<2>`, `Integer` and `Pose` scores to a failed handle, and the build
  reports `NoMass` of that space at the line of the call.
- `tests/refusals.rs::a_quantity_of_another_space_is_refused`: `Uniform::on(Rotation).score(x)`
  for `x` a `Vector::<4>` reports `SpaceMismatch` of `skoro.uniform.log_density`.
- `tests/roles.rs::binds_an_empty_residual_and_nothing_else`: the term binds
  `roles::RESIDUAL` to a node of space `Vector(0)`, and no mean, scale, tangent, log-volume or
  information root.
- `tests/sample.rs::draws_are_functions_of_key_and_address`: the generated value of a uniform
  rotation at row `i` is `uniform_rotation` of `key.split(rows)[i].at(address).uniforms(3)`, the
  same on every run, and differs between addresses.
- `tests/sample.rs::categorical_draws_are_floor_n_u`: for `n` in 2, 3, 7 and 1000, the generated
  value is `⌊n u⌋` of the key's one variate, and lies in `0..n`.
- `sample::tests::category_never_returns_n` (unit): `category(n, u)` at `u = 1 - 2⁻⁵³` and at
  the smallest variate, for `n` of 1, 2, 3, 1000 and 2⁵², is `n - 1` and `0`.
- `tests/sample.rs::categorical_counts_are_uniform`: 60 000 draws of `Categorical::<6>`; the
  chi-square statistic against equal counts is below 20.5, the 0.999 quantile at five degrees of
  freedom.
- `tests/sample.rs::generation_weighs_the_likelihood_alone`: for the overview's model with
  `compass` observed, the log-weight of each generated row is the normal's log-density of
  `compass` at the drawn `heading`, with neither `ln 8π²` nor `ln 2` in it. It shows that the
  sampler is declared exact, so that the uniform terms of the drawn quantities cancel against
  their draws.

### `skoro-linearize`

`skoro-uniform` becomes a dev-dependency.

- `tests/linearize.rs::a_uniform_prior_adds_no_rows`: `heading` uniform and free, `compass`
  normal and observed. `Linearizer::new` succeeds with two factors; the linearized problem's rows
  are the three of `compass`; and the objective is the density query's, the uniform's constant
  included. It shows the term has its place in a linearization with no change to the linearizer.
- `tests/minimize.rs::a_uniform_prior_leaves_the_estimate_to_the_data`: Levenberg–Marquardt from
  a start a radian away converges to `heading = compass` up to the quaternion's sign.
- `tests/refusals.rs::a_quantity_under_a_uniform_alone_is_undetermined`: with `heading` scored
  by `Uniform` and nothing else, the linearizer is made, and an undamped solve reports
  `Undetermined` for its block: the posterior is uniform and has no mode.

### `skoro`

`skoro-uniform` becomes a dev-dependency.

- `tests/sphere.rs::sphere_charts_have_the_mass_of_the_sphere`, amended: both quadratures are
  compared with the declared masses.
- `tests/sphere.rs::sphere_masses_check`: `check_mass` on `SPHERE`, with a normal mass of 1, and
  on `EXPONENTIAL`, with the standard normal's mass of the disc of radius π in the plane,
  `1 - e^{-π²/2}`. The second is the check on a chart that is not global.
- `tests/sphere.rs::uniform_on_the_exponential_sphere`: `Uniform::on(Exponential)` scores and
  generates where `Normal::on(Exponential, ..)` is refused, the term is `-ln 4π`, and the height
  `x₃` of 65 536 draws is uniform on `(-1, 1)` to within the Kolmogorov bound.
- `tests/public_interface.rs`: the additions to the core's public interface, `Mass`,
  `UniformDraw`, `Embedded::with_mass`, `Embedded::mass`, `Space::log_mass`,
  `Space::draw_uniform`, `Error::NoMass` and `testing::check_mass`, as that file records every
  other.

## Open questions

None remain.
