---
summary: A crate `skoro-normal` whose `Normal` scores a quantity of any space with a global chart, with its term built from composites of the core's operations, an exact sampler that runs the space's retraction on the host, the four forms of the scale, the roles with the information root, a likelihood factor over data, and the stacked tangent of a retirement's replacement.
depends_on: [archive/normal/math, chart-volume, archive/core/implementation, archive/measures/implementation]
---

## Overview

When this is built, a crate `skoro-normal` gives skoro its one normal distribution
([The definition](math.svx#the-definition)). A model writes a prior, a motion and a measurement as
it would say them, on flat and curved spaces alike:

```rust
use skoro::scope::{constant, quantity};
use skoro::{Positive, Real, Vector};
use skoro_normal::{Normal, Scale};

writer.build(|| {
    // A position with a prior, and a positive scale estimated with it.
    let p = quantity("p", Vector::<3>);
    Normal::on(Vector::<3>, constant(Vector::<3>, [0.0; 3]), Scale::sd(10.0)).score(p);
    let sigma = quantity("sigma", Positive::<1>);
    Normal::on(Positive::<1>, constant(Positive::<1>, [0.1]), Scale::sd(0.5)).score(sigma);

    // A direction to a landmark, measured on the sphere: the measurement is the quantity, and
    // its mean the direction predicted from a pose, a map into the sphere (`bearing`).
    let u = quantity("u", Sphere);
    Normal::on(Sphere, bearing(pose, landmark), Scale::isotropic(sigma)).score(u);
});
```

`Normal::on(marker, mean, scale).score(x)` adds the term [](eq:archive/normal/math/term) scoring
`x`, attaches the exact sampler of [The sampler](math.svx#the-sampler), and binds the roles of
[The roles](math.svx#the-roles). The term is a handful of nodes in the model: the tangent
$\point \ominus \mu$ by the core's inverse retraction, the whitened residual by a composite of the
scale's form, the normalizer by another, and the log-volume by the core's
[`Space::log_volume`](../measures/implementation.md). Its derivatives of both orders are the core's,
by composition. A query observes `u` or leaves it free, as for any quantity
([](eq:archive/normal/math/measurement)).

Beside it:

- `Scale` holds the scale in one of the four forms of [The scale](math.svx#the-scale), from
  quantities or, through helpers, from constants: a standard deviation, a covariance, a precision.
- `Normal::factor_at(z)` adds the Gaussian part at a value `z` as a likelihood factor, scoring
  nothing: a measurement over data on a type without a volume or a global chart
  ([Without a volume or a global chart](math.svx#without-a-volume-or-a-global-chart)), and the
  replacement form [](eq:archive/normal/math/joint).
- `Stacked` builds the stacked tangent [](eq:archive/normal/math/stacked) of a blanket at constant
  anchors and the replacement [](eq:archive/normal/math/joint) on it
  ([The stacked tangent](math.svx#as-a-likelihood-factor)).
- `roles` holds the role keys a least-squares or Fisher-scoring reader looks up.

A quantity whose space has no global chart is refused with the core's `SpaceMismatch`, one without
a measure with `NoVolume`, as [Why global charts](math.svx#why-global-charts) and the core's
measures require.

## Constraints

Read at commit `d439a74`. Each interface below is pasted from the file named, and followed by what
the design must do because of it.

### Terms, samplers and roles

`crates/skoro-core/src/term.rs`:

```rust
pub struct Term {
    pub log_density: NodeId,
    pub scores: Option<NodeId>,
    pub involves: Box<[NodeId]>,
    pub sampler: Option<Sampler>,
    bindings: Vec<(&'static str, NodeId)>,
}

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

impl TermHandle {
    pub fn sampler(self, sampler: Sampler) -> TermHandle;
    pub fn bind<M: Multiplicity>(self, role: &Role<M>, node: impl Into<QDyn>) -> TermHandle;
}

pub struct Role<M: Multiplicity> { name: &'static str, multiplicity: PhantomData<fn() -> M> }
impl<M: Multiplicity> Role<M> { pub const fn new(name: &'static str) -> Role<M>; }

pub fn score<T: Typed>(scored: Q<T>, log_density: Q<Real>) -> TermHandle;
pub fn factor(log_density: Q<Real>) -> TermHandle;
```

- A term is a mapped `Real` node and the quantity it scores; the normal builds its log-density from
  operations in the open scope and hands it to `score`, which refuses a quantity whose space has no
  measure (`NoVolume`) by itself.
- `Sample::draw` receives values and keys, never spaces, and returns a `Value`, not a `Result`. So
  the sampler must know the scored space when it is made, at build time, and everything that can
  fail (compiling the space's retraction) must fail then, not in `draw`.
- A sampler with `log_density: None` is exact, and the core leaves its term out of the importance
  weight; [%full](statement:archive/normal/math/density) is what licenses the normal to say so.
- Roles are typed keys matched by name, with a multiplicity; `Role::new` is `const`. The core
  defines none, so the normal's live in its crate, and their names are the identity a reader in
  another crate matches.
- A binding must be a node of the model reached from the term's quantities alone. The residual,
  the log-volume and the root must therefore be nodes of their own in the model, not internals of
  one composite call.

### Spaces, their charts and their volumes

`crates/skoro-core/src/retraction.rs` and `crates/skoro-core/src/embedded.rs`:

```rust
impl Space {
    /// The composite of `point ⊕ tangent`, of inputs this space and `Vector(dimension)`.
    pub fn retraction(&self) -> Result<Arc<Composite>, Error>;
    /// The composite of `λ_x(t)`, of inputs this space and `Vector(dimension)`, to `Real`
    /// (`NotContinuous` for a discrete space, `NoVolume` for one without a measure).
    pub fn log_volume(&self) -> Result<Arc<Composite>, Error>;
    /// Whether every chart of this space is global, and otherwise the composite of `t ∈ U_x`.
    pub fn domain(&self) -> Result<Domain, Error>;
    pub fn has_measure(&self) -> bool;
}

pub enum Domain { Global, Within(Arc<Composite>) }
```

and, on handles, `crates/skoro-core/src/ops.rs`: `x.local(from)` for `x ⊖ from`, `mu.log_volume(t)`
for $\lambda_\mu(\tangent)$, `x.retract(t)`, `slice`, `reshape`, `sum`, `matmul`, `log`, `recip`,
`as_real`, the arithmetic operators with literals, and `scope::concatenate(parts, space)` and
`scope::product(parts)`.

- `Space::domain()` is how the normal tells a global chart from one that is not
  ([Why global charts](math.svx#why-global-charts)); it errs for a discrete space and for one
  without a measure, which the normal passes on.
- Every step of the term is already an operation with derivatives of both orders. The normal writes
  no derivative.

### Markers, constants and the space of a handle

`crates/skoro-core/src/q.rs`, `crates/skoro-core/src/scope.rs`:

```rust
pub trait Typed: Copy + Send + Sync + 'static { fn admits(space: &Space) -> bool; }
pub trait Declares: Typed { fn space(&self) -> Space; }

pub fn quantity<T: Declares>(address: impl Into<Address>, marker: T) -> Q<T>;
pub fn quantity_in(address: impl Into<Address>, space: Space) -> QDyn;
pub fn constant<T: Declares>(marker: T, value: impl IntoHostValue<T>) -> Q<T>;
pub fn apply(operation: Arc<dyn Operation>, inputs: &[QDyn]) -> QDyn;
```

- No public function gives the space of a handle inside a scope. The normal learns the scored space
  the way a model makes a quantity: from a marker (`Declares::space`) or from a `Space`. A product,
  whose marker `ProductDyn` declares nothing, is given by its `Space`.
- A constant of an arbitrary `Space` cannot be made from a host value; one of a `Declares` marker
  can. The stacked tangent's anchors are therefore given with their markers.

### Refusing an input

`crates/skoro-core/src/operation.rs` and `error.rs`:

```rust
pub trait Operation: Debug + Send + Sync + 'static {
    fn name(&self) -> &'static str;
    fn params(&self) -> Vec<Param>;
    /// Checks the inputs' spaces and gives the output's. Called when the node is added.
    fn output(&self, inputs: &[Space]) -> Result<Space, Error>;
    fn lower(&self, inputs: &[Space], lowering: &mut Lowering<'_>) -> Result<Lowered, Error>;
}

#[non_exhaustive]
pub enum Error {
    /// An operation does not take inputs of these spaces, or of this many.
    SpaceMismatch { operation: &'static str, inputs: Vec<Space> },
    NoVolume { space: Space, quantity: Option<NodeId> },
    NotContinuous { space: Space },
    ValueShape { space: Space },
    /* … */
}
```

- A feature crate records a failure only through a building function: `scope::apply` of an
  operation whose `output` errs gives a failed handle and records the error at the caller's line.
  The normal refuses a space so, with the core's own variants: `SpaceMismatch` names the normal as
  the operation that does not take a quantity of that space, which is what a chart that is not
  global is to it. No new error is needed.

### Composites and programs

`crates/skoro-core/src/composite.rs` and `compile.rs`:

```rust
impl Composite {
    pub fn define(name: &'static str, params: Vec<Param>, inputs: &[Space],
                  body: impl FnOnce(&[QDyn]) -> QDyn) -> Result<Arc<Composite>, Failure>;
    pub fn call(self: &Arc<Self>, inputs: &[QDyn]) -> QDyn;
    /// A program of the composite alone, whose roots are its inputs.
    pub fn compile(&self, target: &Target) -> Result<Program, Error>;
}
```

- A composite is defined over exact input spaces, and a call refuses inputs of other spaces. The
  normal's composites are defined per form and dimension, so a scale of the wrong form or size is
  refused by the call itself.
- `Composite::compile(&Target::host())` gives a program the sampler can run in `draw`, on the host,
  in double precision, as [The sampler](math.svx#the-sampler) has every draw computed. `Program` is
  `Send + Sync`: the core runs one inside an `Arc` as a `Forward` kernel, which is
  `Send + Sync + 'static`.

### Keys

`crates/skoro-core/src/key.rs`:

```rust
impl Key {
    /// `n` variates in the open interval (0, 1), from the extendable output keyed by this key.
    pub fn uniforms(&self, n: usize) -> Vec<f64>;
}
```

- The draw of a row spends $\dimension$ variates of its key, the $i$-th through the quantile into
  $\zeta_i$, as the mathematics fixes.

### Retirement

`crates/skoro-core/src/retire.rs`:

```rust
pub struct Replacement {
    pub log_density: Q<Real>,
    pub proposals: Vec<(QDyn, Proposal)>,
}
```

- The core adds a replacement as a likelihood factor itself, and gives back no handle on it. A
  replacement built with the stacked tangent can therefore bind no roles (see
  [Open questions](#open-questions)); built outside a retirement, through `factor`, it can.

### Tests of the umbrella

`crates/skoro/tests/common/sphere.rs` declares the acceptance-test sphere with its stereographic
volume, and `crates/skoro/tests/sphere.rs` the guarded exponential sphere with a volume `within`
the ball of radius $\pi$, private to that file.

- The curved tests need both spheres, which are test code of the umbrella crate. They are written
  there, with `skoro-normal` a development dependency of `skoro`, and the exponential sphere moves
  from `tests/sphere.rs` to `tests/common/sphere.rs`, unchanged, so that both files use it.

## Design

### The crate

`crates/skoro-normal`, a member of the workspace, depending on `skoro-core` and `skoro-tensor` and on
nothing else, built only on the core's public interface (charter, position 12). The umbrella `skoro`
does not re-export it: the charter has the umbrella re-export the two crates of the core, and every
feature is a crate of its own. `skoro` takes it as a development dependency, for its tests.

```
crates/skoro-normal/src/
    lib.rs          pub use of the items below
    scale.rs        Form, Scale, triangular(), the constant helpers, host Cholesky
    composites.rs   the cached composites: whiten, normalizer, root, triangular matrix
    normal.rs       Normal: on, in_space, score, log_density, factor_at
    sample.rs       NormalSample, the quantile
    stacked.rs      Stacked
    roles.rs        the role keys
    refuse.rs       Refused, the operation that records a refusal
```

The archive's `skoro-normal` had a `Normal` of a mean and a lower-triangular covariance factor of a
type `Tril` of its own, a sampler drawing from burn's generator, and a `JointNormal` beside it. Kept:
the crate, the binding of residual, mean and scale apart, the information root as a binding, and
helpers for constant scales. Not kept: `Tril`, replaced by the core's product
$\positive^\dimension \times \R^{\dimension(\dimension-1)/2}$
([A triangular factor as a quantity](math.svx#the-scale)); burn's generator, which the charter's keys
replace; the single covariance-factor form, which the four forms replace; the term without the
log-volume; and `JointNormal`, which the stacked tangent and the normal on $\R^D$ replace
([As a likelihood factor](math.svx#as-a-likelihood-factor)).

### The scale

```rust
/// Which of the four forms of the scale a value is, read through W in every one.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum Form {
    /// σ ∈ ℝ>0, Σ = σ²I: a `Positive(1)`.
    Isotropic,
    /// σ ∈ ℝ>0^d, Σ = diag(σ)²: a `Positive(d)`.
    Diagonal,
    /// L with Σ = LLᵀ: a `triangular(d)`.
    CovarianceFactor,
    /// W with Λ = WᵀW: a `triangular(d)`.
    Whitening,
}

/// A scale: its form and the node of its value, a constant or any value of the form's space.
#[derive(Clone, Copy, Debug)]
pub struct Scale {
    form: Form,
    value: QDyn,
}

/// The space of a lower-triangular factor with a positive diagonal: the diagonal, then the entries
/// strictly below it column by column, `Product([Positive(d), Vector(d(d-1)/2)])`, and
/// `Product([Positive(1)])` when `d = 1`.
pub fn triangular(d: usize) -> Space;

impl Scale {
    pub fn isotropic(sigma: Q<Positive<1>>) -> Scale;
    pub fn diagonal(sigma: impl Into<QDyn>) -> Scale;          // Positive<N> or PositiveDyn
    pub fn covariance_factor(l: impl Into<QDyn>) -> Scale;     // of triangular(d)
    pub fn whitening(w: impl Into<QDyn>) -> Scale;             // of triangular(d)

    // Constants, built in the open scope.
    #[track_caller] pub fn sd(sigma: f64) -> Scale;                  // Isotropic
    #[track_caller] pub fn sds(sigma: &[f64]) -> Scale;              // Diagonal
    #[track_caller] pub fn covariance(sigma: &[f64], d: usize) -> Scale;  // CovarianceFactor
    #[track_caller] pub fn precision(lambda: &[f64], d: usize) -> Scale;  // Whitening

    pub fn form(&self) -> Form;
    pub fn value(&self) -> QDyn;
}
```

The helpers take dense matrices row by row and factor them on the host, in double precision:
`covariance` by the Cholesky factorization $\Sigma = L L\transpose$, and `precision` by the factor
$W$ lower triangular with $\Lambda = W\transpose W$, computed as $W = J M\transpose J$ from the
Cholesky factor $M$ of $J \Lambda J$, $J$ the exchange matrix. A matrix that is not symmetric
positive definite gives a failed handle with `ValueShape` of the form's space. The factorization is
a dozen lines of the crate's own, for the small matrices a scale is; faer, which the charter names
for the estimators' solves, would be a dependency for nothing else.

A triangular value is laid out as [](eq:archive/normal/math/root)'s rows and the math's
coordinates read it: the positive factor holds the diagonal, whose tangent coordinates are
$\log L_{aa}$ or $\log W_{aa}$, and the vector holds $L_{ab}$ or $W_{ab}$ for $a > b$, column $b$
after column $b - 1$.

### The composites

`composites.rs` defines, once per form and dimension, and caches in a `static` map keyed by
`(Form, d, Part)` behind a `Mutex`, as the core caches a space's retraction:

| composite | inputs | output | computes |
| --- | --- | --- | --- |
| `skoro.normal.whiten` | `Vector(d)`, the form's space | `Vector(d)` | $r = W \tangent$ |
| `skoro.normal.normalizer` | the form's space | `Real` | $\sum_i \log W_{ii} - \tfrac{\dimension}{2} \log 2\pi$ |
| `skoro.normal.root` | the form's space | `Matrix(d + p, 2d + k)` | $C$ of [](eq:archive/normal/math/root) |
| `skoro.normal.matrix` | `triangular(d)` | `Matrix(d, d)` | the factor as a dense lower-triangular matrix |

with `params` `[Text(form), Integer(d)]`, $p = \dimension(\dimension+1)/2$, and $k$ the dimension of
the form's space: $1$, $\dimension$, or $p$. Their bodies, from the table of
[The scale](math.svx#the-scale):

```rust
// skoro.normal.whiten, q = [t, scale]
match form {
    Form::Isotropic => q[1].recip().as_real().reshape(Space::Real) * q[0],
    Form::Diagonal => q[1].recip().as_real() * q[0],
    Form::Whitening => matrix(d).call(&[q[1]]).matmul(q[0]),
    // Forward substitution, eq:archive/normal/math/forward-substitution, one Real at a time:
    // z_i = (t_i - Σ_{j<i} L_ij z_j) · (1 / L_ii), the reciprocal of the positive diagonal entry.
    Form::CovarianceFactor => forward_substitution(q[1], q[0], d),
}

// skoro.normal.normalizer, q = [scale]
let log_diagonal = match form {
    Form::Isotropic => q[0].log().reshape(Space::Real) * -(d as f64),
    Form::Diagonal => -q[0].log().sum(),
    Form::CovarianceFactor => -q[0].factor(0).log().sum(),
    Form::Whitening => q[0].factor(0).log().sum(),
};
log_diagonal - 0.5 * d as f64 * (2.0 * PI).ln()
```

`skoro.normal.matrix` lays the factor out column by column with `scope::concatenate(columns,
Space::Matrix(d, d))`, each column $b$ the zeros above the diagonal (a constant `Vector(b)`), the
diagonal entry `factor(0).as_real().slice(b..b + 1)`, and the entries below it
`factor(1).slice(..)`. `skoro.normal.root` builds $C$ the same way: its first $\dimension$ rows are
$(W \;\, -W \;\, 0)$, and its scale columns are, for the isotropic and diagonal forms, the constants
$-\sqrt 2$ at the rows $(a, a)$ the table gives, and for the triangular forms the entries of
$T_i$ per coordinate, from $W$ and $L$, the one the form gives and the other by
[](eq:archive/normal/math/forward-substitution) against the columns of the identity. Its rows are
the $\dimension$ residual rows, then $(a, a)$ for $a = 1, \dots, \dimension$, then $(a, b)$ for
$a > b$ in the triangular storage's order; its columns are $\point$, $\mu$, then the scale's tangent
coordinates in the order of its space. The root of a triangular scale has $O(\dimension^4)$ nodes,
$p \times k$ entries each a sum over a $\dimension \times \dimension$ product; for the dimensions of
the states skoro is meant for, $\dimension \le 15$, that is a composite of tens of thousands of
nodes, compiled once and computed only by a query that targets it.

The alternative was one operation of the crate's own for the whole term, lowering to a composite
built for the spaces its `output` is given, which would have needed no marker. It lost because a
role binds a node of the model, and the residual, the log-volume and the root would be inside one
call; and because the sampler needs the space at build time all the same.

### The normal

```rust
/// The normal 𝒩(μ, Σ) on a space, by its mean and scale: a term to add, or a factor.
#[derive(Clone, Debug)]
pub struct Normal {
    space: Space,
    mean: QDyn,
    scale: Scale,
}

/// The nodes of a normal's term, each a node of the model.
#[derive(Clone, Copy, Debug)]
pub struct Parts {
    pub tangent: Q<VectorDyn>,        // x ⊖ μ
    pub residual: Q<VectorDyn>,       // W (x ⊖ μ)
    pub log_volume: Option<Q<Real>>,  // λ_μ(x ⊖ μ), None where it is identically zero
    pub log_density: Q<Real>,         // eq:archive/normal/math/term
}

impl Normal {
    /// The normal on the space `marker` declares.
    pub fn on<T: Declares>(marker: T, mean: Q<T>, scale: Scale) -> Normal;
    /// The normal on `space`, for a product or any handle without a declaring marker.
    pub fn in_space(space: Space, mean: impl Into<QDyn>, scale: Scale) -> Normal;

    /// Scores `x`: the term, its exact sampler and its roles. Refuses a space without a measure
    /// (`NoVolume`), a discrete one (`NotContinuous`) and one whose chart is not global
    /// (`SpaceMismatch` of `skoro.normal`).
    #[track_caller]
    pub fn score(&self, x: impl Into<QDyn>) -> TermHandle;

    /// The term's nodes at `x`, added to the model with no term: what `score` scores.
    #[track_caller]
    pub fn log_density(&self, x: impl Into<QDyn>) -> Parts;

    /// The Gaussian part at `z` as a likelihood factor, scoring nothing, with the residual, mean
    /// and scale bound: `-½|W (z ⊖ μ)|² + Σ log W_ii - (d/2) log 2π`. Needs no measure and no
    /// global chart.
    #[track_caller]
    pub fn factor_at(&self, z: impl Into<QDyn>) -> TermHandle;
}
```

`log_density` builds, in the open scope,

```rust
let t = x.local(self.mean);                                   // the core's ⊖
let r = whiten(form, d).call(&[t, scale]);
let gaussian = (r * r).sum() * -0.5 + normalizer(form, d).call(&[scale]);
let lambda = (!euclidean(&space)).then(|| self.mean.log_volume(t));
let log_density = match lambda { Some(l) => gaussian - l, None => gaussian };
```

where `euclidean(space)` is true of `Real`, `Vector` and `Matrix` and of products of them alone,
whose log-volume is identically zero. Every other space binds its log-volume, a flat embedded type's
included, whose node is the zero its declaration gives.

`score` checks the space first, `space.domain()`: `Ok(Domain::Global)` goes on; `Ok(Within(_))`
refuses with `SpaceMismatch { operation: "skoro.normal", inputs: vec![space] }`, and an `Err` is
refused as it is, both through `refuse.rs`:

```rust
/// An operation that is never added: its `output` gives the error it was made with, so that
/// `scope::apply` records it at the caller's line and gives a failed handle.
#[derive(Debug)]
struct Refused(Error);

impl Operation for Refused {
    fn name(&self) -> &'static str { "skoro.normal" }
    fn params(&self) -> Vec<Param> { Vec::new() }
    fn output(&self, _: &[Space]) -> Result<Space, Error> { Err(self.0.clone()) }
    fn lower(&self, _: &[Space], _: &mut Lowering<'_>) -> Result<Lowered, Error> {
        Err(self.0.clone())
    }
}
```

and `score` then passes the failed handle to `score`, so the term handle it returns is failed and
the scope's failure names the user's line. Otherwise it builds `log_density(x)`, then

```rust
let sample = NormalSample::new(space.clone(), form, d)?;   // a failure here is refused likewise
scope::score(x, parts.log_density)
    .sampler(Sampler {
        sample: Arc::new(sample),
        inputs: vec![self.mean, self.scale.value()],
        log_density: None,
    })
    .bind(&roles::RESIDUAL, parts.residual)
    .bind(&roles::MEAN, self.mean)
    .bind(roles::scale(form), self.scale.value())
    .bind(&roles::INFORMATION_ROOT, root(form, d).call(&[self.scale.value()]));
// and .bind(&roles::LOG_VOLUME, l) when parts.log_volume is Some(l)
```

The sampler's inputs are the mean and the scale, which the term is conditioned on, as `sampler`
requires. A scale that is a constant is folded by the core with every value reached only from it,
its whitening, its normalizer and its root among them ([The scale](math.svx#the-scale)).

`factor_at` builds the Gaussian part at `z` the same way, `z.local(mean)` for the tangent, and
adds it with `scope::factor`; it checks only that the space is continuous, which the core's `local`
does. Its normalizer is the Gaussian one: with a fixed scale it is a constant of every query, and
with a free scale it is not the normalizer of any law, which its documentation says
([Without a volume or a global chart](math.svx#without-a-volume-or-a-global-chart)).

### The sampler

```rust
/// Draws μ ⊕ L Φ⁻¹(u) on the host, from d variates of each row's key.
#[derive(Debug)]
pub struct NormalSample {
    space: Space,
    form: Form,
    dimension: usize,
    /// The space's retraction, compiled for the host when the sampler is made.
    retraction: Arc<Program>,
}

impl NormalSample {
    fn new(space: Space, form: Form, dimension: usize) -> Result<NormalSample, Error> {
        let retraction = space.retraction()?.compile(&Target::host())?;
        Ok(NormalSample { space, form, dimension, retraction: Arc::new(retraction) })
    }
}

impl Sample for NormalSample {
    fn name(&self) -> &'static str { "skoro.normal" }
    fn params(&self) -> Vec<Param> {
        vec![Param::Space(self.space.clone()), Param::Text(self.form.name().into())]
    }
    fn draw(&self, inputs: &[Value], keys: &[Key], target: &Target) -> Value {
        // On the host, in double precision, row by row:
        //   ζ = quantile(key.uniforms(d)),            Wichura's AS 241
        //   t = L ζ  for the form's L:   σ ζ, diag(σ) ζ, L ζ, or the solution of W t = ζ by
        //            eq:archive/normal/math/forward-substitution;
        // then μ ⊕ t by running `retraction` on the host values of μ and t, and the result loaded
        // on `target`, the one cast of the draw.
    }
}

/// The standard normal quantile, by Wichura's algorithm AS 241 (PPND16).
pub fn quantile(p: f64) -> f64;
```

Running the space's own retraction keeps the sampler one function on every space: on $\R^n$ it adds,
on the positive orthant it multiplies by the exponential, on the sphere it is the type's composite.
The alternative, host arithmetic per kind of space, would duplicate the core's retractions and miss
every embedded type. `draw` reads the mean's value from the target to the host, so on a target of
single precision the mean is the rounded one, and the same key gives the same draw at either
precision up to that rounding and the final cast, as the charter's position 11 allows.

### The roles

```rust
pub mod roles {
    /// The whitened residual W (x ⊖ μ), or W (z ⊖ μ) of a factor.
    pub const RESIDUAL: Role<ExactlyOne> = Role::new("skoro.residual");
    pub const MEAN: Role<ExactlyOne> = Role::new("skoro.normal.mean");
    /// The scale as given, one role per form: exactly one of the four is bound.
    pub const ISOTROPIC: Role<AtMostOne> = Role::new("skoro.normal.scale.isotropic");
    pub const DIAGONAL: Role<AtMostOne> = Role::new("skoro.normal.scale.diagonal");
    pub const COVARIANCE_FACTOR: Role<AtMostOne> = Role::new("skoro.normal.scale.covariance_factor");
    pub const WHITENING: Role<AtMostOne> = Role::new("skoro.normal.scale.whitening");
    /// λ_μ(x ⊖ μ), where it is not identically zero.
    pub const LOG_VOLUME: Role<AtMostOne> = Role::new("skoro.log_volume");
    /// The root C of eq:archive/normal/math/root, on terms that score a quantity.
    pub const INFORMATION_ROOT: Role<AtMostOne> = Role::new("skoro.information_root");

    pub fn scale(form: Form) -> &'static Role<AtMostOne>;
}
```

The scale's form is carried by which role binds it, so a reader learns it from a lookup alone, as
[The roles](math.svx#the-roles) asks. The residual, the log-volume and the information root are
named without the normal's prefix: they mean the same of any distribution with a whitened residual
or a closed-form information, and a reader written for them matches by name. The root is bound in
full whatever the scale is; for a constant scale the reader's Jacobian of the scale's coordinates
is zero, and the scale columns contribute nothing, which is the math's first block row. A
likelihood factor binds the residual, the mean and the scale, and no root.

### The stacked tangent

```rust
/// The stacked tangent δ of eq:archive/normal/math/stacked, block by block, each at a constant
/// anchor the builder makes itself.
#[derive(Clone, Debug, Default)]
pub struct Stacked {
    blocks: Vec<QDyn>,
    dimension: usize,
}

impl Stacked {
    pub fn new() -> Stacked;
    /// Appends `x ⊖ p`, `p` the constant `anchor` of `marker`'s space. Refuses a discrete space
    /// (`NotContinuous`, by the core's `local`).
    #[track_caller]
    pub fn block<T: Declares>(self, x: Q<T>, marker: T, anchor: impl IntoHostValue<T>) -> Stacked;
    /// δ, a `Vector(D)`.
    #[track_caller]
    pub fn tangent(&self) -> Q<VectorDyn>;
    /// ℓ_N(δ; m, W) + c, eq:archive/normal/math/joint, as a log-density with no term: what a
    /// `Replacement` holds.
    #[track_caller]
    pub fn log_density(&self, mean: &[f64], scale: Scale, constant: f64) -> Q<Real>;
    /// The same, added as a likelihood factor with its roles bound.
    #[track_caller]
    pub fn factor(&self, mean: &[f64], scale: Scale, constant: f64) -> TermHandle;
}
```

`log_density` and `factor` are `Normal::in_space(Space::Vector(D), constant(VectorDyn(D), mean),
scale)` at `tangent()`, with `constant` added: on $\R^D$ the log-volume is zero and the residual is
$W(\delta - m)$, so nothing new is computed. An anchor is a constant by construction, made from a
host value with its marker, which is how the builder keeps the condition
[The stacked tangent](math.svx#as-a-likelihood-factor) asks without inspecting the model. A blanket
quantity of a product type is given factor by factor, `x.factor(i)` with each factor's marker, since
the product's inverse retraction is its factors' stacked.

## Tests

In `crates/skoro-normal/tests/`, on the flat-in-disguise spaces, at both precisions of the Flex
device where a value is compared, with `testing::Tolerance`:

- `term_is_the_closed_form`: on `Real`, `Vector(3)` and `Matrix(2, 2)`, for each form of a scale of
  known $\Sigma$, the log-density at several points equals $\log \mathcal N(\point; \mu, \Sigma)$
  computed directly on the host; on `Positive(2)` and a product of `Vector(2)` and `Positive(1)`, it
  equals the Gaussian of $\point \ominus \mu$ less $J(\point)$ [](eq:archive/normal/math/jacobian).
  Shows that [](eq:archive/normal/math/term) is built as written, the log-volume of the core's types
  included.
- `four_forms_agree`: one $\Sigma$ given in the four forms (σ for an isotropic $\Sigma$, then a
  diagonal one in three forms, then a full one as factor and as whitening) gives the same
  log-density, residual, draws from one key, and $C\transpose C$. Shows the forms are maps to one
  $W$, as [The scale](math.svx#the-scale) says.
- `forward_substitution_solves`: the covariance factor's residual equals $L^{-1}(\point \ominus \mu)$
  by a dense solve on the host, and its derivatives with respect to $L$, by the passes, agree with
  finite differences. Shows [](eq:archive/normal/math/forward-substitution) and its derivatives.
- `composites_pass_check_operation`: `testing::check_operation` on every composite of
  `composites.rs`, per form, $\dimension \in \{1, 2, 3\}$: values, tangent and cotangent maps and
  Hessian products agree with finite differences. Shows the term has order two
  ([Properties](math.svx#properties), *Derivatives*).
- `density_integrates_to_one`: on `Real` and `Positive(1)`, a quadrature of
  $\exp \logterm_{\mathcal N}$ against Lebesgue measure is $1$. Shows
  [%full](statement:archive/normal/math/density) on the core's spaces.
- `constant_scale_helpers`: `covariance` gives $L$ with $L L\transpose = \Sigma$, `precision` gives
  $W$ lower triangular with $W\transpose W = \Lambda$, and a matrix that is not positive definite is
  refused with `ValueShape`.
- `quantile_matches_reference`: `quantile` at published values of $\Phi^{-1}$, in the centre and in
  both tails, to $10^{-15}$.
- `draws_are_functions_of_key_and_address`: two generations with one key and address give the same
  draws, bit for bit; another address gives others; the draw of a row equals $\mu \oplus L
  \Phi^{-1}(u)$ computed on the host from that row's key. Shows the sampler is the mathematics',
  and keyed as the charter requires.
- `draws_have_the_normal_moments`: over $10^5$ rows, the sample mean and covariance of the draws on
  `Vector(3)` in each form are within five standard errors of $\mu$ and $\Sigma$; on `Positive(2)`,
  those of $\log \randomPoint$ are within five of $\log \mu$ and $\Sigma$. Shows the sampler is
  exact on the flat spaces.
- `exact_sampler_leaves_the_weight_to_the_likelihoods`: a chain of normals generated with a
  measurement observed has the log-weight of the measurement's term alone, as the core's
  `exact_samplers_cancel_in_the_weight` checks with its private normal. Shows `log_density: None`
  is honest.
- `same_key_same_draw_at_both_precisions`: the draws at single and double precision differ by the
  rounding of the mean and of the cast alone.
- `roles_are_bound`: a normal on `Vector(2)` binds the residual, the mean, the scale under its form's
  role and the root, and no log-volume; one on `Positive(2)` binds the log-volume too; a factor binds
  the residual, the mean and the scale and no root. Shows [The roles](math.svx#the-roles).
- `root_squares_to_the_information`: on `Vector(2)` and `Positive(2)`, in each form of a free scale,
  $C\transpose C$ equals the expected negative chart Hessian of the term over
  $(\point, \mu, \theta)$, estimated over $10^5$ draws within five standard errors, and its scale
  block equals $F$ of the table. Shows [%full](statement:archive/normal/math/information) and
  [](eq:archive/normal/math/root).
- `refuses_spaces_it_does_not_take`: `score` of a `Categorical(3)` quantity fails with
  `NotContinuous`; the failure names the caller's line.

In `crates/skoro/tests/`, where the spheres are:

- `normal.rs::sphere_density_integrates_to_one`: on the acceptance-test sphere, for an isotropic and
  an anisotropic scale and a mean near the south pole and far from it,
  $\int_{\R^2} \exp(\logterm_{\mathcal N}(\mu \oplus \tangent) + \lambda_\mu(\tangent))\, \dd \tangent = 1$
  by quadrature in polar coordinates $r = 2 \tan u$, $u \in [0, \pi/2)$. Shows
  [%full](statement:archive/normal/math/density) on a curved space, the log-volume of the
  declaration included.
- `normal.rs::sphere_term_reads_the_declared_volume`: the term at points of the sphere equals the
  Gaussian of $\point \ominus \mu$ less $-2 \log(1 + \lVert \point \ominus \mu \rVert^2/4)$, and the
  log-volume's role binds that node.
- `normal.rs::sphere_sampler_matches_the_density`: expectations of three bounded test functions over
  $10^5$ draws agree, within five standard errors, with their quadratures against the density.
  Shows the sampler exact on a curved space.
- `normal.rs::sphere_information`: over draws, the scale's block of the expected negative Hessian
  equals $F$, and the blocks of $\point$ and $\mu$ approach $\pm\Lambda$ as the scale shrinks, their
  relative error falling as $\sigma^2$. Shows [On a curved space](math.svx#the-expected-information).
- `normal.rs::gauss_newton_reaches_the_model_mode`: two measurements of a direction with different
  isotropic scales; Gauss–Newton with the log-volume's gradient converges to the mode of the full
  log-density, found by a dense search, and without it to the mode of the residuals alone, the two
  differing by more than the tolerance; the observed rate is within $\kappa = \sigma^2$ of the
  residuals' own. Shows [%full](statement:archive/normal/math/gauss-newton).
- `normal.rs::refuses_a_chart_that_is_not_global`: `score` of a quantity of the exponential sphere
  fails with `SpaceMismatch` of `skoro.normal`; `factor_at` on it is accepted.
- `normal.rs::refuses_a_type_without_a_volume`: on the sphere declared without a volume, `score`
  fails with `NoVolume`, and `factor_at` at a constant measurement is accepted, its gradient in the
  mean that of $-\tfrac12 \lVert W(z \ominus \mu) \rVert^2$.
- `normal.rs::stacked_tangent_on_the_sphere`: a stacked tangent of a sphere and a real at constant
  anchors is their tangents laid end to end, and its factor's residual is $W(\delta - m)$.
- `retirement.rs::replacements_written_with_stacked`: the two replacements of the core's retirement
  tests, written with `Stacked::log_density`, give the same densities as the hand-written ones at
  the points those tests use. Shows [](eq:archive/normal/math/joint) is the core's replacement.

The exponential sphere moves to `tests/common/sphere.rs` for the refusal test, and the sphere
declared without a volume is the acceptance test's, declared under another name without
`with_volume`.

## Open questions

> [!TODO] Roles of a replacement
> `Retirement::replace` adds the replacement's factor itself and gives back no handle, so a
> replacement written with `Stacked` binds no roles, while the same factor added with `factor`
> does. A least-squares or Fisher-scoring reader of a model with retirements would then find no
> residual on the replacements. Either the core's `Replacement` gains bindings, which is a
> revision of the core and an entry of its own, or the replacement's roles wait for the elimination
> entry that computes replacements, which revises the core then. This design takes the second and
> documents the gap; is that acceptable?
