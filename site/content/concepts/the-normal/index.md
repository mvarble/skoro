---
title: The normal distribution
summary: skoro's normal is the law of a point moved by normal noise through its space's chart, on every space whose chart is global, with an exact density and sampler, a scale in four forms, roles that estimators read, its tangent among them, and the Gaussian and improper Gaussian replacements of a retirement.
order: 6
depends_on: [concepts/terms, concepts/spaces, chart-volume]
---

The normal is the first distribution of skoro, and lives in a crate of its own, `skoro-normal`,
built on the core's public interface alone. It is one distribution on every space it can be exact
on: the familiar normal on the real line, vectors and matrices, the log-normal on the positive
orthant, and on a curved type, such as a sphere or the [rotations](../rotations-and-motions/), the
normal of its tangent chart pushed onto it. This page shows what it is made of: a
[term](../terms/), a sampler, and the roles an estimator reads.

## The law and its term

The normal $\mathcal N(\mu, \Sigma)$ on a space is the law of $\mu \oplus \varepsilon$, the mean
moved by noise $\varepsilon$ normal in its tangent coordinates, with mean zero and covariance
$\Sigma$. Its log-density is the Gaussian log-density of $\point \ominus \mu$, less the
[log-volume](../spaces/#measures-and-chart-volumes) of the chart at $\mu$ there, which is what
makes it a density against the space's reference measure:

$$
	\ell(\point;\, \mu, W) = -\tfrac12 \bigl\lVert W (\point \ominus \mu) \bigr\rVert^2
	+ \sum_{i} \log W_{ii} - \tfrac{\dimension}{2} \log 2\pi - \lambda_\mu(\point \ominus \mu) ,
$$

where $W$, the _whitening_, is the lower-triangular factor with $W\transpose W = \Sigma^{-1}$, and
$W (\point \ominus \mu)$ is the _whitened residual_. The note on
[densities in tangent charts](../../math/chart-volume/) proves that this is the density of
$\mu \oplus \varepsilon$. A normal is made from its space, its mean and its scale:

```rust
{../../../../crates/skoro-normal/src/normal.rs:26..32}
```

```rust
{../../../../crates/skoro-normal/src/normal.rs:47..60}
```

`Normal::on` reads the space from the marker a quantity was made with; `Normal::in_space` takes the
space itself, for a product, whose marker declares none. The term is built from the core's
operations, as any value of a model is: the tangent by the space's inverse retraction, the residual
and the normalizer by composites of the scale's form, and the log-volume by the core's:

```rust
{../../../../crates/skoro-normal/src/normal.rs:151..177}
```

```rust
{../../../../crates/skoro-normal/src/normal.rs:119..135}
```

On a Euclidean space the log-volume is identically zero and left out. On the positive orthant it is
$\sum_i \log \point_i$, so the normal there is the log-normal with median $\mu$, offered under the
normal's name. On a curved type it is the one the type declares. Every step has derivatives of both
orders, so the term has them too, and a [query](../queries/) differentiates it like any other.

A measurement of a curved quantity is a normal with a computed mean: $z = h(\point) \oplus
\varepsilon$ is `Normal::on(marker, h, scale).score(z)`, and $z$ is a quantity, observed in one
query and drawn in another.

## Scoring a quantity

`score` adds the term, an exact sampler, and the roles:

```rust
{../../../../crates/skoro-normal/src/normal.rs:74..117}
```

It refuses what it cannot be exact on, at the caller's line, with the core's own errors: a discrete
space, a space without a reference measure, and a space whose chart is not global.

A normal is a [`Distribution`](../terms/#a-term) by this `score`, so the core's `sample` makes a
quantity and scores it by a normal in one call, `sample("x", Real, normal)`, building what
`quantity("x", Real)` and `normal.score(x)` build:

```rust
{../../../../crates/skoro-normal/src/normal.rs:180..186}
```

> [!DECISION] The normal requires a global chart
> On a chart that is a diffeomorphism only on part of the tangent space, such as the exponential
> chart of the rotations, noise beyond its domain wraps around, and no closed form makes the density
> and the sampler both exact. A type that carries normals takes a global chart, such as the Cayley
> chart of the rotations or the stereographic chart of a sphere; on any other, a measurement is a
> likelihood factor over data. See [the normal's mathematics](../../archive/normal/math.svx#why-global-charts).

## The scale

The scale is given in whichever of four forms a model has it, and each is read through $W$:

```rust
{../../../../crates/skoro-normal/src/scale.rs:8..19}
```

A standard deviation or a vector of them needs no linear algebra; a covariance factor $L$ is what a
user with a full covariance writes, and its residual is solved by forward substitution; a whitening
$W$ is what a precision, and a retirement's replacement, arrive in. When a scale is estimated, a
triangular factor is a quantity of the core's product of its positive diagonal and the entries
below it, column by column:

```rust
{../../../../crates/skoro-normal/src/scale.rs:40..53}
```

Constant scales have helpers, which factor a covariance or a precision on the host:

```rust
{../../../../crates/skoro-normal/src/scale.rs:88..114}
```

The composites that read the scale are defined once per form and dimension:

```rust
{../../../../crates/skoro-normal/src/composites.rs:80..105}
```

## The sampler

The normal draws $\mu \oplus L\, \Phi^{-1}(u)$, with $L = W^{-1}$ and $u$ the $\dimension$ uniform
variates of the quantity's random key, on the host in double precision. It moves the mean by
the space's own retraction, compiled for the host when the term is added, so that one sampler
serves every space:

```rust
{../../../../crates/skoro-normal/src/sample.rs:88..122}
```

Since the chart is global, the draw has exactly the term's density, and the sampler is exact: it
leaves the importance weight to the likelihoods.

## The roles

A term binds the nodes an estimator needs, under [roles](../terms/) a reader in another crate, such
as [`skoro-linearize`](../linearization/), looks up by name:

```rust
{../../../../crates/skoro-normal/src/roles.rs:11..31}
```

and, on a term that scores a quantity, its tangent at the mean:

```rust
{../../../../crates/skoro-normal/src/roles.rs:43..46}
```

The scale is bound under the role of its form, so a reader learns the form from which is bound. The
tangent says how the residual and the log-volume are made of it, which is what lets the
[linearization](../linearization/#folding-a-log-volume) fold a curved type's log-volume into its
rows, and what the Fisher rows of a free scale read their Jacobian from. The
root $C$ of the expected information has $C\transpose C$ equal to the expected negative Hessian of
the term over $(\point, \mu, \theta)$, exact on the core's spaces. On a curved type it is exact in
the scale's block and holds to second order in the dispersion elsewhere. The log-volume's role lets
a least-squares reader add its gradient, and so reach the model's own mode rather than that of the
residuals alone.

What such a reader takes from the log-volume's role is the varying part of the term's remainder,
what is left of its log-density after the squares of its residual and a normalizer: on a normal it
is the chart's log-volume, and on the [improper factor](#an-improper-factor) below its linear part.

> [!DECISION] The log-volume role holds the varying part of a remainder
> The improper factor binds its linear part under the log-volume role, which is not the volume of
> any chart, so that every least-squares reader reads one role for the varying part of a term's
> remainder. A reader that needs a quantity's chart volume reads it from the quantity's type, never
> from the roles of the terms around it. [Improper Gaussian replacements](../../archive/improper-gaussian/)
> decided it.

## As a likelihood factor

`factor_at(z)` adds the Gaussian part at a value $z$ as a likelihood factor, scoring nothing. It
needs no reference measure and no global chart, which is how a measurement is held on a type
without either. The stacked tangent builds the form of a retirement's Gaussian replacement, a
normal on $\R^D$ at the tangents of the blanket's quantities at constant anchors:

```rust
{../../../../crates/skoro-normal/src/stacked.rs:25..40}
```

A retirement adds its replacement as a likelihood factor itself, and the replacement carries the
roles that factor binds, as a term in a scope does:

```rust
{../../../../crates/skoro-core/src/retire.rs:86..103}
```

`Stacked::replacement` gives a retirement that replacement whole, its residual, mean and scale
bound under the normal's roles exactly as `Stacked::factor` binds them on a factor, through one
helper; the retirement checks each node and binds it to the factor it adds, in the snapshot it
publishes, so that a reader finds the residual of a retired window as it finds any other:

```rust
{../../../../crates/skoro-normal/src/stacked.rs:60..72}
```

```rust
{../../../../crates/skoro-normal/src/stacked.rs:111..119}
```

## An improper factor

A `Stacked` replacement is a normal, so it determines every direction of the blanket. A retirement
whose terms do not, such as a landmark seen by bearing from a single pose, which says in which
direction the landmark lies and nothing of how far, has as its replacement an _improper Gaussian
factor_ in the same stacked tangent $\delta$: a sum of squared affine rows of any number, a linear
part and a constant,

$$
	\logterm(\point_B) = -\tfrac12 \lVert A \delta - b \rVert^2 - \eta\transpose \delta + c ,
$$

which no normalization need make a density of $\delta$, and a likelihood factor never needed one.
Where its rows have full rank it is a normal, and its value is a `Stacked` replacement's; where they
do not, it keeps exactly what the retired terms said and nothing more. `Stacked::improper` builds it
from its rows, right-hand side and constant, each a constant of the model and each shared or given
row by row of a batch, and `linear` gives it a linear part:

```rust
{../../../../crates/skoro-normal/src/improper.rs:25..71}
```

Like `Stacked`, it gives a log-density alone, a retirement's `Replacement`, or a likelihood factor,
the last two binding its roles through one helper: the residual $A \delta - b$, of any number of
components, none included, and, where a linear part is given, $\eta\transpose \delta$ under the
log-volume role. It binds no mean and no scale, which a factor of lower rank does not have.

```rust
{../../../../crates/skoro-normal/src/improper.rs:118..127}
```

A blanket quantity the factor leaves undetermined is scored by no term after the retirement, and is
a parameter: generation draws it from a proposal given with the replacement, and refuses it
otherwise. Factors over the same quantity accumulate as terms of their own, and the model reads them
as their sum: once the bearings of two poses have been retired, along different directions, the
landmark is determined, at the point where the two lines cross. The
[linearization](../linearization/#a-retirement-s-replacement) computes such a replacement from the
retired terms themselves. The mathematics, its forms and its
examples are in [the entry](../../archive/improper-gaussian/math.svx).
