---
title: The normal distribution
summary: skoro's normal is the law of a point moved by normal noise through its space's chart, on every space whose chart is global, with an exact density and sampler, a scale in four forms, and roles that estimators read.
order: 6
depends_on: [concepts/terms, concepts/spaces, chart-volume]
---

The normal is the first distribution of skoro, and lives in a crate of its own, `skoro-normal`,
built on the core's public interface alone. It is one distribution on every space it can be exact
on: the familiar normal on the real line, vectors and matrices, the log-normal on the positive
orthant, and on a curved type, such as a sphere or the rotations, the normal of its tangent chart
pushed onto it. This page shows what it is made of: a [term](../terms/), a sampler, and the roles an
estimator reads.

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

where $W$, the *whitening*, is the lower-triangular factor with $W\transpose W = \Sigma^{-1}$, and
$W (\point \ominus \mu)$ is the *whitened residual*. The note on
[densities in tangent charts](../../math/chart-volume/) proves that this is the density of
$\mu \oplus \varepsilon$. A normal is made from its space, its mean and its scale:

```rust
{../../../../crates/skoro-normal/src/normal.rs:25..31}
```

```rust
{../../../../crates/skoro-normal/src/normal.rs:46..59}
```

`Normal::on` reads the space from the marker a quantity was made with; `Normal::in_space` takes the
space itself, for a product, whose marker declares none. The term is built from the core's
operations, as any value of a model is: the tangent by the space's inverse retraction, the residual
and the normalizer by composites of the scale's form, and the log-volume by the core's:

```rust
{../../../../crates/skoro-normal/src/normal.rs:147..173}
```

```rust
{../../../../crates/skoro-normal/src/normal.rs:115..131}
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
{../../../../crates/skoro-normal/src/normal.rs:73..113}
```

It refuses what it cannot be exact on, at the caller's line, with the core's own errors: a discrete
space, a space without a reference measure, and a space whose chart is not global.

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

A term binds the nodes an estimator needs, under [roles](../terms/) a reader in another crate looks
up by name:

```rust
{../../../../crates/skoro-normal/src/roles.rs:10..25}
```

The scale is bound under the role of its form, so a reader learns the form from which is bound. The
root $C$ of the expected information has $C\transpose C$ equal to the expected negative Hessian of
the term over $(\point, \mu, \theta)$, exact on the core's spaces. On a curved type it is exact in
the scale's block and holds to second order in the dispersion elsewhere. The log-volume's role lets
a least-squares reader add its gradient, and so reach the model's own mode rather than that of the
residuals alone.

## As a likelihood factor

`factor_at(z)` adds the Gaussian part at a value $z$ as a likelihood factor, scoring nothing. It
needs no reference measure and no global chart, which is how a measurement is held on a type
without either. The stacked tangent builds the form of a retirement's Gaussian replacement, a
normal on $\R^D$ at the tangents of the blanket's quantities at constant anchors:

```rust
{../../../../crates/skoro-normal/src/stacked.rs:22..37}
```

```rust
{../../../../crates/skoro-normal/src/stacked.rs:50..59}
```
