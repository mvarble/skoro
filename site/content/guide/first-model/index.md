---
title: A first model
summary: Write a hierarchical normal with `sample`, generate from it and take its density and gradient given an observation, then see what the sugar is made of, from the tests of skoro itself.
order: 1
---

This page starts from a model as it is usually written in skoro, with a distribution and the
function `sample`, and works down from it. It first generates from the model and conditions it on
an observation. It then shows the pieces underneath, each a little more general than the last:
programs of plain arithmetic, the same model without the sugar, and terms written by hand beside
likelihood factors. Each step quotes skoro's own tests, so every line shown compiles and runs. The
[concepts](../../concepts/) explain each piece from the ground up; the links below go there when a
step needs more than it shows.

## A hierarchical model

A point $x$ of the plane is drawn about the origin, and its squared range $\rho$ is measured with
noise:

$$
x \sim \mathcal N(0, \sigma_0^2 I_2), \qquad \rho \mid x \sim \mathcal N(|x|^2, \sigma_1^2).
$$

In skoro the model reads as it is written above:

```rust
{../../../../crates/skoro/tests/guide.rs:17..41}
```

A model is built inside the scope a [writer](../../concepts/building/) opens, and published as a
snapshot when the closure returns. `sample(address, marker, distribution)` makes a quantity at an
address, of the space the marker names, and scores it by the distribution. Here the distribution
is [the normal](../../concepts/the-normal/), from the crate `skoro-normal`: `Normal::on` names its
space, its mean and its scale, here a standard deviation. The mean of $\rho$ is computed from $x$
by ordinary arithmetic on its handle, `(x * x).sum()`, and is a value of the model like any other.
Nothing names the writer inside the closure: everything goes into whichever model's scope is open.

## Generate from the joint

Every quantity the normal scores comes with a sampler that draws it exactly, so the model can be
generated: each quantity drawn in the order it was made, from its distribution given what was drawn
before it, one row per key:

```rust
{../../../../crates/skoro/tests/guide.rs:123..129}
```

`generate` takes the observations, here none, a random key, the number of rows, and a target. It
gives the values drawn and an importance weight for each row; with nothing observed and exact
samplers, every row weighs the same. The test goes on to check that the draws of $x$ and of
$\rho - |x|^2$ have the means and variances the model gives them.

## Condition on an observation

A [query](../../concepts/queries/) names what to compute and what is observed. With $\rho$ observed,
the model's log-density is the joint density of $x$ and the observation, the posterior of $x$ up to
a constant:

$$
\log p(x, \rho) = -\frac{|x|^2}{2\sigma_0^2} - \frac{(\rho - |x|^2)^2}{2\sigma_1^2}
- 2 \log \sigma_0 - \log \sigma_1 - \tfrac32 \log 2\pi .
$$

Compiling the query from the snapshot, for a target, gives a program, and one pass of it gives the
density and its gradient in $x$ at three points at once:

```rust
{../../../../crates/skoro/tests/guide.rs:141..149}
```

The value is the closed form above, and the gradient is
$-x / \sigma_0^2 + 2x(\rho - |x|^2) / \sigma_1^2$, which the test's own functions compute by hand.
The test runs once at single precision and once at double on the CPU, and on a GPU through wgpu
when that feature is on: `target` is each in turn. The model never changes; only the target does.

## Programs of any value

A query need not be of a density. Any value built in a scope can be compiled and differentiated,
which is how the density above was: here a model has one quantity and no terms, and `f` is
$\sum_i (x_i^2 + \sin x_i + x_i)$, computed by the core's
[primitive operations](../../concepts/operations/#the-primitives):

```rust
{../../../../crates/skoro-core/tests/passes.rs:287..293}
```

The test's helper compiles a query of the nodes it is given, from the writer's latest snapshot:

```rust
{../../../../crates/skoro-core/tests/passes.rs:132..137}
```

The program is run at a value of `x`, one row of two entries, and its gradient compared with the
derivative worked out by hand, $2 x_i + \cos x_i + 1$:

```rust
{../../../../crates/skoro-core/tests/passes.rs:294..299}
```

`untyped` drops the static type `Vector<2>` of a handle, which this test does not need.

## The model without the sugar

`sample` does two things, and the hierarchical model can be written with each on its own. A
quantity is made at an address with `quantity`, and the distribution's `score` adds the
[term](../../concepts/terms/) that scores it, a log-density, with its sampler:

```rust
{../../../../crates/skoro/tests/guide.rs:43..57}
```

It builds the same model, and its density with $\rho$ observed is the same at every point. The two
statements are what a model writes when it wants the handle of the term `score` gives back, to bind
a role of its own on it or to give it another sampler; otherwise the term is found from the
quantity it scores.

## Terms written by hand

A term need not come from a distribution at all. The core's own `score(quantity, log_density)`
adds any log-density built from the model's values as the term of a quantity, and `factor` adds a
**likelihood factor**, which scores nothing: the likelihood of data that is not a quantity of the
model. Here a vehicle at $x_0$ and then $x_1$ measures a landmark $m$ from each position, its
motion driven by an inertial reading $\zeta$, and a second motion model is a likelihood factor over
$x_0$, $x_1$ and a scale $\sigma$:

```rust
{../../../../crates/skoro-core/tests/queries.rs:31..57}
```

$\zeta$ and $\sigma$ are scored by no term: they are parameters of the model, such as a sensor
reading it is driven by. `Query::log_density()` is the sum of every term. Compiled and run with a
value for each quantity, it equals the sum of the seven terms computed one at a time:

```rust
{../../../../crates/skoro-core/tests/queries.rs:111..128}
```

With some quantities observed, by `Query::log_density().given(observations)`, the same model gives
the density of the others: the posterior of the vehicle's positions and the landmark given the
measurements, up to a constant, whose gradient is taken just as above, in the
[tangent charts](../../concepts/queries/#passes) of the free quantities.

## A curved quantity

The normal is written the same way on a curved space. Here two directions on a sphere measure an
unknown direction $x$, each with its own scale; a query observes them and estimates $x$, or leaves
them free and simulates them:

```rust
{../../../../crates/skoro/tests/normal.rs:435..448}
```

The [concepts page on the normal](../../concepts/the-normal/) says how its term, sampler and scale
are made, and which spaces it takes.
