---
title: The uniform distribution
summary: skoro's uniform distribution is the law of a space whose reference measure has finite mass, the rotations and the categorical spaces among them, with a constant term, an exact sampler on the host, and a residual of no components, so that a linearization reads it as a factor with no rows; it is how a model says that nothing is known of a quantity.
order: 8
depends_on: [concepts/spaces, concepts/terms, concepts/the-normal, uniform, improper-gaussian]
---

A model often has to say that nothing is known of a quantity before the data are seen: the
heading of a robot at start-up, the first pose of a graph, which of two hypotheses holds. On a
space whose reference measure $\nu$ has [finite mass](../spaces/#masses-and-uniform-points) there
is a law that says exactly that, the uniform distribution $\nu / \nu(\manifold)$. It lives in a
crate of its own, `skoro-uniform`, built on the core's public interface and on one role of
`skoro-normal`. A normal with a large scale is not a substitute on a curved type:
[the normal's page](../the-normal/#a-large-scale-on-a-curved-type) says why.

The spaces it takes are those of finite mass: `Categorical(n)`, `Boolean` among them, with counting
measure of mass $n$; an embedded type that declares its mass, such as the
[rotations](../rotations-and-motions/), of mass $8\pi^2$; and products of these. The note on
[the uniform distribution](../../math/uniform/) has the mathematics.

## The term

The uniform distribution has no parameters, only its space:

```rust
{../../../../crates/skoro-uniform/src/uniform.rs:26..50}
```

Its log-density is the constant $-\log \nu(\manifold)$. `score` adds it as the term of the quantity,
with an exact sampler and a residual of no components:

```rust
{../../../../crates/skoro-uniform/src/uniform.rs:52..86}
```

The term is a call of a composite of the quantity that does not read it. A call is a mapped node,
so it is a log-density, and each call is a node of its own, so two quantities have two terms; the
composite takes the distribution's space, so a quantity of another space is refused by the core's
check of an operation's inputs, with `SpaceMismatch` of `skoro.uniform.log_density`; and its body
is a constant, so its gradient and its Hessian products are zero, and it moves no mode and no step.
A space without a finite mass is refused with the core's `NoMass`, at the caller's line, as the
normal refuses what it cannot be exact on. A uniform is a [`Distribution`](../terms/#a-term) by
this `score`, so the core's `sample` makes a quantity and scores it by one in one call:

```rust
{../../../../crates/skoro-uniform/src/uniform.rs:89..94}
```

The uniform binds the normal's residual role, which means the same of any term with a whitened
residual, and no other: it has no mean, no scale, no tangent and no information to give a root of.

## The sampler

A uniform point is drawn on the host, from the variates of the quantity's random key: by the
space's declared draw, from as many variates as it has dimensions, or, on a categorical space, as
$\lfloor n u \rfloor$ from one variate $u$:

```rust
{../../../../crates/skoro-uniform/src/sample.rs:30..64}
```

The minimum keeps the rounding of $n u$ from ever giving $n$. The sampler is exact, so in a
generation each uniform term cancels against its draw and leaves the
[importance weight](../../math/importance-weights/) to the likelihoods. In the model of a heading
of which nothing is known, read by a compass, with a coin beside it, the weight of each drawn row
is the compass's log-density at the drawn heading, and neither $\log 8\pi^2$ nor $\log 2$ is in it:

```rust
{../../../../crates/skoro-uniform/tests/sample.rs:128..144}
```

## In a linearization

A [linearization](../linearization/) reads the residual of every term with a free quantity, and the
uniform's has no components. Its factor is the [improper Gaussian factor](../../math/improper-gaussian/)
with no rows, no linear part and the constant $-\log \nu(\manifold)$: flat in every direction,
exactly. It adds nothing to the rows a step is solved from, and its constant stays in the objective
and in the evidence of a Laplace approximation, so the linearizer takes it with no change of its
own. A heading under a uniform prior and read by a compass is estimated where the compass reads it.
A quantity that no other term bears on has no rows at all, and its elimination reports it
`Undetermined`, as it should: its posterior is uniform and has no mode. An observed categorical
quantity makes its term a constant of the data, and a free one is refused by a linearization as
any free discrete quantity is.

> [!DECISION] A crate of its own, and a term made of a composite
> The uniform is not a normal in any reading, so it is not a module of `skoro-normal`; it depends
> on that crate only for the residual role, whose meaning is shared. Its term is a composite of a
> constant rather than a primitive operation, which would need a kernel on every backend to
> compute a number, and a draw is a function of the host rather than a composite, since no pass
> crosses it. See [the implementation](../../archive/uniform/implementation.md#the-distribution).
