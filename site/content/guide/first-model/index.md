---
title: A first model
summary: Build a model in a writer's scope, compile a query for a target, and take its value and gradient, from the core's own tests.
order: 1
---

This page walks through what a user of skoro writes today: a model built from quantities and
operations, a query compiled from it for a device and a precision, and a pass that gives a value and
its gradient. Each step quotes the core's own tests, so every line shown compiles and runs. The
[concepts](../../concepts/) explain each piece from the ground up; the links below go there when a
step needs more than it shows.

skoro ships no distributions yet, so the densities here are written out by hand, from the core's
arithmetic.

## Build a model

A model is built inside the scope a [writer](../../concepts/building/) opens. Inside it, a quantity
is made at an address, and ordinary arithmetic on its handle adds operations to the model:

```rust
{../../../../crates/skoro-core/tests/passes.rs:287..293}
```

`x` is an uncertain vector of $\R^2$, and `f` is $\sum_i (x_i^2 + \sin x_i + x_i)$, computed by the
core's [primitive operations](../../concepts/operations/#the-primitives). Neither handle names the
writer: the operations went into whichever model's scope was open. `untyped` drops the static type
`Vector<2>`, which this test does not need.

## Compile a query and take a gradient

A [query](../../concepts/queries/) names what to compute; compiling it from the writer's latest
snapshot, for a target, gives a program. The test's helper compiles a query of the nodes it is
given:

```rust
{../../../../crates/skoro-core/tests/passes.rs:132..137}
```

The program is then run at a value of `x`, one row of two entries, and its gradient compared with
the derivative worked out by hand, $2 x_i + \cos x_i + 1$:

```rust
{../../../../crates/skoro-core/tests/passes.rs:294..299}
```

The test runs once at single precision and once at double on the CPU, and on a GPU through wgpu
when that feature is on: `target` is each in turn. The model above never changes; only the target
does.

## A model with terms

A probabilistic model is made of [terms](../../concepts/terms/): log-densities that score the
quantities drawn from them, or that are likelihood factors over them. Here a vehicle measures a
landmark from two positions:

```rust
{../../../../crates/skoro-core/tests/queries.rs:31..57}
```

`Query::log_density()` is the sum of every term. Compiled and run with a value for each quantity,
it equals the sum of the seven terms computed one at a time:

```rust
{../../../../crates/skoro-core/tests/queries.rs:111..128}
```

With some quantities observed, by `Query::log_density().given(observations)`, the same model gives
the density of the others: the posterior of the vehicle's positions and the landmark given the
measurements, up to a constant. Its gradient is taken just as above, in the
[tangent charts](../../concepts/queries/#passes) of the free quantities.
