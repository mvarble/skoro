---
title: Steps and replacements
summary: Linearize a model's density at a point and iterate damped Gauss–Newton steps to its mode, and retire part of a model with a linearized replacement that keeps what it said, with the crate skoro-linearize, from its own tests.
order: 3
---

The crate `skoro-linearize` reads a model as least squares. This page shows what a user writes with
it: a Gauss–Newton loop that finds the mode of a model of rotations, and a retirement whose
replacement is computed from the retired terms rather than written by hand. Each step quotes the
crate's tests, so every line shown compiles and runs. The
[concepts page](../../concepts/linearization/) explains what each piece computes.

## A step to the mode

A model is linearized for a density query, which says what is observed. Here a chain of two
rotations is measured once, through the normal on the rotations, and a batch of two starting points
is given for the free rotations:

```rust
{../../../../crates/skoro-linearize/tests/common/mod.rs:249..286}
```

A `Linearizer` is built once for the snapshot, the query, the target and the assembly, here the
residual linearization of Gauss–Newton. Its schedule, the order and the junction tree of the
elimination, is computed once too. Each iteration linearizes at the current batch, eliminates every
row along the schedule, and retracts the step onto the quantities:

```rust
{../../../../crates/skoro-linearize/tests/linearize.rs:499..523}
```

The loop stops where the query's own gradient vanishes, log-volumes included: the mode of the
density, not of the residuals alone. Undamped, a step far from the mode can overshoot. The crate's
`LevenbergMarquardt` is the loop to use: it damps each row of the batch on its own, takes a row's
step only where it decreased the density's negative logarithm, and stops each row when a step
promises less than a tolerance in nats. Every row ends with a status, and `decrement` checks that
where it stopped is stationary:

```rust
{../../../../crates/skoro-linearize/tests/minimize.rs:82..99}
```

[The concepts page](../../concepts/linearization/#minimizing-the-levenberg-marquardt-method) says
how the damping and the stop are chosen. A schedule kept on some quantities, `schedule_keeping`,
gives their marginal covariance, and `at_in_chart` the linearization a Laplace approximation reads
its evidence and draws from.

## A replacement for what is retired

A retirement removes the quantities at some addresses and needs a replacement for the terms joined
to them. On the core's scalar chain, a position, its measurement, the next position and its
measurement:

```rust
{../../../../crates/skoro-linearize/tests/replacement.rs:23..41}
```

retiring the first tick is three steps: read the retirement, linearize the replacement at a point
of the quantities it integrates and keeps, and build it in the retirement's own scope:

```rust
{../../../../crates/skoro-linearize/tests/replacement.rs:80..95}
```

The quantity to keep is named, and the measurement retired with the first tick is observed, so the
replacement integrates the first position alone. For terms like these, linear and Gaussian, the
replacement is exact wherever it is linearized, and it is the Kalman filter's update and
prediction:

```rust
{../../../../crates/skoro-linearize/tests/replacement.rs:99..135}
```

The factor the retirement adds is the normal's
[improper factor](../../concepts/the-normal/#an-improper-factor), with its residual bound, so a
later linearization reads it as it reads any term: retiring one tick after another runs the filter.
Where the retired terms do not determine what is kept, as for a landmark seen along one bearing, the
replacement keeps what they say, and replacements over the same quantity add up until they do.
