---
title: Queries and programs
summary: A query names what to compute from a snapshot and what is observed; compiled for a device and a precision, it becomes a program whose passes give values, tangents, gradients and Hessian-vector products in tangent charts.
order: 5
depends_on: [concepts/terms, derivatives-in-charts]
---

A model holds a joint distribution and computes nothing by itself. Computation starts with a
**query**: what to compute, and which quantities are observed. A query is compiled from a
[snapshot](../building/#snapshots-readers-and-branches) for a **target** (a device and a
precision) into a **program**, whose passes compute values and derivatives. This page describes
the three in turn.

## A query

```rust
{../../../../crates/skoro-core/src/query.rs:90..121}
```

A query's targets are some nodes of the model, or the log-density: the sum of every
[term](../terms/), or of some. Its **observations** give values to some quantities. Every quantity
a target is computed from and that is not observed is **free**: the program takes its value as an
input, and differentiates with respect to it. So the same model, with the same terms, answers
different questions: with the measurements observed, the log-density is a posterior of the states
up to a constant; with nothing observed, it is the joint; and the query of a term's own value
needs no observations at all.

`Observations` are built up one quantity at a time, each extension a new set with an identity of
its own. A caller keeps one set and passes it to every compile, and the identity is what the
compiled program is cached under.

## A target

A target is where a program computes and at what precision, checked once, when it is made:

```rust
{../../../../crates/skoro-tensor/src/target.rs:36..79}
```

skoro computes on burn's Flex device (the CPU) and on wgpu devices, in single or double precision.
The model itself commits to neither: precision belongs to the query, and two programs compiled from
one snapshot may run at different precisions side by side. `Target::host()` is the Flex device at
double precision, where the model folds its constants.

## Compiling

A snapshot compiles a query for a target with
`Snapshot::compile(&self, query: &Query, target: &Target) -> Result<Arc<Program>, Error>`.
Compiling walks back from the targets through the compilation graph, keeping only the operations
they need, lowers each operation for the target, expanding composites into their bodies, and folds
what the observations fix: an operation all of whose inputs are observed or constant is computed
once, when the program is made, and never again. A selection by an observed index keeps only the
branch it selects. A snapshot keeps its most recently used programs, so compiling the same query
again, for the same target and the same set of observations, returns the same program.

## Passes

A program runs a **pass** over a batch of rows, given a value of each free quantity:

```rust
{../../../../crates/skoro-core/src/program.rs:288..332}
```

`forward` gives the targets' values. `tangent` pushes tangent frames forward from the free
quantities to the targets; `cotangent` pulls cotangent frames back from the targets to every free
quantity; and `hessian` does the latter with velocities at the free quantities, giving the
second-order products as well. For a program whose single target is a `Real`, such as a
log-density, `gradient` and `hessian_products` seed that target with one.

Every derivative is taken in tangent charts: with respect to a tangent vector at the point, through
the [retraction](../spaces/#how-a-value-moves) of each free quantity, so a gradient has the
dimension of the quantity's tangent space and an estimator's step is retracted back onto it. The
note on [derivatives in charts](../../math/derivatives-in-charts/) gives the mathematics, and says
where the Hessian in a chart differs from the Hessian of the pulled-back function.

A program's `layout` lays its free continuous quantities out in one tangent vector, in ancestral
order, for algorithms that work with the whole of it at once.

## What a pass may refuse

Before running anything, a pass checks that it can be carried: every step on it must have a kernel
of the [order](../operations/#kernels-and-how-far-they-differentiate) it needs. A step is on a pass
when its output is continuous and reaches a target, and one of its continuous inputs is reached from
a free quantity. It needs order one on a gradient pass, and order two on a Hessian pass when one of
its inputs carries a velocity. A program that cannot carry a pass refuses it, naming the first step
that falls short, the composites it was expanded from, and its node, rather than computing a wrong
derivative. In the same spirit, an algorithm that differentiates refuses a program with a free
discrete quantity, which it could not condition on.

> [!DECISION] Algorithms refuse by structure
> What a model can do is read off its structure when a pass is asked for, never declared by its
> user. See
> [position 6 of the charter](../../archive/charter/writeup.md#6-algorithms-refuse-by-structure).
