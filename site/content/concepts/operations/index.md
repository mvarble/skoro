---
title: Operations and their derivatives
summary: Maps between spaces described once and on no device, lowered to kernels that say how far they differentiate, and composed from a few primitives.
order: 2
depends_on: [concepts/spaces, automatic-differentiation]
---

Every value a model computes is the output of an **operation**: a map between
[spaces](../spaces/), such as a sum, an exponential or a matrix product. An operation is described
once, on no device and at no precision, and is turned into something that runs only when a query
is compiled for a target. This page describes operations, the kernels they are lowered to, and
how the core builds everything from a small set of primitives.

## An operation

```rust
{../../../../crates/skoro-core/src/operation.rs:10..21}
```

An operation is known by its name and its parameters, plain values fixed when it is made and
never inputs. The two are enough for a registry to make the operation again, which is how a model
can one day be written out and read back. When a node is added to a model, `output` checks the
inputs' spaces and gives the output's; a mismatch fails the node there, at the line that built it,
rather than when the model is run.

`lower` is asked only when a query is compiled, for one target, and answers in one of two ways:

```rust
{../../../../crates/skoro-core/src/operation.rs:64..70}
```

## Kernels, and how far they differentiate

A kernel computes an operation on one target, and its variant is its **order**: how far it
differentiates.

```rust
{../../../../crates/skoro-core/src/operation.rs:133..147}
```

A kernel of order zero computes its value alone. One of order one also pushes tangent frames
forward and records a pull-back, which carries cotangents back to its inputs. One of order two
also carries the second-order terms that Hessian-vector products need, at velocities given at its
inputs:

```rust
{../../../../crates/skoro-core/src/operation.rs:149..179}
```

These are the maps of [automatic differentiation](../../math/automatic-differentiation/) in
forward, reverse and forward-over-reverse mode, one operation at a time. A program composes them,
and a pass over a program needs every step on it to have the order the pass requires: a gradient
needs order one of every step it differentiates through, and a Hessian pass order two. A program
checks this before it runs anything and refuses a pass it cannot carry, naming the first step that
falls short, as [queries and programs](../queries/) describe. The integer operations, for instance,
have order zero, so they may feed a value but no derivative may be taken through them.

## The primitives

A few primitive operations carry hand-written kernels, the same code at both precisions:

```rust
{../../../../crates/skoro-core/src/primitive/mod.rs:9..17}
```

The Euclidean and positive primitives differentiate to order two; the discrete ones (integer
arithmetic and comparison, the embedding `to_real` and the test `is_nonnegative`) to order zero.
`scope::select` chooses one of several values by a categorical index, and routes the cotangent to
the one it chose.

Code rarely names a primitive. A handle carries them as operators and methods (`x + y`, `x * y`,
`-x`, `x.exp()`, `x.sum()`, `a.matmul(b)`, `p.as_real()`, `p.recip()`), typed by the markers of
their operands, so that adding a vector to a matrix does not compile. A literal operand, as in
`0.5 * x`, becomes a constant node of the model.

## Composites

Everything that is not a primitive, distributions and retractions included, is a **composite**: a
named subgraph with designated inputs and one output, built by the same code a model would use.

```rust
{../../../../crates/skoro-core/tests/passes.rs:25..37}
```

`Composite::define` runs the body once, on handles standing for the inputs, and keeps the nodes it
builds. `call` adds one node to the model, the composite applied to its inputs. When a query is
compiled, the composite is lowered by expanding its body in place, so it differentiates as far as
the operations inside it do, with no derivative written for it. The retractions of the previous
page are composites of this kind.

> [!DECISION] Derivatives come from composition
> Only primitives carry hand-written derivatives; everything else is composed from them, and
> reports the order its parts support. See
> [position 5 of the charter](../../archive/charter/writeup.md#5-derivatives-come-from-composition-and-their-order-is-reported).
