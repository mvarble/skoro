---
title: Spaces and quantities
summary: Every value in a model lives in a space, which fixes how it is stored, how it moves, and the typed handle code holds it by.
order: 1
depends_on: [retraction, product-manifold]
---

Everything a model holds (its uncertain quantities, its constants, and the values computed from
them) lives in a **space**. The space says how a value is stored in a tensor, whether it is
continuous, and, when it is, how it moves: its tangent space and its retraction. This page
introduces the spaces of `skoro-core`, the retractions they move by, and the typed handles code
holds them with.

## The spaces

A space is a closed enumeration, dimensioned when a value is made:

```rust
{../../../../crates/skoro-core/src/space.rs:6..20}
```

The continuous spaces are the real line, vectors and matrices (a matrix stored column by column),
the positive orthant $\positive^n$ of scales and rates, products of continuous spaces, and the
curved types declared by later crates as embedded types. The discrete spaces are a set of
`n` categories, with the truth values as `Categorical(2)`, and the integers.

Two numbers describe a space. Its **storage** is the number of entries a value takes in one row of
a tensor; its **dimension** is that of its tangent space, which a discrete space does not have.
For the flat spaces the two agree. They differ for a curved type, a sphere in $\R^3$ for one, which
is stored as the three coordinates of a point and moves in two dimensions.

```rust
{../../../../crates/skoro-core/src/space.rs:50..69}
```

A product is made by `Space::product`, which refuses a discrete factor: a product is something
derivatives are taken through, and a discrete value has none. Its `blocks` give each factor's
range in the product's storage and in its tangent coordinates, which is how anything that works
on a product reads one factor out of it.

## How a value moves

Derivatives, gradients and the steps an estimator takes are all read in a space's tangent
charts, which its [retraction](../../math/retraction/) $\point \oplus \tangent$ and that
retraction's inverse $\otherPoint \ominus \point$ make. Every continuous space of the core has
both, as composite operations cached per space:

```
{../../../../crates/skoro-core/src/retraction.rs:1..9}
```

On a flat space they are addition and subtraction. On the positive orthant the retraction
multiplies by the exponential of the tangent, so a tangent coordinate is a relative change and a
step never leaves the space. A product retracts each factor by its own retraction, which is what
the note on [product manifolds](../../math/product-manifold/) shows to be a retraction of the
product.

## Typed handles

Code never holds a value. It holds a **handle**: a node's identifier and a marker of its type,
cheap to copy and bound to no model, writer, device or precision.

```rust
{../../../../crates/skoro-core/src/q.rs:7..22}
```

A marker is a zero-sized type that says which spaces its handles may name: `Real`, `Vector<N>`,
`Matrix<M, N>`, `Positive<N>`, `Categorical<N>`, `Boolean` and `Integer`, each checked at compile
time, and `VectorDyn`, `MatrixDyn`, `PositiveDyn`, `CategoricalDyn` and `ProductDyn`, whose size is
fixed by the value instead. A marker that fixes its space (every one but `ProductDyn`) also
`Declares` it, which is what lets a quantity be made from it.

`QDyn` is the handle with no static type. A handle moves between the two: `untyped` forgets the
marker; `widen` turns a fixed size into a dynamic one, which cannot fail; and `narrow` claims a
marker, checked against the node's space when it runs, which can.

```rust
{../../../../crates/skoro-core/src/q.rs:58..86}
```

A handle can also be **failed**. Building a model never panics on a mistake such as adding a
vector to a matrix: the operation returns a failed handle, and everything built on it fails in
turn, as [building a model](../building/) describes.
