---
title: Spaces and quantities
summary: Every value in a model lives in a space, which fixes how it is stored, how it moves, and the typed handle code holds it by.
order: 1
depends_on: [retraction, product-manifold, quotient-manifold, chart-volume]
---

Everything a model holds (its uncertain quantities, its constants, and the values computed from
them) lives in a **space**. The space says how a value is stored in a tensor, whether it is
continuous, and, when it is, how it moves: its tangent space and its retraction. This page
introduces the spaces of `skoro-core`, the retractions they move by, the typed handles code holds
them with, and the measures that densities on them are taken against.

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
`Declares` it, which is what lets a quantity be made from it. Where the space is known only at run time, as when an algorithm builds
nodes for whatever quantities it is given, `constant_in` makes a constant of the space itself:

```rust
{../../../../crates/skoro-core/src/scope.rs:445..453}
```

`QDyn` is the handle with no static type. A handle moves between the two: `untyped` forgets the
marker; `widen` turns a fixed size into a dynamic one, which cannot fail; and `narrow` claims a
marker, checked against the node's space when it runs, which can.

```rust
{../../../../crates/skoro-core/src/q.rs:58..86}
```

A handle can also be **failed**. Building a model never panics on a mistake such as adding a
vector to a matrix: the operation returns a failed handle, and everything built on it fails in
turn, as [building a model](../building/) describes.

## Measures and chart volumes

A density is a density with respect to a measure, and every continuous space with one says which,
through its charts. The chart at $\point$ carries the space's **reference measure** to a density on
the tangent coordinates, the chart's **volume**, and its logarithm $\lambda_\point(\tangent)$ is
the **log-volume**. A random tangent vector $\varepsilon$ of density $f$ moves $\point$ to a point
whose log-density is $\log f(\tangent) - \lambda_\point(\tangent)$ at $\tangent = \otherPoint
\ominus \point$: the note on [densities in tangent charts](../../math/chart-volume/) has the
mathematics. The core gives each space its log-volume as a composite, beside its retraction:

```
{../../../../crates/skoro-core/src/retraction.rs:10..20}
```

```rust
{../../../../crates/skoro-core/src/retraction.rs:60..88}
```

On the Euclidean spaces the chart is a translation and its log-volume is zero. On the positive
orthant it is $\sum_i (\log \point_i + \tangent_i)$, the logarithm of the moved point's
coordinates, which is why a density written in the logarithmic chart of a scale carries the
familiar $-\sum_i \log \otherPoint_i$. On a product it is the sum of the factors'. A discrete
space has counting measure and no chart. On a handle, `x.log_volume(t)` is $\lambda_x(t)$, so
that a density through the chart at `mu` is written

```rust
let t = x.local(mu);
let log_density = gaussian(t) - mu.log_volume(t);
```

Whether $\lambda_x$ depends on the point $x$ at all is a property of the space, read off its
structure rather than its numbers: on an embedded type, from whether its declared log-volume's
composite reads its first input, by `Composite::reads`. Where it does not, as on the rotations and
the spheres, a log-volume written through the chart at a mean is a function of the tangent alone,
which is what lets [the linearization](../linearization/#folding-a-log-volume) fold it into its
rows.

```rust
{../../../../crates/skoro-core/src/retraction.rs:119..143}
```

> [!DECISION] The positive orthant keeps Lebesgue measure
> Its retraction would normalize $\dd \point / \point$, but the densities of scales in common
> use, the gamma and the log-normal among them, are against Lebesgue measure, and their modes
> with them. See [the entry on reference measures](../../archive/measures/math.svx#the-core-s-own-types).

**Embedded types.** A curved type has no Lebesgue measure. It may declare its chart's volume,
against the measure its retraction normalizes, the one whose charts have volume $1$ at their
centre: Haar measure on a group, surface measure on a sphere, the same however the type is
stored, and for a type [stored up to a group](#types-stored-up-to-a-group), the measure on its
points.

```rust
{../../../../crates/skoro-core/src/embedded.rs:115..133}
```

```rust
{../../../../crates/skoro-core/src/embedded.rs:292..296}
```

The log-volume $\Lambda(\iota(\point), \tangent)$ is an operation on the point's storage and the
tangent, written as a composite, smooth as computed at $\tangent = 0$, and its derivatives are its
parts'. The domain is where the chart is a diffeomorphism: all of $\R^\dimension$, or where a
composite to `Boolean` says, such as the ball of radius $\pi$ for an exponential chart. A
declaration asks one more thing of the type's inverse retraction: that it invert the chart on all
of the chart's image, not only near its centre. The sphere of the core's tests declares its
stereographic chart's volume so:

```rust
{../../../../crates/skoro/tests/common/sphere.rs:76..92}
```

`testing::check_volume` holds a declaration to the area formula of the
[note](../../math/chart-volume/#the-measure-a-retraction-normalizes), computed from first-order
passes over the retraction, and checks that the volume is $1$ at the centre and that the inverse
holds across the domain.

> [!DECISION] A chart's volume is declared, not computed
> Computing it from the retraction would put a derivative of the retraction into the log-density,
> so its Hessian would need third derivatives, which no operation has. The area formula is what a
> test checks the declaration against instead. See
> [the entry on reference measures](../../archive/measures/math.svx#declared-not-computed).

**A space without a measure.** An embedded type that declares no volume has none, and neither
does a product with it among its factors; `has_measure` says so. Such a space has no densities: no
term may score a quantity of it, no proposal may draw it, and no retirement may integrate it out,
each refused with `NoVolume`. It may still be conditioned on, observed, and estimated by
optimization, and a measurement of it is held as a likelihood factor over the measured value.

## Types stored up to a group

Some types cannot be stored as a submanifold of their storage. A rotation stored as a unit
quaternion is stored twice, at $q$ and at $-q$, so the rotations are not a set of quaternions but
the quaternions taken up to their sign. An embedded type may declare such a storage: the finite
group of maps of its storage that carry a stored point to another storage of the same point, given
by its generators, each a signed permutation of the storage's entries:

```rust
{../../../../crates/skoro-core/src/embedded.rs:45..54}
```

```rust
{../../../../crates/skoro-core/src/embedded.rs:331..361}
```

A point of such a type is the set of its stored values, and the type's charts are read on the
points. The retraction must carry the group, $R(g \point, \tangent) = g\, R(\point, \tangent)$, so
that a step from either representative reaches the same point; the inverse must not see it,
$Q(g \point, \otherPoint) = Q(\point, \otherPoint) = Q(\point, g \otherPoint)$, so that a tangent
coordinate is a property of the points; the projection must carry it as the retraction does; and a
declared log-volume must not see it either. Then every function the core computes from a quantity
of the type, its densities, its gradients and Hessians in the charts, its steps, is the same at
every representative, and none of the core's derivatives changes, since they are taken at the
stored value as before (the note on [quotients by a finite group](../../math/quotient-manifold/)
proves it, and the note on [densities in tangent charts](../../math/chart-volume/#quotients) carries
the measure and the volume to the points). A declared chart is global and full when it is so on the points: the Cayley chart of the
rotations reaches only the quaternions on the side of its base point, but every rotation but the
half-turns.

`testing::check_embedded` and `check_volume` check each condition for each generator, and that no
generator fixes a stored point; `testing::close_up_to` compares two stored values as points, up to
the group. What the core cannot check is a map a model writes on the inclusion, which reads the
stored value itself: it is a map of the points only if it gives the same at every representative,
and a type's maps say which they are.
