---
summary: The improper Gaussian factor as `Stacked::improper` in skoro-normal, a builder of its rows, right-hand side, linear part and constant over a stacked tangent, which gives a retirement's replacement or a likelihood factor with its residual and its linear part bound, and the core's support, through every pass, for the values of no entries that a factor of no rows needs.
depends_on: [archive/improper-gaussian/math, archive/normal/math, archive/core/math]
---

## Overview

When this is built, `skoro-normal` writes the improper Gaussian factor of
[%full](statement:archive/improper-gaussian/math/factor) beside the normal's `Stacked`
replacement. A caller builds the stacked tangent as before, block by block at constant anchors, and
gives the factor's numbers: the number of rows $r$, the rows $A$, the right-hand side $b$, the
constant $c$ and, where it has one, the linear term $\eta$. What comes back is an `Improper`, whose
`replacement` a retirement adds and whose `factor` a model adds directly, each binding the whitened
residual $A \delta - b$ and, where $\eta$ is given, the linear part $\eta\transpose \delta$ under
the log-volume role ([The roles](math.svx#the-roles)).

The bearing of [Two examples](math.svx#two-examples), retiring the pose $\point_k$ with $z_k$
observed:

```rust
use skoro::{Addresses, Vector};
use skoro_normal::Stacked;

let a = [0.6, 0.8]; // the unit direction a_k
let s = (sigma * sigma + p_var).sqrt(); // s_k, with P_k = p_var I
let factor = writer
    .retire(Addresses::new().under("pose/0"), observed)?
    .replace(|_| {
        Stacked::new()
            .block(landmark, Vector::<2>, q)
            // one row, a_kᵀ / s_k, stored column by column
            .improper(1, &[a[0] / s, a[1] / s][..], &[(z_hat + dot(a, p) - dot(a, q)) / s][..],
                -0.5 * (2.0 * PI * s * s).ln())
            .replacement()
    })?;
```

The landmark is scored by no term, before the retirement and after it, and so is a parameter of
the model; the factor constrains it along $a_k$ alone. A second retirement along another direction adds a second factor,
and the model's log-density is their sum, a normal in the landmark
([Accumulation](math.svx#accumulation)).

The factor's value is [%full](statement:archive/improper-gaussian/math/factor). Its derivatives are
those of $\delta$ composed with $A$ and $\eta$, computed by the core's passes through the nodes the
builder makes ([Derivatives](math.svx#derivatives)). Nothing in it is new mathematics for the core:
it is a matrix product, a difference, a sum of squares, an inner product and a sum, at nodes the
core already has.

A factor of no rows has a residual of no entries, which the core panics on today.
[Values of no entries](#values-of-no-entries) says where, and [A factor of no rows](#a-factor-of-no-rows)
what the entry recommends.

## Constraints

Read at `19b4b35`, the last commit to change `crates/`.

### The stacked tangent

`crates/skoro-normal/src/stacked.rs`:

```rust
#[derive(Clone, Debug, Default)]
pub struct Stacked {
    blocks: Vec<QDyn>,
    dimension: usize,
}

impl Stacked {
    pub fn new() -> Stacked;
    #[track_caller]
    pub fn block<T: Declares>(mut self, x: Q<T>, marker: T, anchor: impl IntoHostValue<T>) -> Stacked;
    pub fn dimension(&self) -> usize;
    #[track_caller]
    pub fn tangent(&self) -> Q<VectorDyn>;
    #[track_caller]
    pub fn log_density(&self, mean: &[f64], scale: Scale, constant: f64) -> Q<Real>;
    #[track_caller]
    pub fn replacement(&self, mean: &[f64], scale: Scale, constant: f64) -> Replacement;
    #[track_caller]
    pub fn factor(&self, mean: &[f64], scale: Scale, constant: f64) -> TermHandle;
}

/// What the normal's roles are bound on: a term in a scope, or a retirement's replacement.
trait Binds: Sized {
    fn bind_role<M: Multiplicity>(self, role: &Role<M>, node: QDyn) -> Self;
}
```

`block` makes each anchor a constant itself, from a host value with its marker, so that the
condition of [The stacked tangent](math.svx#setting), anchors that are constants, holds by
construction. The improper factor reads $\delta$ from `tangent()` and inherits that guarantee. Its
numbers must be constants of the model in the same way, made by the builder from host values, for
the same reason: the core checks that a replacement is reached from no quantity outside the blanket,
and a constant is reached from none. `Binds` is private to `stacked.rs`, and a factor and a
replacement are bound through one helper so that the two bind alike; the improper factor needs the
same.

### Constants, rows and the order of a matrix's entries

`crates/skoro-core/src/scope.rs` and `crates/skoro-core/src/into_host.rs`:

```rust
/// A constant of `marker`'s space (`ValueShape` when the value does not fit it).
#[track_caller]
pub fn constant<T: Declares>(marker: T, value: impl IntoHostValue<T>) -> Q<T>;

/// One value per row, pinning the batch to their number.
pub struct Rows<V>(pub Vec<V>);

/// In storage order: column by column.
impl IntoHostValue<MatrixDyn> for &[f64] { ... }
impl IntoHostValue<VectorDyn> for &[f64] { ... }
impl IntoHostValue<Real> for f64 { ... }
impl<T, V: IntoHostValue<T>> IntoHostValue<T> for Rows<V> { ... }
```

A constant given as `Rows` pins the batch, so that each row of a batch can have numbers of its own,
as the math asks ([In skoro's terms](math.svx#in-skoro-s-terms)), while its space, and so $r$ and
$D$, is one for every row. A value that does not fit the space is refused with `ValueShape` at the
caller's line, and values that pin different batches with `BatchSize` when the query is compiled.
The builder therefore takes its numbers as `impl IntoHostValue<_>`, as `block` takes its anchors,
and checks nothing the core already checks.

A `MatrixDyn` given as a slice is read **column by column**. A caller with $A$ row by row, as a
host QR factorization often gives it, transposes it before the call. The builder documents this and
does not reorder: a second convention for the same marker would be one more thing to get wrong.

### Replacements and roles

`crates/skoro-core/src/retire.rs` and `crates/skoro-normal/src/roles.rs`:

```rust
pub struct Replacement {
    pub log_density: Q<Real>,
    pub proposals: Vec<(QDyn, Proposal)>,
    /// Each a role's name and the node bound to it.
    pub bindings: Vec<(&'static str, QDyn)>,
}

impl Replacement {
    pub fn bind<M: Multiplicity>(mut self, role: &Role<M>, node: impl Into<QDyn>) -> Replacement;
}

/// The whitened residual `W (x ⊖ μ)` of a term, or `W (z ⊖ μ)` of a factor.
pub const RESIDUAL: Role<ExactlyOne> = Role::new("skoro.residual");
/// The log-volume `λ_μ(x ⊖ μ)`, where it is not identically zero.
pub const LOG_VOLUME: Role<AtMostOne> = Role::new("skoro.log_volume");
```

The residual role takes exactly one node and the log-volume role at most one, so the factor binds
the residual always and the linear part only when $\eta$ is given. Both roles are named without the
normal's prefix already, since they "mean the same of any distribution with a whitened residual",
as the module's documentation says. Their descriptions speak of the normal alone, and the
log-volume's of a chart's volume, which the math's decision widens
([The roles](math.svx#the-roles)).

### Values of no entries

A factor of no rows has a residual in $\R^0$, and the core panics on it. A constant of `Vector(0)` or `Matrix(0, 2)`, compiled into a
query, panics in `HostValue::load` (`crates/skoro-tensor/src/host.rs:68`):

```text
Broadcast arguments must be positive or -1! Got 0
```

which burn's `Tensor::expand` raises for a target dimension of zero. A value of no entries made
without a constant, `x.slice(0..0)` of a `Vector(2)` quantity, runs forward through a product, a
matrix product and a sum, and then panics in the same way in the reverse pass, in the pullback of
`Sum` (`crates/skoro-core/src/primitive/euclidean.rs`), which expands its cotangent to the input's
shape. On the host device, burn itself makes a `[3, 0]` tensor, sums it over its empty dimension to
zeros, multiplies matrices with an empty inner or outer dimension, and concatenates an empty tensor
with another; only `expand` refuses. skoro calls `expand` at sixteen places, in `skoro-tensor`'s
`host.rs` and in `skoro-core`'s `program.rs` and the primitives' `rule.rs`, `euclidean.rs` and
`positive.rs`. What wgpu does with a buffer of no bytes has not been tried, and is not assumed.

## Design

### The factor

In `crates/skoro-normal/src/improper.rs`, a module of its own beside `stacked.rs`, exported as
`skoro_normal::Improper`:

```rust
//! The improper Gaussian factor `-½|A δ - b|² - ηᵀδ + c` over a stacked tangent at constant
//! anchors: a sum of squared affine rows of any number, a linear part and a constant, which no
//! normalization need make a density of `δ`.

use skoro_core::scope::{constant, factor};
use skoro_core::{IntoHostValue, MatrixDyn, Q, Real, Replacement, TermHandle, VectorDyn};

use crate::stacked::Binds;
use crate::{Stacked, roles};

/// An improper Gaussian factor over a stacked tangent: its nodes, made in the open scope by
/// [`Stacked::improper`]. Its value is `-½|A δ - b|² - ηᵀδ + c`; it binds the residual
/// `A δ - b` and, when a linear part is given, `ηᵀδ` under the log-volume role.
#[derive(Clone, Copy, Debug)]
pub struct Improper {
    tangent: Q<VectorDyn>,
    dimension: usize,
    /// `A δ - b`, a `Vector(r)`.
    residual: Q<VectorDyn>,
    /// `ηᵀδ`, when a linear part is given.
    linear: Option<Q<Real>>,
    constant: Q<Real>,
}

impl Stacked {
    /// The improper Gaussian factor over `δ` with `rows` rows: `a` the `rows × D` matrix `A`,
    /// **column by column** as a `MatrixDyn` is stored, `b` the right-hand side, a
    /// `Vector(rows)`, and `c` the constant, and no linear part. Each may be shared or given per
    /// row of a batch with `Rows`. Any number of rows is a factor, none included; the rows are
    /// kept as given, and none is added or removed.
    ///
    /// **The order of the blocks is load-bearing.** The columns of `A`, and the entries of `η`,
    /// follow the order of the `block` calls, factor by factor, each over its quantity's tangent
    /// coordinates. Nothing checks that correspondence, because nothing can: a matrix of the right
    /// width is the rows of a factor in any order.
    #[track_caller]
    pub fn improper(
        &self,
        rows: usize,
        a: impl IntoHostValue<MatrixDyn>,
        b: impl IntoHostValue<VectorDyn>,
        c: impl IntoHostValue<Real>,
    ) -> Improper {
        let tangent = self.tangent();
        let a = constant(MatrixDyn(rows, self.dimension()), a);
        let b = constant(VectorDyn(rows), b);
        Improper {
            tangent,
            dimension: self.dimension(),
            residual: a.matmul(tangent) - b,
            linear: None,
            constant: constant(Real, c),
        }
    }
}

impl Improper {
    /// The same factor with the linear part `ηᵀδ`, `eta` a `Vector(D)`, shared or per row.
    /// Whether a factor has one is a matter of its shape, the same for every row of a batch: an
    /// `eta` of zeros is still bound.
    #[track_caller]
    pub fn linear(self, eta: impl IntoHostValue<VectorDyn>) -> Improper {
        let eta = constant(VectorDyn(self.dimension), eta);
        Improper {
            linear: Some((eta * self.tangent).sum()),
            ..self
        }
    }

    /// `δ`.
    pub fn tangent(&self) -> Q<VectorDyn>;
    /// `A δ - b`.
    pub fn residual(&self) -> Q<VectorDyn>;
    /// `ηᵀδ`, when a linear part was given.
    pub fn linear_part(&self) -> Option<Q<Real>>;

    /// `-½|A δ - b|² - ηᵀδ + c`, as a log-density alone, which binds no roles.
    #[track_caller]
    pub fn log_density(&self) -> Q<Real> {
        let value = (self.residual * self.residual).sum() * -0.5 + self.constant;
        match self.linear {
            Some(linear) => value - linear,
            None => value,
        }
    }

    /// The factor as a retirement's `Replacement`, with no proposals and its roles bound as
    /// [`factor`](Improper::factor) binds them. A blanket quantity it leaves undetermined needs a
    /// proposal from the caller to be generated.
    #[track_caller]
    pub fn replacement(&self) -> Replacement {
        let replacement = Replacement {
            log_density: self.log_density(),
            proposals: Vec::new(),
            bindings: Vec::new(),
        };
        self.with_roles(replacement)
    }

    /// The factor added as a likelihood factor, scoring nothing, with its roles bound.
    #[track_caller]
    pub fn factor(&self) -> TermHandle {
        self.with_roles(factor(self.log_density()))
    }

    /// `target` with the residual bound, and the linear part under the log-volume role when
    /// there is one: the same for a factor and a replacement.
    #[track_caller]
    fn with_roles<B: Binds>(&self, target: B) -> B {
        let target = target.bind_role(&roles::RESIDUAL, self.residual.untyped());
        match self.linear {
            Some(linear) => target.bind_role(&roles::LOG_VOLUME, linear.untyped()),
            None => target,
        }
    }
}
```

`Binds` becomes `pub(crate)` in `stacked.rs`, and nothing else there changes.

Each piece maps to the mathematics directly: `tangent` is $\delta$
[](eq:archive/normal/math/stacked), `residual` is $A \delta - b$, `linear` is $\eta\transpose
\delta$, and `log_density` is the factor of
[%full](statement:archive/improper-gaussian/math/factor). Its derivatives are the core's passes
through these nodes: on the flat-in-disguise types, where $\D \delta$ is constant in every chart,
the residual's Jacobian is constant and the Gauss–Newton matrix is the exact negative Hessian; on a
curved type, the core's inverse retraction gives $\D \delta$ and its curvature
([Derivatives](math.svx#derivatives)).

A refusal is the core's. A blanket quantity of a discrete type is refused by `block`
(`NotContinuous`), as before. Numbers that do not fit their spaces, an `a` of other than
$r \cdot D$ entries, a `b` of other than $r$, an `eta` of other than $D$, are refused by `constant`
with `ValueShape` at the caller's line, through `#[track_caller]`. Numbers that pin different
batches are refused with `BatchSize` when the query is compiled. The builder reads no number, so no
number changes what it builds.

**Alternatives that lost.**

- *Numbers given as nodes*, `Q<MatrixDyn>` and so on, made by the caller. That would let a caller
  pass a node reached from a quantity, which the core's check of a replacement then refuses at the
  retirement, far from the mistake, and which, added as a factor, would no longer be the
  mathematics' factor with constant numbers. Host values keep the numbers constants by
  construction, as anchors are.
- *The linear term as an argument of `improper`*, an `Option` beside the others. `None` of an
  `impl IntoHostValue` needs a type annotation at every call that has no $\eta$, which is most of
  them, and the builder method reads as the mathematics does: a factor, with a linear part.
- *A struct of the numbers*, built by the caller and passed to `Stacked`. It holds nothing the
  arguments do not, and the numbers must become constants in the scope where the factor is built
  anyway, which the call does.
- *A conversion to `Stacked`'s mean, whitening and constant* when $A$ has full rank, by
  [%full](statement:archive/improper-gaussian/math/full-rank). The math keeps `Stacked` for a
  caller that has a mean and a scale and the improper factor for one that has rows, and chooses no
  form by the numbers; a conversion would read them, and only the tests need it.

**Against the archive.** The archive's `JointNormal` (`crates/skoro-normal/src/joint.rs` on
`archive`) was the replacement of its sliding windows, $-\tfrac12 \lVert W \delta - d \rVert^2 + n$
over a stacked tangent at frozen anchors. Three things are kept. The anchors are frozen as
constants, so that the factor does not follow the quantities it summarizes. The right-hand side is
explicit, rather than folded into a moved anchor, which the archive noted is exact only on a flat
type, and which keeps the factor's value and Jacobian at the anchors exactly its rows on any type.
And the numbers are pinned to the batch, one set for each row. Three are not. The archive's `W` was
triangular and of full rank, and the rows here are of any number, which is the point of the entry.
Its normalizer was one `f64` shared by every row, and the constant here may differ from row to row,
as a linearization's does. And it named its quantities by id, beside a map of their points, where
`Stacked` names them by their handles, one block at a time, with each anchor beside its quantity.
Its warning that the order of the blocks is load-bearing carries over unchanged, to the
documentation of `improper`.

> [!DECISION] The builder reduces no rows
> [In skoro's terms](math.svx#in-skoro-s-terms) says that a reduction to $\min(r, D)$ rows is a
> change of form the library may apply, and that it applies no other. This design applies none, by
> a host QR of $[A \;\, b]$ or otherwise. A reduction would read the numbers of every row on the
> host, and would change the residual a reader finds from the one the caller wrote, while the
> `linearization` entry, the one producer in sight, already gives $\min(m', D)$ rows. A caller with
> more rows than the blanket has coordinates reduces them itself, or keeps them, at no cost but
> their number. A later entry may give the builder the reduction, as the math allows, should a
> caller need it.

### The roles' descriptions

In `crates/skoro-normal/src/roles.rs`, the descriptions say what the math decided, and nothing
else changes:

```rust
//! The roles a normal's term binds, by which a reader in another crate finds its parts.
//!
//! The residual, the log-volume and the information root are named without the normal's prefix:
//! they mean the same of any term with a whitened residual or a closed-form information, the
//! improper factor of [`Improper`](crate::Improper) among them.

/// The whitened residual: `W (x ⊖ μ)` of a normal's term, `W (z ⊖ μ)` of its factor, and
/// `A δ - b` of an improper factor, of any number of components.
pub const RESIDUAL: Role<ExactlyOne> = Role::new("skoro.residual");

/// The varying part of a term's remainder: what is left of its log-density after the squares of
/// its residual and a normalizer constant in its free quantities, where that is not identically
/// zero. On a normal it is the chart's log-volume `λ_μ(x ⊖ μ)`, and on an improper factor its
/// linear part `ηᵀδ`. It is not the volume of a chart: a reader that needs a quantity's chart
/// volume reads it from the quantity's type.
pub const LOG_VOLUME: Role<AtMostOne> = Role::new("skoro.log_volume");
```

The name stays, as the math decided, so that every least-squares reader reads one role for the
varying part of a remainder.

### A factor of no rows

A factor of no rows is part of the mathematics
([The roles](math.svx#the-roles): "a residual of no components is a residual"), and the
`linearization` entry relies on it, since a retirement whose retired residuals have no more
components than the retired quantities have coordinates leaves $\min(m', D) = 0$ rows. It needs the core to carry values of no entries through every
pass, which it does not ([Values of no entries](#values-of-no-entries)). The builder above needs no
special case once it does: the matrix product of a `Matrix(0, D)`, the sum of squares
of a `Vector(0)` and the inner product with a `Vector(0)` are all the right values, zero.

A stacked tangent of no blocks is another matter, and stays as it is: `tangent()` concatenates no
blocks, which the core refuses (`SpaceMismatch` of `skoro.concatenate`), so `improper` over it is
refused at the caller's line. The math's blanket has at least one continuous quantity, and a
retirement whose blanket has none has a replacement that is a constant, which a caller writes as
`constant(Real, c)` with no factor to build.

> [!DECISION] The core is fixed in this entry
> The entry that needs values of no entries is the one that makes the core carry them. Its build
> changes `skoro-tensor` and `skoro-core` as well as `skoro-normal`, and the tests of `skoro`; its
> `crates` list all four at publish, and it is pushed only after a run with an adapter. A chore entry for the core first would
> have set this entry aside for a change of a few lines; padding a factor of no rows with one zero
> row, which part 2 of [%full](statement:archive/improper-gaussian/math/forms) allows on shapes
> alone, would leave the core as it is but give a reader a residual of one component where the math
> says it has none, and the `linearization` entry a row it does not count.

The fix is in the tensor layer: one function that broadcasts a tensor of any kind to a shape, in
`crates/skoro-tensor/src/value.rs`, exported from the crate's root beside `Points` and `Frame`:

```rust
use burn::tensor::kind::Basic;

/// `tensor` broadcast to `shape`, of as many dimensions or more, as burn's `expand` does; where
/// `shape` has a dimension of zero, which `expand` refuses, the tensor of no entries of that
/// shape, of `tensor`'s kind, element type and device.
pub fn broadcast<const D: usize, const D2: usize, K: Basic>(
    tensor: Tensor<D, K>,
    shape: [usize; D2],
) -> Tensor<D2, K> {
    if shape.contains(&0) {
        // expand's rule, aligned from the last dimension: each of the tensor's is 1 or the shape's.
        debug_assert!(D <= D2 && tensor.dims().iter().rev().zip(shape.iter().rev())
            .all(|(&from, &to)| from == 1 || from == to));
        // No entries in, none out: slicing every dimension to nothing keeps the kind, the element
        // type and the device, and a tensor of no entries reshapes to any shape of none.
        tensor.slice(std::array::from_fn::<_, D, _>(|_| 0..0)).reshape(shape)
    } else {
        tensor.expand(shape)
    }
}
```

A scratch run of this body, without the assertion, on the host device at double precision, gave
tensors of no entries of the shapes asked from float, integer and boolean tensors, `[1, 2]` to
`[3, 0, 2]` and a `[1, 0, 1]` to `[2, 3, 0, 5]` among them, with the float element type kept; to
`[3, 4, 2]` it was `expand`'s; and the sum of a `[3, 0]` tensor over its second dimension was three
zeros. The
alternative, `Tensor::zeros` of the shape, exists for float and integer tensors with an explicit
element type, but not alike for every kind, and would need the dtype read back from the tensor.

Every call of `expand` in `crates/` becomes a call of `broadcast`, with the tensor and the shape it
has today:

| file | line at `19b4b35` | what it broadcasts |
| --- | --- | --- |
| `skoro-tensor/src/host.rs` | 68, 75 | a shared host value to its batch |
| `skoro-core/src/program.rs` | 775, 780, 793 | a shared value or frame to the batch of a pass |
| `skoro-core/src/primitive/rule.rs` | 176, 182 | a value, and a cotangent, across a frame's columns |
| `skoro-core/src/primitive/euclidean.rs` | 489, 502, 532, 563, 619, 625, 646 | a scalar across a vector in `Product` and `Scaling`, and across a frame's columns; a matrix across columns in `MatrixProduct`; the cotangent in `Summation`'s pullback |
| `skoro-core/src/primitive/positive.rs` | 157, 165 | `Selection`'s mask, in its value and across a frame's columns |

Where `expand` follows an `unsqueeze_dim`, at `rule.rs:176` and `euclidean.rs:563` (`repeat`), the
`unsqueeze_dim` keeps its place, `broadcast(x.unsqueeze_dim::<3>(2), [n, d, k])`.
`burn::tensor::kind::Basic` is the bound of burn's own `Tensor` methods, so the function takes every
tensor `expand` takes.

The run of `x.slice(0..0)` in [Values of no entries](#values-of-no-entries) exercised the forward
pass only, on the host device, before the first `expand` of the reverse pass stopped it; the tangent, cotangent and Hessian passes beyond the places that call
`expand` are untried. The test `values_of_no_entries_through_every_pass` below is what finds out.
A further failure it finds in a kernel is fixed in that kernel, in the same way: an operation with
no entries has the value of no entries, or a sum over none, zero. If a fix needs more than a guard
around one burn call, the build stops and says so, since the design would then be incomplete.
A change to `skoro-tensor` and to the primitives' kernels is pushed only after
`cargo test --workspace --features wgpu` passes on a machine with an adapter, and its commit says
so; wgpu's handling of empty buffers is the part least known, and a failure there is reported the
same way.

## Tests

In `crates/skoro-normal/tests/improper.rs`, at both precisions of the Flex device where a value is
compared, with the tolerances of `tests/common`:

- `value_is_the_definition`: over a `Vector(2)` quantity and a `Positive(1)` quantity at constant
  anchors, $D = 3$, with two rows (rank two, below $D$), a non-symmetric $A$ given column by column,
  a right-hand side, a linear term and a constant, the log-density at several points equals
  $-\tfrac12 \lVert A \delta - b \rVert^2 - \eta\transpose \delta + c$ computed on the host, with
  $\delta = (x - p, \log y - \log q)$; the residual equals $A \delta - b$. A square non-symmetric
  $A$ on a `Vector(2)` alone checks the order of the entries: given transposed, it fails. Shows
  [%full](statement:archive/improper-gaussian/math/factor) is built as written.
- `full_rank_is_the_stacked_normal`: for an $A$ of rank $D$ and an $\eta$ both in its row space and
  not, the improper factor's log-density equals `Stacked::log_density` with the mean $m$, the
  whitening $W$ (by `Scale::precision` of $A\transpose A$) and the constant $c_{\mathcal N}$ of
  [%full](statement:archive/improper-gaussian/math/full-rank), computed on the host, at several
  points. Shows the proposition, and that the improper factor contains `Stacked`.
- `forms_give_one_function`: from one factor, an orthogonal $Q$ applied to $[A \;\, b]$, zero rows
  appended with their $e$ and $c$ corrected, a shift of $b$ by $u$ with $\eta$ and $c$ corrected, and
  a moved anchor on the `Vector(2)` and on the `Positive(1)` quantity with $b$ and $c$ corrected,
  each give the same log-density at every point. Shows
  [%full](statement:archive/improper-gaussian/math/forms).
- `positive_example_is_exact`: the positive example of [Two examples](math.svx#two-examples), with
  the $A$, $b$, $\eta = (0, 1)$ and $c$ the math gives, equals the closed form
  $-\log \point - (\log \point - \varphi)^2 / (2 s^2) - \tfrac12 \log (2\pi s^2)$ at several points
  of $(\varphi, \point)$, at anchors other than the points.
- `roles_are_bound`: a factor without a linear part binds the residual and no log-volume; with one,
  the residual and the linear part under the log-volume role; neither binds a mean, a scale or an
  information root. Shows [The roles](math.svx#the-roles).
- `gauss_newton_is_the_hessian_on_flat_types`: on the `Vector(2)` and `Positive(1)` blanket, in the
  logarithmic chart of the positive quantity, the Hessian products of the factor's log-density
  equal $-\D \delta\transpose A\transpose A\, \D \delta$ times the velocity, and its gradient
  $-\D \delta\transpose (A\transpose (A \delta - b) + \eta)$, at several points. Shows
  [Derivatives](math.svx#derivatives) on the flat-in-disguise types.
- `rows_differ_across_a_batch`: $A$, $b$, $\eta$ and $c$ given as `Rows` of three, each row's
  log-density equals that row's factor built alone; numbers of the wrong length are refused with
  `ValueShape` at the caller's line; `Rows` of three beside `Rows` of two are refused with
  `BatchSize` when the query is compiled.
- `no_rows_is_its_linear_part`: a factor of no rows, with and without a linear term, has the
  log-density $-\eta\transpose \delta + c$, a residual of no components, and runs forward, tangent,
  cotangent and Hessian passes, its gradient $-\D \delta\transpose \eta$.
- `no_blocks_is_refused`: `improper` over a `Stacked` with no blocks fails with `SpaceMismatch` of
  `skoro.concatenate`, naming the caller's line.

In `crates/skoro/tests/`, where the retirements and the spheres are:

- `retirement.rs::bearings_accumulate`: the planar bearing of
  [Two examples](math.svx#two-examples), a landmark with no prior and two poses with priors, each
  measured along its own direction. Retiring the first pose with the improper replacement of one
  row leaves a model whose log-density in the landmark equals, at several points, the integral of
  the retired terms computed in closed form; retiring the second leaves two factors whose sum
  equals the replacement of retiring both at once, constants included, and is the normal of
  [%full](statement:archive/improper-gaussian/math/full-rank) whose mean is the point where the two
  lines cross, computed on the host. Shows [Accumulation](math.svx#accumulation) without a sum.
- `retirement.rs::improper_replacement_binds_its_roles`: the replacement of the first retirement
  above binds the residual on the factor the retirement adds, and the factor of the positive example
  binds its linear part under the log-volume role, as the same factors added with `factor` do.
- `retirement.rs::undetermined_blanket_needs_a_proposal`: the positive example, retiring $\theta$,
  retires the normal that scored $\point$, so that $\point$, scored by no term after it and left
  undetermined by a factor of rank one on $D = 2$, is refused generation without a proposal, as the
  core refuses every parameter, and is generated with one given on the replacement.
- `sphere.rs::improper_factor_on_the_sphere`: over a point of the acceptance-test sphere and a
  `Real`, the factor equals the definition with $\delta$ read by the sphere's inverse retraction,
  and its gradient agrees with central differences in the sphere's chart. Shows the factor on a
  curved type, where $\D \delta$ moves with the point.

For values of no entries, in the crates the fix changes:

- `crates/skoro-tensor/tests/values.rs::broadcast_to_no_entries`: `broadcast` of a float, an
  integer and a boolean tensor to shapes with a dimension of zero, of as many dimensions and of
  more, gives tensors of those shapes, of the same kind and element type, at both precisions; to
  shapes without one, it equals `expand`.
- `crates/skoro-tensor/tests/values.rs::values_of_no_entries_load`: a shared constant of no entries
  loads at a batch of three, and a per-row one of three rows of no entries loads as itself.
- `crates/skoro-core/tests/passes.rs::values_of_no_entries_through_every_pass`: a `Vector(0)`
  constant and `x.slice(0..0)` of a `Vector(2)` quantity, through a product, a difference, a matrix
  product with a `Matrix(0, 2)`, a sum and a concatenation, give the right values in the forward,
  tangent, cotangent and Hessian passes, at both precisions, and under the `wgpu` feature on a
  machine with an adapter.

## Open questions

None remain. Whether the library reduces the rows is settled in [The factor](#the-factor), and where
values of no entries are fixed in [A factor of no rows](#a-factor-of-no-rows).
