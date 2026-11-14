---
title: Linearization and elimination
summary: A density query is linearized at a batch of points into factored least-squares problems over its free quantities, by its residuals or by its information roots, and each problem is eliminated in square-root form on the host into a step, marginals, an evidence, draws, or a retirement's replacement; the Levenberg–Marquardt method iterates the steps to the mode.
order: 8
depends_on:
    [
        concepts/queries,
        concepts/the-normal,
        square-root-elimination,
        linearization,
        levenberg-marquardt,
        laplace-approximation,
        fisher-scoring,
    ]
---

Every method that reads a model as least squares rests on two pieces. Linearization replaces a
query's negative log-density near a point with a sum of squared affine residuals and a linear term
over the tangent coordinates of its free quantities. Elimination then solves the problem those rows
pose, or removes some of its variables from it. A Gauss–Newton step, a Levenberg–Marquardt step, a
Laplace approximation and a Kalman filter's update all end in them. skoro has them in two crates.
`skoro-elimination` is the note [elimination in square-root form](../../math/square-root-elimination/)
as code, numbers and matrices alone. `skoro-linearize` turns a [query](../queries/) into such
problems, reading the roles that [the normal](../the-normal/#the-roles) binds. This page shows each
piece. The mathematics is in the notes on
[linearization and the Gauss–Newton step](../../math/linearization/),
[the Levenberg–Marquardt method](../../math/levenberg-marquardt/),
[the Laplace approximation](../../math/laplace-approximation/) and
[Fisher scoring](../../math/fisher-scoring/), and why skoro made each of its choices is in
[the entry](../../archive/linearization/).

## The factored problem

A problem is a sum of factors of rows over blocks of variables, with a linear term and a constant:

$$
	m(\tangent) = c + \sum_f \tfrac12 \bigl\lvert A_f\, \tangent_{K_f} - b_f \bigr\rvert^2 + \eta\transpose \tangent .
$$

Which blocks each factor involves is its _structure_, kept apart from the numbers so that one
structure serves many problems:

```rust
{../../../../crates/skoro-elimination/src/structure.rs:10..18}
```

```rust
{../../../../crates/skoro-elimination/src/problem.rs:5..29}
```

A linear term that is absent is absent by structure, and then exactly zero, which is what keeps it
from being solved at the square of the rows' condition number when it is not there. Damping, the
$\tfrac12 \lambda\, \tangent\transpose D \tangent$ of a Levenberg–Marquardt step, is given at
elimination and appended as rows, so it leaves the structure as it is.

## Scheduling an elimination

The order of elimination, the elimination tree and the junction tree of its cliques depend on the
structure alone. A `Schedule` computes them once:

```rust
{../../../../crates/skoro-elimination/src/structure.rs:106..138}
```

The order is COLAMD on the block pattern, one row per factor and one column per block, which faer
provides. A marginal keeps some blocks, which are put last by ordering the others and appending
them. A replacement eliminates some blocks from the factors joined to them alone. Blocks are grouped
into cliques along chains whose separators nest, as
[the note's junction tree](../../math/square-root-elimination/#cliques-and-the-junction-tree)
describes, and each clique is eliminated as one front. Every row of a batch and every iteration of
a loop shares one schedule.

## Eliminating

For each clique, the rows of its factors, the new factors its children left and any damping rows
are stacked into one front, frontal columns first. The front is reduced by faer's unpivoted
Householder QR, which gives the conditional of the clique's blocks on its separator and a new factor
there, as the lemma of [eliminating one block](../../math/square-root-elimination/#eliminating-one-block)
says. The frontal blocks' linear terms are pushed onto the separator by one forward substitution.
The normal equations are never formed.

```rust
{../../../../crates/skoro-elimination/src/eliminate.rs:9..20}
```

A diagonal entry of $R$ that is small against its column refuses its block as undetermined, at a
threshold set by the precision the rows were computed at:

```rust
{../../../../crates/skoro-elimination/src/eliminate.rs:336..343}
```

> [!DECISION] The rank test follows the rows' precision
> Rows computed in single precision carry rounding of about $10^{-7}$, which a threshold fixed for
> double precision lets through as a huge step. The threshold $u^{3/4}$ of the rows' own roundoff
> sits between rounding and determination at either precision.
> [Linearization and elimination](../../archive/linearization/implementation.md) decided it.

What an elimination answers is a traversal of its conditionals:
[the minimizer, $R^{-1} z$, the log-determinant and the integral](../../math/square-root-elimination/#what-the-eliminated-problem-answers);
the marginal on the kept blocks, by one more factorization with their coordinates reversed, as a
lower-triangular whitening with a positive diagonal; and, for a replacement, what is left on the
kept blocks, reduced to at most as many rows as they have coordinates with no decision of rank,
with the constant of integrating the eliminated blocks out.

## Linearizing a query

A `Linearizer` is built once for a snapshot, a density query, a target and an assembly:

```rust
{../../../../crates/skoro-linearize/src/linearizer.rs:14..22}
```

The query's free continuous quantities, in the order of its program's layout, are the blocks. Each
term reached from one of them is a factor over the free quantities it involves. A term reached from
none is a constant of the query. The linearizer refuses by structure, naming the term or the
quantity, what the mathematics cannot linearize: a free discrete quantity, a factor that binds no
residual, and, for the residual assembly, a factor whose scale is reached from a free quantity,
whose omitted curvature is as large as what its rows keep:

```rust
{../../../../crates/skoro-linearize/src/linearizer.rs:131..196}
```

Every factor's rows come from one tangent pass of one program, whose targets are the nodes the
factors bind. A pass's frames are dense in their columns, and the identity frame of every free
quantity would carry the whole Jacobian, mostly zeros. So the seeds are compressed by colouring the
_factor structure_, the bipartite graph of
[the note](../../math/square-root-elimination/#order-fill-and-the-elimination-tree) whose nodes are
the free quantities and the factors, with an edge between a factor and each quantity it involves:
two quantities joined to a common factor get different colours, and each colour owns a few columns.
A node a term binds is reached from no quantity outside the term, so a factor's Jacobian is cut,
unmixed, from its quantities' colours:

```rust
{../../../../crates/skoro-linearize/src/linearizer.rs:352..386}
```

A factor's rows are its residual's Jacobian, and its right-hand side the negated residual. The
linear term is the gradient of the bound log-volumes that are not folded, and it is never formed as
the query's gradient less what the rows account for: that difference would be rounding, which the
solve would amplify. The constant makes the linearization agree with the query's negative
log-density at the point.

The rows are read onto the host once, and a point at which any of them is not finite is refused
there, naming the first such factor's term and the row of the batch, rather than carried into the
elimination and a step that could only be rejected. Such a point is outside a term's chart: a
rotation given a half-turn from its mean, where the [rotations' chart](../rotations-and-motions/#where-a-map-is-undefined)
is cut, or a value that is already non-finite.

```rust
{../../../../crates/skoro-linearize/src/assemble.rs:104..124}
```

### Folding a log-volume

On a curved type, the normal's log-volume is a function of its tangent alone, and its gradient lies
in the span of its own rows. It is carried by the right-hand side, $-(r + \hat c)$ with
$\hat c = W^{-\top} \nabla \lambda(s)$, rather than by the linear term. A model of normals on the
Euclidean and curved types then has no linear term at all. Folding is decided by the term's roles
and its scored quantity's type, never by a number:

```rust
{../../../../crates/skoro-linearize/src/linearizer.rs:320..350}
```

`Space::volume_reads_point` reads, from the type's declared log-volume, whether it depends on the
base point; on a positive type it does, and the log-volume stays in the linear term. $\hat c$ is
computed on the host, from the space's own log-volume and the whitening read off the information
root:

```rust
{../../../../crates/skoro-linearize/src/assemble.rs:252..319}
```

### The Fisher rows of a free scale

The residual assembly refuses a free scale. `Assembly::Fisher` takes it, replacing the Hessian in
the scale by the expected information through the root the normal binds. A factor with a free scale
gives $J^\mu = W\, \D s$ above the root's scale rows composed with the scale's Jacobian, and the
scale residual $u$ on the right, so that the scale's score is carried by the rows too:

```rust
{../../../../crates/skoro-linearize/src/assemble.rs:424..473}
```

With every scale fixed the two assemblies are the same, row for row, and one schedule serves both.

## Steps, marginals and the Laplace approximation

A `Linearized` holds one problem per row of the batch. Eliminated along a schedule, it gives a
`Solved`, whose minimizer is a step that the layout retracts onto the quantities, and whose
refusals of a block name its quantity:

```rust
{../../../../crates/skoro-linearize/src/solved.rs:38..63}
```

A Laplace approximation is these pieces on the _chart linearization_ at a mode, which subtracts the
log-volume of the charts of the quantities integrated over, so that its Gaussian is centred where
the chart is: on a log-normal quantity, at its median. Its marginals name the quantities they are
over, and which of them are curved:

```rust
{../../../../crates/skoro-linearize/src/solved.rs:28..36}
```

> [!DECISION] The precision is the rows' alone
> A Laplace approximation's precision, and a replacement's, is $J\transpose J$, without the
> curvature of the residuals or of the log-volumes. On a curved quantity it errs by a relative
> amount of the order of the noise's variance, and adding the log-volumes' curvature alone was no
> closer to the posterior. Every function that reports a covariance says so.
> [Linearization and elimination](../../archive/linearization/math.svx#what-the-covariance-leaves-out)
> decided it.

A draw takes one key per row, standard normals by the normal's quantile, and $R^{-1} z$:

```rust
{../../../../crates/skoro-linearize/src/solved.rs:137..157}
```

The Fisher linearization gives steps and covariances, and refuses an evidence, a draw and a chart
linearization: those approximate an integral of the density, whose curvature in a free scale is not
the expected information.

## Minimizing: the Levenberg–Marquardt method

A step from one linearization is Gauss–Newton's, and far from the mode it can overshoot. A
`LevenbergMarquardt` runs the damped method on every row of a batch at once, each row with its own
damping, its own point and its own stop, sharing with the others only the passes that linearize
them all:

```rust
{../../../../crates/skoro-linearize/src/minimize.rs:11..37}
```

Each iteration damps each row by Marquardt's diagonal
[](eq:math/levenberg-marquardt/marquardt), the squared norm of each coordinate's column of its rows, so
that the damping is invariant to the units of the coordinates. It eliminates the row with its own
`Damping`, retracts every row's step in one pass, and reads $f$ at the trial points by
`Linearizer::objective`, one forward pass of the query's density program. The gain
[](eq:math/levenberg-marquardt/gain), the decrease obtained over the decrease promised, decides each
row alone: a step that decreased $f$ is taken, and the damping follows Nielsen's schedule
[](eq:math/levenberg-marquardt/schedule):

```rust
{../../../../crates/skoro-linearize/src/minimize.rs:167..185}
```

A row that does not converge is not an error. Every row ends with a `Status` and the history of its
iterations, and the caller decides what a row that ran out of damping or iterations means:

```rust
{../../../../crates/skoro-linearize/src/minimize.rs:39..48}
```

> [!DECISION] A row stops on what a step promises
> The method stops a row when a step promises at most the tolerance $\epsilon$, in nats like $f$,
> whether the step is accepted or not. Once the promise is of the order of the rounding of $f$, the
> gain has a random sign; a rule that waited for an acceptance would reject steps of a converged
> row until its damping passed its limit, and report it as failed. A step's promise is invariant to
> the chart's coordinates, where a gradient's norm is not.
> [Examples that run](../../archive/examples/math.svx#a-levenberg-marquardt-method) decided it.

The damped promise bounds the distance to the mode only while the damping is small, so stationarity
is checked apart from the stop, by `decrement`: the undamped promise
$\tfrac12 \lvert J \tangent_0 \rvert^2$, which is zero exactly where the query's gradient is.

```rust
{../../../../crates/skoro-linearize/src/minimize.rs:195..219}
```

## A retirement's replacement

A [retirement](../terms/#retirement) removes some quantities and needs a replacement for the terms joined to
them, the integral of those terms over the quantities it integrates. A `Replacing` linearizes the
query of those terms alone, free in the quantities to integrate and in the blanket quantities to
keep, with the retirement's own observed values:

```rust
{../../../../crates/skoro-linearize/src/replacing.rs:120..158}
```

The replacement is the [improper factor](../the-normal/#an-improper-factor) of the normal, in the
kept quantities' stacked tangent at the linearization point, built in the retirement's scope:

```rust
{../../../../crates/skoro-linearize/src/replacing.rs:186..228}
```

Its number of rows, and whether it has a linear term, are fixed by the retired terms and their
types, so every row of a batch has a replacement of one shape. For linear-Gaussian terms it is
exact, and retiring one state at a time from a chain is the Kalman filter. Otherwise, linearized at
the retired terms' mode it is an iterated extended Kalman filter's step. Where the retired terms do
not determine the blanket, the replacement keeps what they say, and replacements over the same
quantity add until they determine it.
