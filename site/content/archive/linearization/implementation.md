---
summary: Two crates, `skoro-elimination`, which eliminates factored least-squares problems in square-root form on the host, and `skoro-linearize`, which linearizes a query at a batch of points into such problems, by its residuals or by its information roots, and computes a retirement's replacement as an improper Gaussian; with the few additions to the core and the normal that they need.
depends_on: [archive/linearization/math, square-root-elimination, archive/normal/implementation, archive/improper-gaussian/implementation]
---

## Overview

When this is built, two crates bring back the two pieces of
[the mathematics](math.svx#the-problem).

- **`skoro-elimination`** is the note [elimination in square-root form](../../math/square-root-elimination/)
  as code, and nothing else: a problem [](eq:math/square-root-elimination/factored) of blocks,
  factors of rows and a linear term, its symbolic factorization (an order and a junction tree,
  computed from the factor structure alone), and its numeric elimination by unpivoted Householder
  QR on the host, in double precision, through faer. It knows nothing of models.
- **`skoro-linearize`** turns a query of a model into such problems. A `Linearizer` is built once
  for a snapshot, a density query, a target and an assembly, the residual linearization of
  [%full](statement:archive/linearization/math/linearization) or the Fisher linearization of
  [%full](statement:archive/linearization/math/fisher); it refuses by structure what the
  mathematics refuses, and plans the elimination once. At each batch of points it gives one
  problem per row. A `Replacing` computes a retirement's replacement, the improper Gaussian
  [](eq:archive/linearization/math/information-form) of
  [%full](statement:archive/linearization/math/replacement), and builds it with the normal's
  `Improper`.

A Gauss–Newton loop, which the entry does not write but its tests do, reads:

```rust
use skoro::{Query, Target};
use skoro_linearize::{Assembly, Linearizer};

let snapshot = writer.snapshot();
let query = Query::log_density().given(observations.clone());
let target = Target::host();
let linearizer = Linearizer::new(&snapshot, &query, &target, Assembly::Residual)?;
let schedule = linearizer.schedule()?; // the order and the junction tree, once
let mut point = initial;       // the values of the free quantities, a batch of rows
for _ in 0..20 {
    let linearized = linearizer.at(&point)?;   // one problem per row
    let solved = linearized.eliminate(&schedule, None)?;
    let step = solved.minimizer(&target)?;     // `Points`, [rows, D]
    point = linearizer.layout().retract(&point, &step, &target)?;
}
```

and a retirement whose replacement is linearized at the point `at`:

```rust
use skoro::Addresses;
use skoro_linearize::Replacing;

let retirement = writer.retire(Addresses::new().under("tick/0"), observed.clone())?;
let replacing = Replacing::new(&retirement, &[x1.untyped()], &Observations::new(), &target)?;
let linearized = replacing.at(&at)?;           // rows, right-hand side, linear term, constant
retirement.replace(|_| linearized.build())?;   // the `Improper` factor, its roles bound
```

The pieces map onto the mathematics as follows.

| mathematics | code |
| --- | --- |
| the linearization [%full](statement:archive/linearization/math/linearization), its rows, right-hand sides, $\eta$ and $c_0$ | `Linearizer::at`, a `Problem` per row |
| the chart linearization, with $\lambda_{\hat \point}$ | `Linearizer::at_in_chart` |
| the factor structure, [](eq:archive/linearization/math/variables) | `Linearizer::structure`, a `Structure` |
| the step [](eq:archive/linearization/math/step) | `Solved::minimizer` |
| damping, and the promised decrease [](eq:archive/linearization/math/promised) | `Damping`, `Problem::promised` |
| the Laplace approximation's table ([The Laplace approximation](math.svx#the-laplace-approximation)) | `Solved::{minimizer, marginal, draw, log_integral}` on a chart linearization |
| the replacement [](eq:archive/linearization/math/improper), [](eq:archive/linearization/math/information-form) | `Replacing`, `LinearizedReplacement::build` |
| the Fisher linearization [%full](statement:archive/linearization/math/fisher), [](eq:archive/linearization/math/fisher-matrix) | `Assembly::Fisher` |
| the refusals of [Linearizing a query](math.svx#linearizing-a-query) and [Information roots](math.svx#information-roots) | `skoro_linearize::Error` |

Building them asks a little of the crates beside them, all of it public API that a feature crate
needs and the core and the normal do not yet expose ([Additions to the core and the
normal](#additions-to-the-core-and-the-normal)): the core gives a query's terms and observations,
the union of two sets of observations, the quantities a node is reached from, whether a space's
chart volume reads its point, and a constant of a space known at run time; the normal binds the tangent $\point \ominus \mu$ of its
term under a role, and stacks a block of a space known at run time.

## Constraints

Read at commit `b69134a`. Each interface is pasted from the file named, and followed by what the
design must do because of it.

### Queries, programs and passes

`crates/skoro-core/src/query.rs`:

```rust
pub struct Query {
    pub(crate) targets: Targets,
    pub(crate) observations: Observations,
}

impl Query {
    pub fn of(targets: impl IntoIterator<Item = QDyn>) -> Query;
    pub fn log_density() -> Query;
    pub fn log_density_of<T: Into<NodeId>>(terms: impl IntoIterator<Item = T>) -> Query;
    pub fn given(self, observations: Observations) -> Query;
}
```

A linearization is of a density query, and needs to know which terms the query sums and what it
observes, to compile programs of its own that target the terms' bound nodes under the same
observations. Neither is readable outside the core. The design adds two accessors.

`crates/skoro-core/src/compile.rs` and `crates/skoro-core/src/program.rs`:

```rust
impl Snapshot {
    /// The program of `query` on `target`: from the snapshot's cache, under the query's targets,
    /// its observation set's identity and the target, or compiled and cached.
    pub fn compile(&self, query: &Query, target: &Target) -> Result<Arc<Program>, Error>;
}

/// How many programs a snapshot keeps, the most recently used.
const CACHED: usize = 32;

impl Program {
    pub fn free(&self) -> &[Free];
    pub fn layout(&self) -> &Layout;
    pub fn refuse_free_discrete(&self) -> Result<(), Error>;
    pub fn forward(&self, at: &Values) -> Result<Values, Error>;
    /// Tangent frames at free continuous quantities in, at the targets out. A free quantity given
    /// no frame has a zero one.
    pub fn tangent(&self, at: &Values, tangents: &Frames) -> Result<(Values, Frames), Error>;
    pub fn cotangent(&self, at: &Values, seeds: &Frames) -> Result<(Values, Frames), Error>;
    pub fn gradient(&self, at: &Values) -> Result<(Values, Frames), Error>;
}
```

Every frame of one pass has the same number of columns (`Program::columns` checks it), and a frame
of one row is shared by every row. A program is cached under its query's targets and observation
set, 32 to a snapshot, so the design compiles a fixed, small number of programs per linearizer and
never one per term: a model of more than 32 terms would otherwise evict its own programs on every
iteration. A pass's frames are dense in their columns, so the number of columns is the cost to
keep small ([The passes](#the-passes)).

`crates/skoro-core/src/layout.rs`:

```rust
impl Layout {
    pub fn quantities(&self) -> &[NodeId];
    pub fn dimension(&self) -> usize;
    /// The product's blocks: each quantity's ranges in storage and in tangent coordinates.
    pub fn blocks(&self) -> Vec<Block>;
    /// The step of an estimator: the quantities' values stacked into one point of the product,
    /// moved by the product's retraction along `tangent`, `[rows, dimension]`, on `target`, and
    /// sliced back.
    pub fn retract(&self, at: &Values, tangent: &Points, target: &Target) -> Result<Values, Error>;
}

impl Frames {
    pub fn identity(layout: &Layout) -> Frames;
}
```

The program's layout is the order of the free continuous quantities, ancestral, and its blocks are
the blocks of the elimination: block $k$ of a `Structure` is `layout.quantities()[k]`, of dimension
`blocks()[k].tangent.len()`. A step is retracted by `Layout::retract`, which wants a `Points` of
`[rows, D]` on the target.

### Terms, roles and the compilation graph

`crates/skoro-core/src/term.rs` and `crates/skoro-core/src/model.rs`:

```rust
pub struct Term {
    pub log_density: NodeId,
    /// `s_t`: the quantity the term scores, `None` for a likelihood factor.
    pub scores: Option<NodeId>,
    /// `V_t`, sorted: the quantities the log-density is reached from, with the one it scores.
    pub involves: Box<[NodeId]>,
    pub sampler: Option<Sampler>,
    pub(crate) bindings: Vec<(&'static str, NodeId)>,
}

impl Term {
    pub fn bound<M: Multiplicity>(&self, role: &Role<M>) -> Result<M::Handles, Error>;
}

/// The quantities among the roots `id` is reached from, walking back through mapped nodes and
/// stopping at nodes fixed in the model.
pub(crate) fn roots(model: &Model, id: NodeId) -> Vec<NodeId>;

impl Snapshot {
    pub fn terms(&self) -> Vec<&Term>;
    pub fn term(&self, id: impl Into<NodeId>) -> Option<&Term>;
    pub fn space(&self, id: impl Into<NodeId>) -> Option<&Space>;
    pub fn node(&self, id: impl Into<NodeId>) -> Option<&Node>;
}
```

A term's variables $K_t$ [](eq:archive/linearization/math/variables) are `involves` less the
quantities the query observes, which is what the program's layout lists. The third refusal asks
whether the scale a term binds is reached from a free quantity, which is `roots` of the scale's
node: private to the core. Compiling `Query::of([scale])` for each term would answer it with public
API, but one program per term is what the cache cannot hold. The design makes `roots` public,
under a name of its own.

`crates/skoro-normal/src/roles.rs`:

```rust
pub const RESIDUAL: Role<ExactlyOne> = Role::new("skoro.residual");
pub const MEAN: Role<ExactlyOne> = Role::new("skoro.normal.mean");
pub const ISOTROPIC: Role<AtMostOne> = Role::new("skoro.normal.scale.isotropic");
pub const DIAGONAL: Role<AtMostOne> = Role::new("skoro.normal.scale.diagonal");
pub const COVARIANCE_FACTOR: Role<AtMostOne> = Role::new("skoro.normal.scale.covariance_factor");
pub const WHITENING: Role<AtMostOne> = Role::new("skoro.normal.scale.whitening");
/// The varying part of a term's remainder: what is left of its log-density after the squares of
/// its residual and a normalizer constant in its free quantities, where that is not identically
/// zero. On a normal it is the chart's log-volume `λ_μ(x ⊖ μ)`, and on an improper factor its
/// linear part `ηᵀδ`. [...]
pub const LOG_VOLUME: Role<AtMostOne> = Role::new("skoro.log_volume");
/// The root `C` of the expected information, on a term that scores a quantity.
pub const INFORMATION_ROOT: Role<AtMostOne> = Role::new("skoro.information_root");
pub fn scale(form: Form) -> &'static Role<AtMostOne>;
```

The roles are the normal's, so `skoro-linearize` depends on `skoro-normal`. `LOG_VOLUME`'s
description already says what [](eq:archive/linearization/math/split) asks of $\lambda_t$, the
varying part of the remainder, so the role keeps its name ([the mathematics left it to this
document](math.svx#blanket-quantities-the-rows-do-not-determine)): renaming a role every bound
term carries would buy a name and nothing else.

`crates/skoro-normal/src/normal.rs`, where a normal's term binds them:

```rust
let parts = self.log_density(x);
let scale = self.scale.value();
let mut term = score(x.narrow::<Scored>(), parts.log_density)
    .sampler(/* ... */)
    .bind(&roles::RESIDUAL, parts.residual)
    .bind(&roles::MEAN, self.mean)
    .bind(roles::scale(form), scale)
    .bind(&roles::INFORMATION_ROOT, root.call(&[scale]));
if let Some(log_volume) = parts.log_volume {
    term = term.bind(&roles::LOG_VOLUME, log_volume);
}
```

with `parts.tangent` the node $s_t = \point \ominus \mu$, which is bound to nothing. Three things
need it, and none can be had otherwise:

- the folded right-hand side $\hat c_t = W_t^{-\top} \nabla \lambda_t(s_t)$ needs the value of
  $s_t$, at which the chart's log-volume is differentiated;
- the Fisher linearization's $J^\mu_t = W_t \D s_t$, for a term whose scale is free, needs the
  Jacobian of $s_t$ with the scale held, which no bound node gives: the residual's Jacobian mixes
  in the scale's own derivative, and the mean's misses the scored quantity;
- folding is a promise about $s_t$, that the residual is $W_t s_t$ and the log-volume
  $\lambda_\mu(s_t)$, and a promise belongs to a role, not to a reader that recognises a normal by
  its other bindings.

The design adds the role.

`crates/skoro-normal/src/composites.rs`, the root the normal binds:

```rust
/// `skoro.normal.root`: the root `C` of the expected information, of input the form's space, to
/// `Matrix(d + p, 2d + k)`, `p = d(d+1)/2` and `k` the dimension of the form's space. Its rows
/// are the `d` residual rows, then `(a, a)`, then `(a, b)` for `a > b` in the triangular
/// storage's order; its columns `x`, `μ`, then the scale's tangent coordinates.
pub(crate) fn root(form: Form, d: usize) -> Result<Arc<Composite>, Error>;
```

and its body, which puts $W$ in the columns of $\point$:

```rust
// The columns of x and μ: (W; 0) and (-W; 0).
```

So $W_t$ is the top-left $d \times d$ block of $C_t$'s value, lower triangular in every form, and
$C_{\theta, t}$ is the block of rows $d..d+p$ and columns $2d..2d+k$. A `Matrix` is stored column
by column. The scale residual $u_t$ is indexed as those rows: the $d$ diagonal pairs, then the
pairs $(a, b)$, $a > b$, column $b$ after column $b - 1$.

### Spaces and their volumes

`crates/skoro-core/src/retraction.rs` and `crates/skoro-core/src/embedded.rs`:

```rust
impl Space {
    /// The composite of `λ_x(t)`, the log-volume of the chart at `x`, of inputs this space and
    /// `Vector(dimension)`, to `Real` (`NotContinuous` for a discrete space, `NoVolume` for one
    /// without a measure).
    pub fn log_volume(&self) -> Result<Arc<Composite>, Error>;
    pub fn has_measure(&self) -> bool;
}

pub struct Volume { log_volume: Arc<Composite>, domain: Domain }

impl Embedded {
    pub fn volume(&self) -> Option<&Volume>;
}

impl Composite {
    pub fn compile(&self, target: &Target) -> Result<Program, Error>;
}
```

with the table at the head of `retraction.rs`:

```text
| `Positive` | `sum(positive.log(x) + t)` | global |
| `Embedded`, with a volume | `Λ(include(x), t)` | the declaration's, `Ω(include(x), t)` |
```

The chart linearization's $\lambda_{\hat \point}(0)$ and $\nabla \lambda_{\hat \point}(0)$, and the
folded $\nabla \lambda_t(s_t)$, are the value and the gradient of `Space::log_volume()`'s composite,
compiled alone, at a point and a tangent. Folding needs to know whether $\lambda_x(t)$ reads $x$:
on a positive type it does, on the library's curved types (`skoro.spatial.rotation`'s
`-3 log(1 + |φ|²/16)` and the test sphere's `-2 log(1 + |t|²/4)`, each reading only `q[1]`) it does
not. Nothing in the core says so; the design adds the question to `Space`, answered from the
composite's body.

### Retirement and the improper factor

`crates/skoro-core/src/retire.rs`:

```rust
pub struct Plan {
    pub retired: Vec<QDyn>,
    pub observed: Observations,
    pub integrated: Vec<QDyn>,
    pub terms: Vec<Arc<Term>>,
    pub blanket: Vec<(QDyn, Space, Address)>,
    pub unproposed: Vec<QDyn>,
}

impl Retirement<'_> {
    pub fn plan(&self) -> &Plan;
    pub fn snapshot(&self) -> &Snapshot;
    pub fn replace(self, body: impl FnOnce(&Plan) -> Replacement)
        -> Result<Option<TermHandle>, Failure>;
}
```

`replace` consumes the retirement and runs `body` inside a building scope. Everything numeric is
therefore done before it, on `retirement.snapshot()`, and the body only makes constants and nodes.
The blanket comes as `(QDyn, Space, Address)`: spaces known at run time, with no marker.

`crates/skoro-normal/src/stacked.rs` and `crates/skoro-normal/src/improper.rs`:

```rust
impl Stacked {
    pub fn block<T: Declares>(mut self, x: Q<T>, marker: T, anchor: impl IntoHostValue<T>)
        -> Stacked;
    /// [...] `a` the `rows × D` matrix `A`, **column by column** as a `MatrixDyn` is stored [...]
    /// **The order of the blocks is load-bearing.** The columns of `A`, and the entries of `η`,
    /// follow the order of the `block` calls [...]
    pub fn improper(&self, rows: usize, a: impl IntoHostValue<MatrixDyn>,
        b: impl IntoHostValue<VectorDyn>, c: impl IntoHostValue<Real>) -> Improper;
}

impl Improper {
    /// The same factor with the linear part `ηᵀδ`, `eta` a `Vector(D)`, shared or per row.
    pub fn linear(self, eta: impl IntoHostValue<VectorDyn>) -> Improper;
    pub fn replacement(&self) -> Replacement;
}
```

`block` wants a marker that declares the space, and a blanket quantity of a product or curved type
has none to hand (`ProductDyn` is not `Declares`), nor does the core make a constant from a
`Space` and a `HostValue` (`crates/skoro-core/src/scope.rs` has `constant<T: Declares>` alone). The
design adds `constant_in` to the core and `block_in` to `Stacked`. The order of the blocks, and
that `A` is stored column by column, are kept by `Replacement::build`, which takes both from one
place.

### faer

The charter puts the solves where conditioning lives on the host, through faer
([position 11](../charter/writeup.md#11-burn-behind-a-layer-of-skoro-s-own-precision-chosen-per-query-solves-on-the-host)).
faer 0.24.4 is the current release (its minimum Rust version, 1.84, is below the workspace's 1.95),
and it has what the elimination needs, at the paths the archive used:

```rust
// faer-0.24.4/src/linalg/solvers.rs
pub struct Qr<T> { /* unpivoted Householder QR */ }
impl<T: ComplexField> Qr<T> {
    pub fn thin_R(&self) -> MatRef<'_, T>;
}

// faer-0.24.4/src/sparse/linalg/colamd.rs
/// computes the approximate minimum degree ordering for reducing the fill-in
/// during the sparse qr factorization of a matrix with the sparsity pattern of $A$
pub fn order<I: Index>(perm: &mut [I], perm_inv: &mut [I], A: SymbolicSparseColMatRef<'_, I>,
    control: Control, stack: &mut MemStack) -> Result<(), FaerError>;
```

faer has no constrained COLAMD.

## Design

### The crates

```text
crates/skoro-elimination/   numbers only: faer, and nothing of skoro
    src/lib.rs              Matrix, Structure, Problem, Damping, re-exports
    src/structure.rs        the factor structure, the order, the junction tree: Schedule
    src/eliminate.rs        the numeric elimination: Elimination, its conditionals and constants
    src/answers.rs          minimizer, solves, marginal, integral, draws, the reduced factor
crates/skoro-linearize/     skoro-core, skoro-normal, skoro-elimination
    src/lib.rs              re-exports, Error
    src/linearizer.rs       Linearizer: refusals, factors, folding, the programs and the seeds
    src/assemble.rs         Linearized: the passes at a point, and each row's Problem
    src/solved.rs           Solved: a batch's eliminations, mapped back to quantities
    src/replacing.rs        Replacing and LinearizedReplacement
```

Two crates, as the archive had (`skoro-dense` and `skoro-linearize`), and for a reason of the
mathematics rather than of dependencies: the note says the elimination "is a note about numbers
and matrices alone, and nothing in it depends on where the rows came from", and a crate with no
skoro dependency keeps it so, and is tested against the note's examples alone. The alternative, one
crate with an `elimination` module, would hold the same line by convention only. The charter's
"each later feature is a crate of its own" ([position 12](../charter/writeup.md#12-two-crates-and-an-umbrella))
allows either; the umbrella `skoro` re-exports neither, as it does not re-export the normal.

Both crates use the workspace's lints and edition. `skoro-elimination` depends on
`faer = { version = "0.24.4", default-features = false, features = ["std", "linalg", "sparse-linalg"] }`,
added to the workspace's dependencies. `skoro-linearize` depends on `skoro-tensor`, `skoro-core`,
`skoro-normal` and `skoro-elimination`, and has `skoro-spatial` as a dev-dependency for its tests on
rotations.

> [!DECISION] faer, not the note's reflections by hand
> The normal factors its scales with a dozen lines of its own, since faer would have been a
> dependency for nothing else. Here the factorizations are the crate: blocked Householder QR of
> fronts of any size, triangular solves, and an approximate minimum-degree order. faer gives all
> three, tested and fast, and the charter names it for exactly these solves. Writing them by hand
> would reproduce [%full](statement:math/square-root-elimination/triangular-root) faithfully and
> slowly, and COLAMD not at all. faer is behind the crate's own `Matrix`, so no faer type is
> public and a faer upgrade touches `skoro-elimination` alone.

### Additions to the core and the normal

Each is small, public, and tested where it lives. They change `skoro-core` and `skoro-normal`, and
this entry lists both among its crates at publish.

**The core.**

```rust
// crates/skoro-core/src/query.rs
impl Query {
    /// The observations the query is given.
    pub fn observations(&self) -> &Observations;
    /// The terms whose log-densities a density query sums, in the order they were added: every
    /// term of `snapshot` for `log_density()`, the named ones for `log_density_of`; `None` for a
    /// query of nodes.
    pub fn terms(&self, snapshot: &Snapshot) -> Option<Vec<NodeId>>;
}

// crates/skoro-core/src/query.rs
impl Observations {
    /// This set with the values of `other` for the quantities this set does not observe: a new
    /// set. Where both observe a quantity, this set's value is kept.
    pub fn union(self, other: &Observations) -> Observations;
}

// crates/skoro-core/src/model.rs
impl Snapshot {
    /// The quantities `id` is reached from, walking back through mapped nodes and stopping at
    /// nodes fixed in the model, sorted: what `Term::involves` holds for a log-density.
    pub fn reached_from(&self, id: impl Into<NodeId>) -> Vec<NodeId>;
}

// crates/skoro-core/src/composite.rs
impl Composite {
    /// Whether the output is reached from the input at `index` through the body. An operation is
    /// taken to read every input it is given, a composite call among them, so the answer errs
    /// towards `true`.
    pub fn reads(&self, index: usize) -> bool;
}

// crates/skoro-core/src/retraction.rs
impl Space {
    /// Whether `λ_x(t)` depends on the point `x`: false on the Euclidean spaces, whose log-volume
    /// is zero, true on `Positive`, the factors' on a product (true if any factor's is), and on
    /// an embedded type whether its declared `Λ` reads its first input (`NotContinuous`,
    /// `NoVolume` as `log_volume`).
    pub fn volume_reads_point(&self) -> Result<bool, Error>;
}

// crates/skoro-core/src/scope.rs
/// A constant of `space` (`ValueShape` when the value does not fit it), for a space known only at
/// run time.
#[track_caller]
pub fn constant_in(space: Space, value: HostValue) -> QDyn;
```

`reached_from` is the private `roots`, under the name its meaning has: in the compilation graph,
the quantities among the roots from which a node is reached. `reads` walks the body's
model from the output through `Node::inputs`, as `roots` does, and is conservative where it cannot
look: a composite called inside the body is taken to read all its inputs. That errs towards not
folding, which is always correct ([Folding](#folding)). `constant_in` is `constant` with the space
given rather than declared.

**The normal.**

```rust
// crates/skoro-normal/src/roles.rs
/// The tangent `s = x ⊖ μ` of a normal's term, which scores a quantity: its residual is `W s`,
/// with `W` the whitening of the bound scale, and its log-volume, where it binds one, is
/// `λ_μ(s)`, the chart's log-volume of the scored quantity's space at `μ`.
pub const TANGENT: Role<AtMostOne> = Role::new("skoro.normal.tangent");

// crates/skoro-normal/src/stacked.rs
impl Stacked {
    /// `x ⊖ p` for a quantity of `space`, known at run time, with the constant anchor `anchor`,
    /// shared or per row: `block` without a marker (`ValueShape`, `NotContinuous`).
    #[track_caller]
    pub fn block_in(self, x: QDyn, space: &Space, anchor: HostValue) -> Stacked;
}
```

`Normal::score` binds `parts.tangent` under `TANGENT`. `factor_at` does not: it scores nothing, its
scale must be fixed wherever this entry reads it, and it binds no log-volume, so nothing would read
the role on it.

### The elimination

`skoro-elimination` implements the note. Its public types:

```rust
/// A dense matrix, row by row: the crate's own, so that no faer type is public.
#[derive(Clone, PartialEq, Debug)]
pub struct Matrix { rows: usize, columns: usize, data: Vec<f64> }

impl Matrix {
    pub fn new(rows: usize, columns: usize, data: Vec<f64>) -> Option<Matrix>;
    pub fn zeros(rows: usize, columns: usize) -> Matrix;
    pub fn rows(&self) -> usize;
    pub fn columns(&self) -> usize;
    pub fn get(&self, row: usize, column: usize) -> f64;
    pub fn data(&self) -> &[f64];
}

/// The factor structure of [](eq:math/square-root-elimination/factored): blocks of given
/// dimensions, and for each factor the blocks it involves, in increasing order.
#[derive(Clone, PartialEq, Debug)]
pub struct Structure { dimensions: Vec<usize>, factors: Vec<Vec<usize>> }

impl Structure {
    /// Refuses a factor that names a block twice or one that does not exist
    /// (`Error::Structure`).
    pub fn new(dimensions: Vec<usize>, factors: Vec<Vec<usize>>) -> Result<Structure, Error>;
    pub fn dimension(&self) -> usize; // D, the sum of the blocks'
    pub fn blocks(&self) -> &[usize];
    pub fn factors(&self) -> &[Vec<usize>];
}

/// One factor's rows: `A_f`, its columns those of its blocks in the structure's order, and `b_f`.
#[derive(Clone, PartialEq, Debug)]
pub struct Rows { pub a: Matrix, pub b: Vec<f64> }

/// A problem [](eq:math/square-root-elimination/factored) with a constant:
/// `c + Σ_f ½|A_f τ_{K_f} - b_f|² + ηᵀτ`.
#[derive(Clone, Debug)]
pub struct Problem {
    structure: Arc<Structure>,
    factors: Vec<Rows>,
    /// `η`, stacked over all blocks, or `None`: absent by structure, and then exactly zero.
    linear: Option<Vec<f64>>,
    constant: f64,
    /// The unit roundoff of the precision the rows were computed at, for the rank test.
    roundoff: f64,
}

impl Problem {
    /// Refuses rows whose shapes do not fit the structure (`Error::Shape`).
    pub fn new(structure: Arc<Structure>, factors: Vec<Rows>, linear: Option<Vec<f64>>,
        constant: f64, roundoff: f64) -> Result<Problem, Error>;
    /// `c + Σ ½|A_f τ - b_f|² + ηᵀτ` at `tangent`.
    pub fn value(&self, tangent: &[f64]) -> f64;
    /// `½|J τ|² + λ τᵀ D τ`, [](eq:archive/linearization/math/promised).
    pub fn promised(&self, tangent: &[f64], damping: &Damping) -> f64;
}

/// `½ λ τᵀ D τ` of [](eq:math/square-root-elimination/damping): `λ` and the positive diagonal
/// `D`, one entry per coordinate.
#[derive(Clone, Debug)]
pub struct Damping { pub lambda: f64, pub diagonal: Vec<f64> }
```

**The schedule.** The symbolic factorization of
[Order, fill and the elimination tree](../../math/square-root-elimination/#order-fill-and-the-elimination-tree):

```rust
/// The symbolic factorization: an order, an elimination tree and the junction tree of its
/// cliques, for one structure and one set of blocks to eliminate. Shared by every problem of the
/// structure.
#[derive(Clone, Debug)]
pub struct Schedule { structure: Arc<Structure>, order: Vec<usize>, cliques: Vec<Clique>, /* ... */ }

struct Clique {
    /// Eliminated together, consecutive in the order.
    frontals: Vec<usize>,
    separator: Vec<usize>,
    parent: Option<usize>,
    /// The structure's factors first collected here.
    factors: Vec<usize>,
    children: Vec<usize>,
}

impl Schedule {
    /// Every block, in a COLAMD order of the block pattern.
    pub fn all(structure: &Arc<Structure>) -> Result<Schedule, Error>;
    /// Every block but `keep`, which are kept: the schedule of a marginal on `keep`.
    pub fn keeping(structure: &Arc<Structure>, keep: &[usize]) -> Result<Schedule, Error>;
    /// The blocks `eliminate` alone, from the factors joined to them: the schedule of a
    /// replacement.
    pub fn eliminating(structure: &Arc<Structure>, eliminate: &[usize])
        -> Result<Schedule, Error>;
    /// A given order, for the tests of order-independence.
    pub fn in_order(structure: &Arc<Structure>, order: Vec<usize>) -> Result<Schedule, Error>;
    pub fn order(&self) -> &[usize];
    pub fn cliques(&self) -> usize;
}
```

COLAMD runs on the block pattern, one row per factor and one column per block, as the archive and
GTSAM do: every scalar column of a block has the block's pattern, so ordering blocks is as good and
cheaper. `keeping` orders the pattern with the kept columns removed and appends them, as the
archive did for want of a constrained COLAMD; that ignores the fill among the kept blocks, which
are eliminated last in one front whatever their order. The elimination tree parents each block to
the first block of its separator in the order, and the cliques are its chains under
[](eq:math/square-root-elimination/clique-chain), each block joining its parent's clique when the
equation holds, children before parents. A structure depends on the query and not on any number,
so a schedule serves every row of a batch and every iteration of a loop, and both assemblies.

**The numeric elimination.** For each clique in order, the rows of its factors, the new factors its
children left, and the damping rows of its frontals ($\sqrt{\lambda D_k}$, diagonal, right-hand side
zero), are stacked into one `[A_front | A_sep | b]`, frontal columns first, and reduced by faer's
unpivoted `Qr`, as [%full](statement:math/square-root-elimination/eliminate-one) says. The linear
terms of the frontal blocks are pushed by one forward substitution $R\transpose y = \eta_k$, which
gives $\gamma = y\transpose d - \tfrac12 \lvert y \rvert^2$ and the separator's new linear term
$\eta_S - T\transpose y$. Pivoting would mix frontal and separator columns, so it is never used.
A frontal row whose diagonal entry is negative has its sign changed, row and right-hand side
together, so that every $R$ has a positive diagonal.

```rust
/// What an elimination leaves: a conditional per clique, the factors left on the blocks it did not
/// eliminate, and the constants.
#[derive(Clone, Debug)]
pub struct Elimination {
    schedule: Schedule,
    /// `R τ_f + T τ_s = d - y`, per clique, `R` upper triangular with a positive diagonal.
    conditionals: Vec<Conditional>,
    /// The new factors on the kept blocks, and the original factors joined to none eliminated.
    left: Vec<(Vec<usize>, Rows)>,
    left_linear: Option<Vec<f64>>,
    /// `c + Σγ + ½|e|²`: the constant of what is left.
    constant: f64,
    /// `Σ log R_ii` over the eliminated blocks, and their total dimension.
    log_determinant: f64,
    eliminated: usize,
}

impl Problem {
    /// Eliminates the schedule's blocks, with the damping rows appended if given. Refuses a block
    /// its rows do not determine (`Error::Undetermined { block }`).
    pub fn eliminate(&self, schedule: &Schedule, damping: Option<&Damping>)
        -> Result<Elimination, Error>;
}
```

Minimizing and integrating leave the same factors and differ in a constant
([Eliminating one block](../../math/square-root-elimination/#eliminating-one-block)), so there is
one elimination, and each answer reads the constant it needs.

**The rank test.** A frontal diagonal entry with

$$
	\lvert R_{ii} \rvert \le u^{3/4} \lvert a_i \rvert ,
$$

where $a_i$ is the column of the stacked front before its reduction and $u$ the problem's
`roundoff`, refuses its block as undetermined. An exact dependence, a gauge freedom or a quantity
no row reaches, leaves a diagonal entry at the level of the rows' rounding, a small multiple of
$u \lvert a_i \rvert$; a determined direction leaves one of order $\lvert a_i \rvert / \kappa$. The
threshold $u^{3/4}$ sits between them with a margin of $u^{-1/4}$ on the side of rounding: about
$1.1 \times 10^{-12}$ for rows computed in double precision, where the archive's fixed $10^{-12}$
was, and about $3.8 \times 10^{-6}$ for rows computed in single precision, where the archive's
threshold let an exact dependence through as a huge step, as its own comment said. A direction
determined to fewer than a quarter of the rows' digits is refused with the undetermined ones; at
single precision that is a condition number above about $2.6 \times 10^5$, where the step would
have been noise.

**What the elimination answers.** Each is
[a traversal of the elimination tree](../../math/square-root-elimination/#what-the-eliminated-problem-answers):

```rust
impl Elimination {
    /// The minimizer of a complete elimination, back-substituted from the roots: `τ*`.
    pub fn minimizer(&self) -> Result<Vec<f64>, Error>;
    /// `min φ`: the constant left by a complete elimination.
    pub fn minimum(&self) -> Result<f64, Error>;
    /// `log ∫ e^{-φ}` of a complete elimination, [](eq:math/square-root-elimination/integral):
    /// `(D/2) log 2π - Σ log R_ii - min φ`.
    pub fn log_integral(&self) -> Result<f64, Error>;
    /// `R⁻¹ z`, from the roots to the leaves: a draw from `N(0, (AᵀA)⁻¹)` for standard normal `z`.
    pub fn solve(&self, z: &[f64]) -> Result<Vec<f64>, Error>;
    /// `Σ log R_ii`: half the log-determinant of `AᵀA` over the eliminated blocks.
    pub fn log_determinant(&self) -> f64;
    /// The marginal on the kept blocks of a `Schedule::keeping` elimination, by one more QR of what is
    /// left, with the kept coordinates reversed, as the note says
    /// ([%full](statement:math/square-root-elimination/marginal)): refuses kept blocks the rows do
    /// not determine.
    pub fn marginal(&self) -> Result<Marginal, Error>;
    /// What is left of a `Schedule::eliminating` elimination, reduced by one more QR with no pivoting
    /// and no decision of rank, with the constant of integrating the eliminated blocks out:
    /// [](eq:archive/linearization/math/improper).
    pub fn reduced(&self) -> Reduced;
}

/// A Gaussian marginal: `W` lower triangular with a positive diagonal over the kept blocks, in
/// their order, `W τ ≈ d`, and its mean `W⁻¹ d`.
pub struct Marginal { pub blocks: Vec<usize>, pub whitening: Matrix, pub mean: Vec<f64> }

impl Marginal {
    pub fn covariance(&self) -> Matrix; // W⁻¹ W⁻ᵀ, by triangular solves
    pub fn log_determinant(&self) -> f64; // Σ log W_ii
}

/// `-½|A τ - b|² - ηᵀτ + c` over the kept blocks: [](eq:archive/linearization/math/information-form)
/// in numbers.
pub struct Reduced {
    pub blocks: Vec<usize>,
    /// `min(m', D)` rows by `D` columns.
    pub a: Matrix,
    pub b: Vec<f64>,
    pub linear: Option<Vec<f64>>,
    pub constant: f64,
}
```

`minimizer`, `minimum`, `log_integral` and `solve` refuse an elimination that kept blocks
(`Error::Incomplete`). A minimizer with a linear term is computed as the note computes
[](eq:math/square-root-elimination/complete-square): the pushes $y$ are already in the conditionals'
right-hand sides, so back substitution gives it, the part from $\eta$ at $\kappa(J)^2$ and the rest
at $\kappa(J)$. The constant of `reduced` is
[](eq:archive/linearization/math/improper) with $c_0$ the problem's constant: $\tfrac{d_E}{2}
\log 2\pi$, less the log-determinant of the eliminated blocks, less the constant left, which holds
$c_0$, $\Gamma$ and both residuals $\tfrac12 \lvert e \rvert^2$, the fronts' and the last QR's.

The number of rows of `Reduced::a` is decided by shapes alone: the factors left have $m'$ rows in
all, and the last QR keeps $\min(m', D)$ of them whatever the numbers.
`linear` is present exactly when the problem's was or a front pushed one, which depends on whether
the problem's was present, not on its values: a problem with `linear: None` has no pushes.

**Errors.**

```rust
#[derive(Clone, Debug, PartialEq)]
#[non_exhaustive]
pub enum Error {
    /// A structure that names a block twice in a factor, or a block that does not exist.
    Structure { factor: usize },
    /// Rows, a linear term or a damping diagonal whose shape does not fit the structure.
    Shape { factor: Option<usize> },
    /// A block its rows do not determine.
    Undetermined { block: usize },
    /// An answer that needs every block eliminated, asked of an elimination that kept some.
    Incomplete,
}
```

### The linearizer

```rust
/// Which rows a linearization is assembled from.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Assembly {
    /// [%full](statement:archive/linearization/math/linearization): each factor's residual.
    Residual,
    /// [%full](statement:archive/linearization/math/fisher): each factor's residual, and, where
    /// the scale is free, its information root's scale rows.
    Fisher,
}

/// A density query of a snapshot, ready to be linearized at any batch of points: its factors, its
/// structure, its programs and its seeds, all fixed by the model and the query.
pub struct Linearizer {
    snapshot: Snapshot,
    target: Target,
    assembly: Assembly,
    /// The query's density program: `f`, and the layout.
    density: Arc<Program>,
    /// The program of every bound node the assembly reads.
    rows: Arc<Program>,
    factors: Vec<Factor>,
    structure: Arc<Structure>,
    /// The compressed identity frames of [The passes](#the-passes), shared by every row.
    seeds: Frames,
    colors: Vec<Range<usize>>,
    /// The terms reached from no free quantity: constants of the query.
    constants: Vec<NodeId>,
}

impl Linearizer {
    pub fn new(snapshot: &Snapshot, query: &Query, target: &Target, assembly: Assembly)
        -> Result<Linearizer, Error>;
    /// The free continuous quantities, in the order of the blocks.
    pub fn layout(&self) -> &Layout;
    pub fn structure(&self) -> &Arc<Structure>;
    /// Each factor's term, in the order of the structure's factors.
    pub fn terms(&self) -> Vec<NodeId>;
    /// `Schedule::all` of the structure.
    pub fn schedule(&self) -> Result<Schedule, Error>;
    /// `Schedule::keeping` the blocks of `quantities` (`Error::NotFree` for one that is not).
    pub fn schedule_keeping(&self, quantities: &[NodeId]) -> Result<Schedule, Error>;
    /// The linearization at `point`, one problem per row.
    pub fn at(&self, point: &Values) -> Result<Linearized, Error>;
    /// The chart linearization at `point` over `quantities`, each of whose spaces must have a
    /// volume (`NoVolume`): the linear term less `∇λ_p̂(0)` and the constant less `λ_p̂(0)`
    /// (`Error::NotAnIntegral` of the Fisher assembly).
    pub fn at_in_chart(&self, point: &Values, quantities: &[NodeId])
        -> Result<Linearized, Error>;
}
```

`Linearizer::new` does, once:

1. **The terms and the free quantities.** `query.terms(snapshot)`, or `Error::NotADensity` for a
   query of nodes. The density program is `snapshot.compile(query, target)`, and
   `refuse_free_discrete` is the first refusal. Its layout gives the blocks.
2. **The factors.** A term's variables are `involves` less what is not in the layout, and a term
   with none is a constant of the query
   ([](eq:archive/linearization/math/variables)). For each factor, in the order `query.terms`
   gives:
   - its residual, `term.bound(&RESIDUAL)`, or `Error::NoResidual { term }`, the second refusal;
   - its scale, the one of the four scale roles it binds, if any; the scale is *free* when
     `snapshot.reached_from(scale)` meets the layout. With `Assembly::Residual`, a free scale is
     `Error::ScaleNotFixed { term, quantity }`, the third refusal, naming the first free quantity
     it is reached from. With `Assembly::Fisher`, it needs `INFORMATION_ROOT` and `TANGENT`
     bound, or is `Error::NoInformationRoot { term, quantity }`;
   - its log-volume, `LOG_VOLUME`, and whether it is folded ([Folding](#folding)).
3. **The structure.** One block per layout quantity, one factor per factor term over its variables'
   blocks.
4. **The programs and the seeds.** The rows program, `Query::of(targets).given(observations)`,
   with the targets every factor needs ([The passes](#the-passes)), compiled on the target; and
   the seeds.

Every refusal names a term or a quantity, and none depends on a number: what refuses one row of a
batch refuses all of them.

```rust
#[derive(Clone, Debug, PartialEq)]
#[non_exhaustive]
pub enum Error {
    Core(skoro_core::Error),
    Elimination(skoro_elimination::Error),
    /// A query of nodes, where a density query was needed.
    NotADensity,
    /// A factor that binds no residual.
    NoResidual { term: NodeId },
    /// A factor of the residual linearization whose bound scale is reached from a free quantity.
    ScaleNotFixed { term: NodeId, quantity: NodeId },
    /// A factor of the Fisher linearization whose scale is free and which binds no information
    /// root, or no tangent.
    NoInformationRoot { term: NodeId, quantity: NodeId },
    /// A free quantity its rows do not determine.
    Undetermined { quantity: NodeId },
    /// A quantity named where a free continuous one of the linearizer was needed.
    NotFree { quantity: NodeId },
    /// A retirement whose retired terms do not determine a retired quantity.
    RetiredUndetermined { quantity: NodeId },
    /// An integral, a draw or a chart linearization asked of the Fisher linearization, which
    /// approximates no integral.
    NotAnIntegral,
    /// A value given for the replacement's query that leaves a quantity of the retired terms
    /// neither free nor observed, or observes one it should keep.
    Blanket { quantity: NodeId },
}
```

`Error::Undetermined` is the elimination's `Undetermined { block }` with the block mapped back to
its quantity; the free discrete quantity and the integral without a volume are the core's
`FreeDiscrete` and `NoVolume`, wrapped.

### Folding

A factor's log-volume is folded, [%full](statement:archive/linearization/math/linearization), when
it binds `TANGENT`, `LOG_VOLUME` and `INFORMATION_ROOT`, its scale is held (fixed, or held at
$\hat \point$ by the Fisher assembly), and the space of the quantity it scores has
`volume_reads_point() == Ok(false)`. A term that lacks the root is not folded, which is safe, since
$\hat c_t$ reads $W_t$ off it. The tangent role promises $r_t = W_t s_t$ and
$\lambda_t = \lambda_\mu(s_t)$; a log-volume that does not read its point makes $\lambda_t$ a
function of $s_t$ alone; and a held scale makes $W_t$ constant. That is the statement's hypothesis,
decided by the term's roles and its scored quantity's type, never by a number, so every row of a
batch folds the same terms. On the library's types it folds the normal's log-volume on the
rotations, the poses, the extended poses and the sphere, and not on a positive type, whose
log-volume reads its point, as [Folding a log-volume into its rows](math.svx#linearizing-a-query)
says it should not. A log-volume that is not folded is carried by $\eta$, which is correct whatever
the type: folding is a gain in accuracy, never a condition of correctness.

The folded right-hand side is computed from its definition,

$$
	\hat c_t = W_t^{-\top} \nabla \lambda_\mu(s_t) ,
$$

on the host. $\hat s_t$ and $\hat \mu_t$ are values the rows pass gives (the targets bound under
`TANGENT` and `MEAN`), $W_t$ is the top-left block of $C_t$'s value, and $\nabla \lambda_\mu$ is
the gradient at its tangent input of `space.log_volume()`'s composite, compiled once per space on
`Target::host()` and run at $(\hat \mu_t, \hat s_t)$ for every folded term of that space and every
row at once, stacked on the batch axis. $W_t\transpose \hat c_t = \nabla \lambda_\mu$ is solved by
back substitution, $W_t$ being lower triangular. Two alternatives were set aside.

- **Solving $J_t\transpose c = \nabla \lambda_t$ for $c$** needs no tangent and no root, since
  every solution gives the same $m$. But it is solved at $\kappa(J_t)$, not $\kappa(W_t)$, and
  when the free quantities reach the term through fewer coordinates than it has, it needs a rank
  decision, which numbers would make.
- **Differentiating with respect to the scored quantity**, $\hat c_t = (\partial_x r_t)^{-\top}
  \partial_x \lambda_t$, is well conditioned, but needs the scored quantity free in some program,
  and a measurement is observed: it would compile a second rows program under altered
  observations, which the core cannot yet form.

### The passes

The rows program's targets, per factor $t$:

| target | read for |
| --- | --- |
| the residual $r_t$ | its value $\hat r_t$, and its Jacobian $J_t$ |
| the log-volume $\lambda_t$, when bound and not folded | its gradient, a part of $\eta$ |
| `TANGENT` and `MEAN`, when folded | $\hat s_t$ and $\hat \mu_t$ for $\hat c_t$ |
| `INFORMATION_ROOT`, when folded or of a free scale | $W_t$, and $C_{\theta, t}$ |
| `TANGENT` and the scale, when the scale is free (Fisher) | $\D s_t$ and $\D \theta_t$ |

All of them come from one tangent pass of the rows program, at the point. The pass takes frames at
the rows program's free quantities and at no others (`Program::columns` refuses a frame at any
other node with `FrameShape`), so the seeds are built from `rows.free()`. Every one of them is a
free quantity of the density program, since every target is reached from its term's variables, and
`Linearizer::new` checks it; a free quantity of the density program that no target is reached from
has no rows at all, and is refused at elimination as undetermined.

The identity frame of the layout, `Frames::identity`, would do, and is what
[The passes](math.svx#linearizing-a-query) describes, but its frames are dense in all $D$ columns:
every target's frame is `[rows, dimension, D]`, which for a map of a few thousand poses is the whole
Jacobian, mostly zeros, at every row of the batch. A cotangent pass per target, seeded at that
target alone, keeps the columns down but runs one pass per target. The design compresses the
identity frame instead, by colouring, as sparse Jacobians are compressed. The colouring is read off
the factor structure, the bipartite graph of
[the note](../../math/square-root-elimination/#order-fill-and-the-elimination-tree) whose nodes are
the free quantities and the factors, with an edge between a factor and each quantity it involves,
drawn for the running example in [the mathematics' Setting](math.svx#setting):

- two free quantities *conflict* when they are joined to a common factor in the factor structure;
- a greedy colouring, in the layout's order, gives each quantity the first colour that none of the
  quantities it conflicts with has;
- colour $c$ owns a range of columns as wide as the largest block of its colour, and the seed of a
  quantity $q$ of colour $c$ is the identity of its dimension placed in $c$'s first columns: a
  frame of one row, shared by every row of the batch.

That each factor's columns come out unmixed is read off the
[compilation graph](../core/math.svx#the-compilation-graph), whose nodes are the model's values
and whose edges run from each input of an operation to its output, and whose roots are the
quantities and constants. A node bound to a term's role is reached from no quantity outside the
term (`TermHandle::bind` refuses it with `BindingOutsideTerm`), so the roots a target of factor $t$
is reached from are among $V_t$, and its free roots among $K_t$. The quantities of $K_t$ conflict
pairwise, so they have different colours, and the columns of $J_t$ for $q \in K_t$ are exactly
$q$'s colour's columns of the target's frame. The number of columns is the colours' total width,
which the factor structure alone fixes: on a chain of poses with landmarks it is a few blocks,
whatever the length of the chain. The colouring is computed once, in `Linearizer::new`.

The pass gives frames on the target, at the query's precision; each target's values and frames
cross to the host once per pass, as `Frame::to_host` gives them, and are cut into each row's rows
there. As in the archive, that is one transfer per target, which a device pays for in fences; one
transfer for all of them is left until it shows.

The density program's forward pass gives $f(\hat \point)$ for each row, the sum over every term,
constants of the query included.

### Assembling a row

```rust
/// A batch's linearization: one problem per row, over the linearizer's structure.
pub struct Linearized {
    problems: Vec<Problem>,
    assembly: Assembly,
    layout: Layout,
    quantities: Vec<NodeId>,
    /// The quantities of a curved type, whose covariances carry an error of order `σ²`.
    curved: Vec<NodeId>,
}

impl Linearized {
    pub fn problems(&self) -> &[Problem];
    /// Eliminates every row along `schedule`, which must be of the linearizer's structure.
    pub fn eliminate(&self, schedule: &Schedule, damping: Option<&Damping>)
        -> Result<Solved, Error>;
}
```

For each row of the batch, and each factor $t$, with the Jacobians cut from the frames:

- **Residual assembly.** The rows are $A_t = J_t$, over its blocks in the structure's order, and the
  right-hand side $b_t = -(\hat r_t + \hat c_t)$, with $\hat c_t = 0$ where the log-volume is not
  folded.
- **Fisher assembly, fixed scale.** The same: $J^\mu_t = J_t$.
- **Fisher assembly, free scale.** The rows are $J^\mu_t = W_t \D s_t$ above
  $J^\theta_t = C_{\theta, t} \D \theta_t$, and the right-hand side $-(\hat r_t + \hat c_t)$ above
  $-u_t$, with $u_t$ computed from $\hat r_t$ as the statement defines it and laid out as $C_t$'s
  rows. A factor of this kind has $d + p$ rows for a residual of $d$ components.

The linear term is $\eta = \sum_t \nabla \lambda_t$ over the bound log-volumes that are not folded,
each the tangent frame of its target, added into its blocks; it is `None` when no factor has one,
and then exactly zero. It is never formed as $g - \sum_t J_t\transpose (r_t + \hat c_t)$
([Why $\eta$ is read off the log-volumes](math.svx#linearizing-a-query)), and no pass of the query's
gradient is run at all. The constant is $c_0 = f(\hat \point) - \tfrac12 \sum_t \lvert \hat r_t +
\hat c_t \rvert^2$, less $\tfrac12 \sum_t \lvert u_t \rvert^2$ in the Fisher assembly, so that
$m(0) = f(\hat \point)$. The roundoff is the target precision's: $2^{-53}$ or $2^{-24}$.

`at_in_chart` adds the chart's term for the named quantities: for each, `space.log_volume()`'s
composite, compiled once per space on the host, gives $\lambda_{\hat \point_q}(0)$ and its gradient
at $(\hat \point_q, 0)$; the gradients are subtracted from the linear term and the values from the
constant. Whether the linear term is then present is decided by the types, not by the gradients'
values: it is, if it was, or if one of the quantities is of a space whose chart volume reads its
point, or whose gradient at zero, a constant of the type when it does not, is not zero. That
constant is computed once per space and type, at any point, since it is the same at every point;
on the library's curved types it is exactly zero, their log-volumes being even, and on a positive
type the volume reads its point. So a model of normals with fixed scales on the Euclidean and
curved types has no linear term in the chart either, as
[When there is a linear term](math.svx#blanket-quantities-the-rows-do-not-determine) says.

### The answers of a batch

```rust
/// A batch's eliminations, one per row, mapped back to the linearizer's quantities.
pub struct Solved {
    eliminations: Vec<Elimination>,
    assembly: Assembly,
    layout: Layout,
    quantities: Vec<NodeId>,
    curved: Vec<NodeId>,
}

impl Solved {
    pub fn eliminations(&self) -> &[Elimination];
    /// Each row's minimizer as a `Points` of `[rows, D]` on `target`, for `Layout::retract`.
    pub fn minimizer(&self, target: &Target) -> Result<Points, Error>;
    /// Each row's marginal, of a `schedule_keeping` elimination, with its quantities. The
    /// covariance of a curved quantity errs by a relative amount of order `σ²`, the noise's
    /// variance in its chart, against the second-order approximation and the posterior alike.
    pub fn marginal(&self) -> Result<Vec<QuantityMarginal>, Error>;
    /// `log ∫ e^{-m}` per row: with a chart linearization at a mode, the Laplace evidence
    /// (`Error::NotAnIntegral` of a Fisher linearization).
    pub fn log_integral(&self) -> Result<Vec<f64>, Error>;
    /// A draw per row, `τ* + R⁻¹ z` with `z` standard normal from `key`, retracted onto the
    /// quantities (`Error::NotAnIntegral` of a Fisher linearization).
    pub fn draw(&self, at: &Values, key: &Key, target: &Target) -> Result<Values, Error>;
}

/// A marginal of the linearization, over named quantities.
pub struct QuantityMarginal {
    /// The kept quantities, in the order of the marginal's blocks.
    pub quantities: Vec<NodeId>,
    /// Those of them of a curved type, whose covariance is reported with an error of order `σ²`.
    pub curved: Vec<NodeId>,
    pub marginal: Marginal,
}
```

**What the Fisher linearization does not give.** It gives steps, damping and marginal covariances,
and [What it gives, and what it does not](math.svx#information-roots) says it gives no evidence
and no replacement: those approximate an integral of the density, whose curvature in a free scale
is not the expected information. `log_integral` and `draw` of a Fisher linearization, and
`at_in_chart` of a Fisher linearizer, are refused with `Error::NotAnIntegral`; a draw is a draw
from the Gaussian that approximates the integrand, and so belongs with the evidence. `Replacing`
builds its linearizer with `Assembly::Residual` always.

**The order of the error.** [The precision is the rows' alone](math.svx#what-the-covariance-leaves-out)
asks that wherever the covariance of a curved quantity is reported, the order of its error be
stated with it. The code states it twice: the documentation of every function that gives a
covariance or a replacement says that on a curved quantity it errs by a relative amount of order
$\sigma^2$, and a `QuantityMarginal` names the curved quantities among its own, those whose space
is `Space::Embedded` or a product with such a factor, so that a reader that prints a covariance can
print the caveat beside it.

A draw takes its variates as every draw in skoro does
([position 10](../charter/writeup.md#10-randomness-is-explicit)): one key per row by `Key::split`,
$D$ uniforms from each by `Key::uniforms`, and standard normals by `skoro_normal::quantile`.
`Elimination::solve` maps them to $R^{-1} z$, the minimizer is added, and `Layout::retract` takes
the sum onto the quantities.

The Laplace approximation of [the mathematics](math.svx#the-laplace-approximation) is these pieces
on a chart linearization at the mode: `at_in_chart` over every free quantity, then the minimizer,
the marginal, a draw and the evidence. The entry builds no `Laplace` type: which point is the mode,
and how it is found, are the method's.

Damping is a `Damping` passed to `eliminate`, and leaves the schedule as it is; the promised decrease
is `Problem::promised` of each row at the damped minimizer. The schedule of $\lambda$ and the choice
of $D$ are left to the method's entry.

### The replacement

```rust
/// A retirement's replacement, ready to be linearized at any batch of points: the query of the
/// retired terms alone, free in the quantities to integrate and to keep.
pub struct Replacing {
    linearizer: Linearizer,
    integrated: Vec<NodeId>,  // E_F
    kept: Vec<(NodeId, Space)>, // B_F, in the blanket's order
    schedule: Schedule,       // Schedule::eliminating the blocks of E_F
}

impl Replacing {
    /// `keep` the continuous blanket quantities the replacement stays a function of; `fixed` the
    /// values of the other blanket quantities the retired terms involve. The values of `E_O` are
    /// the retirement's own.
    pub fn new(retirement: &Retirement<'_>, keep: &[QDyn], fixed: &Observations, target: &Target)
        -> Result<Replacing, Error>;
    /// The replacement linearized at `at`, which holds `p̂_E` and the anchors `p̂_B`. On a curved
    /// quantity of the blanket its precision errs by a relative amount of order `σ²`, the noise's
    /// variance in its chart, and more over a sequence of retirements that measure nothing.
    pub fn at(&self, at: &Values) -> Result<LinearizedReplacement, Error>;
}

/// The numbers of [](eq:archive/linearization/math/information-form) per row, with the anchors.
pub struct LinearizedReplacement {
    blocks: Vec<(NodeId, Space, HostValue)>, // each kept quantity, its space, its anchors per row
    rows: usize,                             // min(m', D)
    a: Vec<Vec<f64>>,                        // per row, column by column
    b: Vec<Vec<f64>>,
    linear: Option<Vec<Vec<f64>>>,
    constant: Vec<f64>,
}

impl LinearizedReplacement {
    /// The improper factor as the core's `Replacement`, its roles bound: to be called inside
    /// `Retirement::replace`'s body.
    #[track_caller]
    pub fn build(&self) -> skoro_core::Replacement;
    pub fn rows(&self) -> usize;
    pub fn reduced(&self) -> Vec<Reduced>;
}
```

`Replacing::new` reads the retirement's plan and snapshot, and

1. checks that every quantity of `keep` is a continuous blanket quantity (`Error::Blanket`);
2. builds the query of the retired terms,
   `Query::log_density_of(plan.terms).given(plan.observed.clone().union(fixed))`, and a
   `Linearizer` of it with `Assembly::Residual`, which refuses as any linearization does;
3. checks that the program's free continuous quantities are exactly $E_F \cup B_F$
   (`Error::Blanket`, naming the first that is not): a quantity of $E_F$ or of $B_F$ that `fixed`
   observes, or a quantity of the retired terms given no value, would each make another integral;
4. schedules the elimination of $E_F$'s blocks, `Schedule::eliminating`.

`at` linearizes in the chart of $E_F$ alone, `at_in_chart(at, E_F)`, since $E_F$ is integrated
against its reference measure and $B_F$ is not; eliminates $E_F$ along the schedule, where an
undetermined block is `Error::RetiredUndetermined`, naming the retired quantity; and reads
`Elimination::reduced` per row. The kept blocks are the layout's in its order, which is the
blanket's ancestral order; the anchors are the values of `at` at them. The shape, $\min(m', D)$
rows and whether the linear term is present, is the structure's, the same for every row,
singular rows included: the last QR decides no rank.

`build` makes, in the open scope, the stacked tangent with `Stacked::new().block_in(x, space,
anchor)` for each kept quantity in the layout's order, the factor with
`stacked.improper(rows, Rows(a), Rows(b), Rows(c))`, `.linear(Rows(η))` where the linear term is
present, and `.replacement()`. The order of the blocks, and the order of `A`'s columns, both come
from the layout, so the correspondence `improper`'s documentation calls load-bearing is kept by
construction; `A` is turned from the elimination's row-by-row `Matrix` to column-by-column storage
in this one place. A batch of one row passes plain values, so that the factor pins no batch.

> [!DECISION] The retirement's own values, and the caller's for the blanket
> The values of $E_O$ are those the retirement was read with, `plan.observed`, and the caller
> gives only the values of the blanket quantities it fixes, joined to them by the core's new
> `Observations::union`, which keeps the retirement's value of any quantity both name. A single set
> from the caller, holding $E_O$'s values again, would have needed no core addition, but a value of
> $E_O$ that differed from the one the retirement removed would then have given a replacement at
> the wrong point, silently, since nothing could compare the two.

### What is kept from the archive, and what is not

The archive's `skoro-dense` and `skoro-linearize` (`git show archive:crates/skoro-linearize/src/lib.rs`,
at `019e888`) are the shape of this design: a model-free elimination crate and a bridge; COLAMD on
the block pattern, constrained by appending the kept blocks; a junction tree computed once per
structure and shared by a batch; unpivoted QR on each front, frontal columns first; the marginal's
lower-triangular whitening by reversed QR; and the refusal of a term whose normalizer moves.

What changed, and why:

- **Fisher scoring is rows with a right-hand side.** The archive's Fisher curvature had
  homogeneous rows, $b = 0$, and solved the normal equations against the query's gradient
  (`BayesTree::solve_normal`), which is the $\kappa(J)^2$ solve and the difference of rounded
  gradients that [Why $\eta$ is read off the log-volumes](math.svx#linearizing-a-query) avoids. The
  scale residual $u_t$ puts the scale's score in the rows' right-hand side.
- **The replacement is an improper Gaussian.** The archive anchored a `JointNormal` of full rank at
  the marginal; a replacement here is the `Improper` factor in one form whatever the rank, so that
  a batch's replacements share a shape.
- **Folding.** The archive carried no log-volume in its rows; this design folds the curved types'.
- **The rank threshold depends on the precision** of the rows, where the archive's was fixed for
  double precision and failed silently on single.
- **Jacobians are compressed by colouring.** The archive's `residual_blocks` differentiated each
  term's block on its own; one coloured tangent pass gives every factor's Jacobian with a number of
  columns fixed by the structure.

## Tests

Every test below is added by the build. Where the mathematics asks for both precisions, the test
runs on `Target::host()` and on the Flex device at single precision, with tolerances of the
precision's order; no kernel changes, so none needs a wgpu adapter.

### The core and the normal

In `crates/skoro-core/tests/`:

- `query_gives_its_terms_and_observations` (`queries.rs`): `terms` of `log_density()` is every term
  in order, of `log_density_of` the named ones, of `Query::of` `None`; `observations` is the set
  given, of the same identity.
- `reached_from_is_what_a_term_involves` (`terms.rs`): for every term of a model with mapped nodes
  and fixed nodes, `reached_from(log_density)` equals `involves` less the scored quantity where the
  log-density does not read it.
- `composites_read_their_inputs` (`composite.rs`): a composite reading both inputs, one reading the
  second alone, and one calling another composite that ignores its input, which `reads` reports as
  read.
- `volume_reads_point_by_space` (`retractions.rs`): false on `Real`, `Vector`, `Matrix`, true on
  `Positive`, on a product with a positive factor, false on the spatial rotation and the test
  sphere, true on a test type whose `Λ` reads its point; `NoVolume` on a type without one.
- `constant_in_is_constant` (`building.rs`): `constant_in(space, value)` and `constant(marker,
  value)` compile to the same value, shared or per row; a value that does not fit is `ValueShape`.

In `crates/skoro-normal/tests/`:

- `score_binds_the_tangent` (`roles.rs`): the node under `TANGENT` is `x ⊖ μ`, and the residual
  is `W` times it, at both precisions, on a vector, a positive and a rotation. `roles_are_bound`,
  which lists every role a term binds, gains `TANGENT`: bound by `score`, not by `factor_at`.
- `block_in_is_block` (`improper.rs`): an improper factor over blocks stacked by `block_in` equals
  the same factor over `block`, on a vector and a positive.

### The elimination

In `crates/skoro-elimination/tests/elimination.rs`, each against the note:

- `triangular_root_of_the_example`: the note's $3 \times 2$ example gives $R$, $d$ and $e$ up to the
  signs of rows, the minimizer $(1, 1)$ and the minimum $225$.
- `the_ill_conditioned_example_keeps_its_digits`: $\varepsilon = 10^{-8}$ gives $(1, 1)$ to within
  $10^{-14}$, where Cholesky of the normal equations, written in the test, fails.
- `eliminating_one_block_is_the_lemma`: on random blocks, the conditional, the new factor's matrix
  (the Schur complement of part 3 of [%full](statement:math/square-root-elimination/eliminate-one)),
  its gradient at zero, and $\gamma$, against dense computations.
- `two_scalars`: the prior and the odometry give the rows of the note's example, the conditional
  $\tangent_1 = \tangent_2 / 2$, and the marginal variance $2$.
- `a_linear_term_in_the_row_space_stays_there`: with $\eta = A\transpose u$, the new factor's
  linear term is $A'^\top u_2$, and folding it into $b$ gives the same minimizer.
- `orders_agree`: on random sparse problems, the minimizer, the log-determinant, the minimum and
  every marginal agree across `Schedule::all`, a reversed order and random orders, to $10^{-10}$, and
  with dense computations; the marginal on $K$ is the $K$ block of $(A\transpose A)^{-1}$.
- `the_star_eliminates_its_leaves_first` and `the_map_has_three_cliques`: the note's star and its
  map of three poses and two landmarks give the plans of its figures, and the merged junction tree
  gives the same answers as one block per clique.
- `the_integral_is_the_determinant`: `log_integral` against
  $\tfrac{D}{2} \log 2\pi - \tfrac12 \log \det A\transpose A - \min \phi$ computed densely.
- `solve_is_the_inverse_root`: `solve` applied to each coordinate vector gives the columns of
  $R^{-1}$, so that $R^{-1} R^{-\top} = (A\transpose A)^{-1}$ exactly, not by sampling.
- `marginal_whitening_is_lower_triangular`: a positive diagonal, $W\transpose W$ the Schur
  complement.
- `damping_appends_rows_and_keeps_the_schedule`: the damped minimizer solves
  $(A\transpose A + \lambda D) \tangent = A\transpose b - \eta$; the schedule is the same object before
  and after; `promised` equals $m(0) - m(\tangent)$ of the undamped problem.
- `an_undetermined_block_is_refused`: a gauge freedom and a block no row reaches are
  `Undetermined`, naming the block, at the double-precision roundoff, and an exact dependence in
  rows rounded to single precision is refused at the single-precision roundoff.
- `replacing_from_some_factors_is_the_split`: [](eq:math/square-root-elimination/replace):
  `Schedule::eliminating` of $E$ from $\phi_{\ni E}$, plus $\phi_{\not\ni E}$, has the minimum and the
  integral of the whole; two disjoint replacements on the same kept block add to the replacement of
  both.
- `reduced_rows_have_their_shape`: `Reduced` has $\min(m', D)$ rows, the same for rows whose last
  QR is singular, and its constant is [](eq:archive/linearization/math/improper) computed densely.

### The linearization

In `crates/skoro-linearize/tests/linearize.rs`:

- `the_linearization_agrees_with_the_query`, [](eq:archive/linearization/math/agreement): on
  models of normals on vectors, positives and rotations, at both precisions, $m(0)$ is the query's
  negative log-density, $\nabla m(0)$ its cotangent pass, and each $J_t$ agrees with central
  differences of $r_t$ along $\hat \point \oplus h \tangent$; only the columns of $K_t$ are
  nonzero, which the colouring makes so and the test checks against `Frames::identity`.
- `linear_gaussian_terms_are_exact`, [%full](statement:archive/linearization/math/exact):
  $m(\tangent) = f(\hat \point \oplus \tangent)$ at several $\tangent$, at a point that is not the
  mode, on vectors and on positives with log-normal terms.
- `the_step_is_gauss_newton`, [](eq:archive/linearization/math/step): on a small, well-conditioned
  model, the minimizer equals a dense solve of the normal equations.
- `the_step_keeps_its_digits`: a model whose rows are the note's ill-conditioned example gives the
  step $(1, 1)$ to within $10^{-14}$.
- `steps_converge_on_rotations`: iterated steps on normals of rotations converge to the mode of
  [](eq:archive/normal/math/gn-objective), the log-volumes included.
- `the_linear_term_is_zero_without_log_volumes`: on Euclidean normals with fixed scales, every
  problem has `linear: None`, and the step equals, bit for bit, the minimizer of a problem the test
  assembles from the same rows by hand, with no gradient pass anywhere.
- `folded_log_volumes_are_in_the_right_hand_sides`: on normals of rotations, observed and free,
  `linear` is `None`; $J_t\transpose \hat c_t$ equals the cotangent pass of $\lambda_t$; and steps
  iterated from the same start converge to the mode reached by steps whose problem carries the
  same gradients as a linear term instead.
- `positive_log_volumes_are_a_linear_term`: on log-normal terms, `linear` is present and equals the
  log-volumes' gradients, $(1, \dots, 1)$ per scored positive coordinate.
- `batch_rows_share_the_schedule`: a batch of five points is linearized and eliminated along one
  schedule, and each row's step equals the step of that row linearized alone, to rounding.
- `damping_promises_the_decrease`: the damped step solves
  $(J\transpose J + \lambda D) \tangent = -g$, and [](eq:archive/linearization/math/promised)
  equals $m(0) - m(\tangent)$.
- `the_laplace_approximation_is_centred_in_the_chart`: on a quantity scored by a log-normal of
  median $\mu$, the chart linearization at $\mu$ has a zero minimizer, the variance of the
  logarithm, and the evidence of a linear-Gaussian model of positives equals the closed form, and
  one-dimensional quadrature.
- `draws_have_the_marginal_covariance`: the sample covariance of many draws in the chart is within
  sampling error of the marginal's, and two draws from one key are equal.
- `marginals_name_their_curved_quantities`: a marginal over a vector, a positive and a rotation
  names the rotation alone as curved.

In `crates/skoro-linearize/tests/fisher.rs`:

- `fisher_rows_are_a_root_of_the_expected_information`,
  [](eq:archive/linearization/math/fisher-matrix): on normals with free means and an isotropic,
  a diagonal, a covariance-factor and a whitening scale, on vectors and positives, the assembled
  $A\transpose A$ equals the normal's closed-form information, and a Monte Carlo average of the
  Hessian over draws of the scored quantities.
- `fisher_gradient_is_the_query_gradient`: $\nabla m_F(0) = g$, a free scale's included, against
  the query's cotangent pass.
- `fisher_with_fixed_scales_is_the_residual_linearization`: problem for problem, row for row.
- `fisher_scoring_converges_to_the_estimate`, [](eq:archive/linearization/math/scoring-step): on
  draws from normals with free means and an isotropic or a diagonal scale, iterated scoring steps
  reach the maximum-likelihood estimate in closed form; the isotropic scale's step is
  $(\lvert r \rvert^2 - \dimension) / 2 \dimension$.
- `fisher_marginals_are_the_inverse_information`: the marginal covariances equal the blocks of the
  inverse expected information, a free scale's among them.
- `fisher_shares_the_schedule`: both assemblies of one query have one structure, and one schedule
  serves both.
- `fisher_gives_no_integral`: `log_integral`, `draw` and `at_in_chart` of the Fisher assembly are
  `NotAnIntegral`.

In `crates/skoro-linearize/tests/replacement.rs`:

- `retiring_the_scalar_chain_is_the_kalman_filter`: retiring $\randomPoint_0$ and $\bfz_0$ from
  the core's scalar chain gives [](eq:archive/core/math/kalman), and retiring $\randomPoint_0$
  alone [](eq:archive/core/math/joint-blanket) with $c = 0$; each equals in value the `Stacked`
  replacement with the mean, whitening and constant of [](eq:archive/linearization/math/stacked),
  at several points; and the model after the retirement, queried for $\randomPoint_1$, gives the
  log-density the model retired with the exact replacement of `crates/skoro/tests/retirement.rs`
  gives, at several points.
- `retiring_a_chain_one_state_at_a_time_is_the_kalman_filter`: on a linear-Gaussian chain of
  several states, the replacements' means, covariances and accumulated constants are the Kalman
  filter's and its evidence.
- `a_nonlinear_chain_is_the_extended_kalman_filter`: linearized at the prediction, the predicted
  mean agrees with an extended Kalman filter's to second order in the update's step, and its
  covariance to first order; linearized at the retired terms' mode, found by Gauss–Newton on them,
  it equals an iterated extended Kalman filter's step.
- `improper_replacements_accumulate`: on the planar example of
  [Blanket quantities the rows do not determine](math.svx#blanket-quantities-the-rows-do-not-determine),
  each retirement leaves one row on $\ell$, the two summed equal the replacement of retiring both
  poses and measurements at once, constants included, and a query of $\ell$ then gives its exact
  posterior; a query after the first retirement alone refuses $\ell$ as undetermined.
- `a_bearing_replacement_is_the_linearized_integral`: on a nonlinear bearing to a landmark from one
  pose, the replacement equals a dense computation of the same linearized integral.
- `the_positive_replacement_is_exact`: retiring $\theta$ from the positive example gives one row
  along $\tangent - \varphi$ and the linear term $(0, 1)$, and the exact replacement's value.
- `rotations_and_vectors_leave_no_linear_term`: replacements on Euclidean and rotation quantities
  have no linear term, by structure.
- `a_batch_of_replacements_has_one_shape`: a batch whose rows include one with singular rows gives
  replacements of the same number of rows and the same presence of a linear term.

In `crates/skoro-linearize/tests/refusals.rs`, each naming what it names:

- `a_free_discrete_quantity_is_refused` (`FreeDiscrete`);
- `a_factor_without_a_residual_is_refused` (`NoResidual`), and a factor with a residual of no
  components, an improper replacement of zero rows, is accepted;
- `a_free_scale_is_refused_by_the_residual_linearization` (`ScaleNotFixed`, naming the term and the
  scale's quantity);
- `a_free_scale_without_a_root_is_refused_by_fisher` (`NoInformationRoot`), on a normal's
  `factor_at` with a free scale;
- `an_undetermined_quantity_is_refused` (`Undetermined`), a landmark seen along one bearing and a
  gauge freedom;
- `an_undetermined_retired_quantity_is_refused` (`RetiredUndetermined`);
- `an_integral_without_a_volume_is_refused` (`NoVolume`), by `at_in_chart` and by `Replacing`;
- `a_mismatched_blanket_is_refused` (`Blanket`): a kept quantity outside the blanket, and a
  retired term's quantity given no value.

## Open questions

None remains. The decisions a reader may want to revisit are marked as such above: the two crates,
faer, the tangent role, folding from the space's volume, the rank threshold, colouring, and the
replacement's `given`.
