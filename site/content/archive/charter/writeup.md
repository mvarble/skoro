---
summary: A model is written once, generatively, and any inference algorithm reads what it needs from the model's structure, refusing by structure when the model does not fit.
---

## Context

### The problem skoro exists for

Estimation software for robots and autonomous systems is, for the most part, factor-graph
software. In GTSAM and its relatives a model is a set of *factors*: residual functions whose
weighted squares are summed and minimized. That formulation has two costs.

The first is that it is not generative. A factor is one particular reading of a conditional
density (its negative logarithm, seen as a least-squares residual), and the conditional itself is
never written down. Consider a range measurement $z$ between a vehicle at $x$ and a target at $y$:

$$
	z \sim \mu(\mathrm{d}z \mid x, y)
$$

That one statement covers three situations: estimating $x$ and $y$ jointly from $z$, estimating
$x$ from $z$ when $y$ is known, and simulating $z$ from given $x$ and $y$. A factor-graph library
needs a different factor class for the first two, different key management for each, and has no
account of the third at all, since a residual cannot be sampled from. Even a Kalman filter does
not fit naturally: the filter is a statement about a hierarchy of conditional distributions, and
the factor graph holds only the residuals of the posterior that hierarchy implies.

The second cost is that a factor graph serves one algorithm. It is built for maximum a posteriori
estimation of a nonlinear least-squares problem, and an algorithm that needs anything else (a
sampler, a filter, a variational approximation) has to be told the model's structure again, in its
own terms.

Probabilistic programming languages make the opposite choice. Gen.jl in particular separates
*modeling* from *inference*: a model is a generative program, and inference algorithms are written
against an interface to it. That separation is the one skoro wants. What those languages are not
built for is the model that lives over time. In sequential estimation the model grows as
measurements arrive, its pieces are not alike (different sensors, different measurement models at
different times), old states are retired as they stop mattering, and estimation runs concurrently
with all of it. A language that builds a program once and then runs inference on it does not
manage that well, and is rarely embedded in a real-time Rust system.

### What came before

The repository started over on 2026-10-09. The library as it stood is kept on the
[`archive`](https://github.com/mvarble/skoro/tree/archive) branch: fourteen crates, a model client
over a shared graph, a hand-written engine for first- and second-order derivatives on manifolds,
the Normal distribution, SO(3) and SE(3), and a block-sparse Gauss–Newton machinery with
marginalization for sliding windows. Most of it worked, and much of it is coming back. What it
taught, and what this document responds to, is a short list:

- **Every operation hand-wrote four derivative maps**, which made each new distribution expensive
  and was where wrong derivatives hid.
- **Conditioning was a mutation.** Observing a quantity changed the model, for every reader,
  which made "simulate, then condition, then infer" and posterior predictive checks awkward.
- **One lock guarded the model**, and every handle carried access to it.
- **Type erasure lost information.** A manifold assembled at run time could retract a point but
  could not differentiate the retraction, so product manifolds were out of reach.
- **Implementations were duplicated** across compile-time and run-time dimensioned types.
- **Randomness was global**: no seed, and no way to sample one quantity on its own.
- **Fourteen crates** kept useful internals out of reach of their siblings.

## Purpose

> [!DECISION] What skoro is
> Skoro is a library for writing probabilistic models once, generatively, and running different
> inference algorithms on the same model. The model is built forward, the way a user thinks of it:
> each uncertain quantity, and the conditional distribution it has given the quantities it depends
> on. The library records the structure as it is built, much as automatic differentiation records
> a computation, together with whatever later computations need. An inference algorithm reads that
> structure. If the model meets the algorithm's assumptions, the algorithm runs; if not, it says so
> with an error. The user never describes the model a second time for the algorithm's benefit.

Three kinds of user follow from this, in order of priority:

1. **People developing inference algorithms.** The reading surface (what an algorithm can discover
   about a model, and how cheaply) matters as much as the building surface. Exploring new
   algorithms is skoro's first purpose.
2. **People building real-time estimators** for robots and autonomous systems: heterogeneous
   sequential models that grow, retire old states, and are estimated while measurements stream in,
   on CPU or GPU, from Rust, often under several hypotheses at once about what the measurements
   came from.
3. **People doing Bayesian inference** with quantities that live on manifolds, such as rotations
   and poses, or that need second-order derivatives.

Skoro is not a general probabilistic programming language. Each of its models has a fixed
structure, and structure that is itself random is handled one level up, over a family of models
(see below). It does not compete with Stan or NumPyro on breadth of distributions or with GTSAM on
the maturity of its least-squares solvers. What it offers that they do not is the combination: one
generative model, a structure that algorithms can read, manifold-aware second-order derivatives,
and a model that lives across threads and over time.

## Positions

### 1. A model is the joint; conditioning belongs to the query

A model holds the generative structure and nothing else: the joint distribution of its quantities,
never a posterior. Which quantities are observed, and their values, is part of each *query*: an
inference call, a density evaluation, a generation. `observe` builds such an observation set; it
does not change the model.

The same model can then serve several questions at once, from several threads: one estimator
treats the target's position as known, another estimates it, a third simulates measurements. Two
uses are worth naming, because they motivated the position:

- **Checking an inference algorithm against simulation.** Generate $(x, y)$ from the joint
  $\mu(\mathrm{d}x, \mathrm{d}y)$, condition on the generated $y$, infer
  $\hat\kappa(\cdot \mid y)$, and compare it with the generated $x$ (the error of a point estimate,
  or the concentration of the approximate posterior around $x$).
- **Checking an estimator's consistency.** Generate predicted measurements
  $z \sim \kappa(\mathrm{d}z \mid \hat{x})$ from an estimate and compare them with the
  measurements actually received. With conditioning in the model, this needed a way to pretend the
  model was joint again; with conditioning in the query, it is one more query.

### 2. Models are built forward, change by appending and retiring, and random structure lives one level up

A model grows by adding quantities, each together with its conditional distribution. The one
shrinking change is *retirement*: in sequential estimation, quantities that were once estimated
eventually stop mattering, so they are marginalized onto what remains and then removed. Appending
and retiring are the only mutations a model sees.

Within one model, structure is fixed: it is whatever the builder created, and a random draw does
not add or remove nodes. Random structure is nevertheless central to skoro's purpose (which
measurement came from which target, whether a target was born or has died, which motion model is
in force), and it is handled in two ways.

**Bounded structure, inside one model.** When the structure has a known bound, it is expressed
with discrete quantities and the selection primitive (position 7). Data association is a
categorical choice per measurement among target slots and clutter; a random number of objects up
to a maximum is a set of slots, each with an existence variable gating its terms; a switching model
is a discrete mode selecting among parameter sets; a nonparametric prior is truncated. The graph
never changes, and samplers move the discrete quantities. This costs computation on empty slots
and needs a bound, and asks nothing new of the core.

**Unbounded structure, over a family of models.** In general, the random thing is *which model*.
A structural hypothesis $S$ (this measurement belongs to that target; a target was born here) lives
outside any graph, with a prior $p(S)$, and each hypothesis is a model built from it by ordinary
appending and retiring. Inference over structure (multiple-hypothesis tracking, sequential Monte
Carlo over hypotheses, reversible-jump Markov chain Monte Carlo with birth and death moves) keeps
one model per live hypothesis, runs ordinary inference inside each, and weighs the hypotheses by
$p(S)$ and an estimate of each model's evidence. A birth is an append on one hypothesis's model,
a death a retirement, and pruning a hypothesis is dropping its model.

Multiple-hypothesis estimation is a first-class use of skoro, and the core is shaped for it in two
ways. Hypotheses that differ by one association share almost everything, so models *branch*
cheaply from one another (position 3). And the same quantity must be recognizable across
hypotheses, so that they can be compared, merged and pruned, which is what *addresses* are for
(position 4). The algorithms that infer structure are later entries; the core only has to make
them possible.

This is close to what Gen.jl does, without what made it impractical here. A Gen trace records which
random choices a program made, which fixes its structure, together with their values. In skoro the
structure is an explicit hypothesis and the values live in a typed graph, so nothing has to reflect
over typed programs.

### 3. One writer per branch, many snapshots

A model has one writer, which appends and retires, and publishes immutable *snapshots* that share
structure with earlier ones. Inference runs on a snapshot, without a lock: a long sampling run
never blocks the thread adding measurements, and the thread adding measurements never changes a
model out from under an estimator. Memory belonging to retired quantities is reclaimed when the
last snapshot holding it is dropped.

A snapshot can also be *forked*: turned into a new writer whose appends and retirements form a
branch of their own, copying nothing that the branch does not change. Branches are how a model
under several structural hypotheses is held (position 2): each hypothesis is a branch, a new
measurement associated three ways is three cheap forks, and a pruned hypothesis is a branch nobody
holds any longer. There is one writer per branch, never two writers on one.

This replaces the archive's design, in which a shared graph sat behind one lock, every handle
carried a client that could change it, and drop guards fed a mark-and-sweep collector. It also
removes the need for a separate read-only access level: a snapshot is read-only by construction.

**Building reads like ordinary code.** A model is written as functions of quantities, with no
builder in their signatures. An operator and a function a user writes are the same kind of thing:

```rust
// The range from a vehicle at `x` to a target at `y`: an ordinary function of quantities.
fn range<B: Backend>(x: Q<B, Euclidean<3>>, y: Q<B, Euclidean<3>>) -> Q<B, Real> {
    (y - x).norm()
}
```

(The names in these sketches are illustrative; the core's implementation settles them.) `y - x`
and `range(x, y)` both add nodes to the model, and neither names the writer. They can do
so because building happens inside a *building scope* that a writer opens, and every operation on
a quantity adds its node to the writer whose scope is open on the current thread:

```rust
let x = writer.build(|| {
    let x = vehicle_prior.sample_as("vehicle/0");
    let z = Normal::new(range(x, target), sigma).sample_as("range/0");
    x
})?;
```

The writer is the only thing that changes a model, and a scope is how it lends itself to the code
that builds; outside a scope, an operation on a quantity is refused. A `Q` is then a plain typed
identifier, cheap to copy: it holds no reference to a writer, so it can be sent to other threads and used in
queries against any snapshot that contains its quantity. The same `Q` is valid on every branch
that shares the quantity, which multiple-hypothesis building depends on: each hypothesis's writer
opens its own scope, and the same function of the same quantities builds into each.

```rust
for (hypothesis, writer) in branches.iter_mut() {
    writer.build(|| associate(measurement, hypothesis.target(), hypothesis.clutter()))?;
}
```

Errors cannot be returned from `y - x`, and an operator that panicked on a shape mismatch would
break the rule that errors are values. So an operation that fails inside a scope still returns a
`Q`, marked failed: anything built on it is failed too, and the scope reports the first failure,
with where it happened, as the error of `build`. Nothing a failed scope built is published.

**How a scope works.** The exclusive access a writer needs is taken once, when the scope opens,
not by each operation: `build` borrows the writer mutably for the length of the closure, so the
compiler guarantees nothing else touches it meanwhile. Opening the scope stages a copy-on-write
draft of the latest snapshot and registers that draft with the current thread. An operation finds
the draft through the thread and borrows it only for as long as it takes to insert its own node,
never while user code or a distribution runs, so a distribution's `sample_as` can build its
log-density with ordinary operations inside the same scope. The draft belongs to one thread and is
invisible to readers, who hold immutable snapshots; when the scope ends without a failure, the
draft becomes the latest snapshot in a single atomic step, so a reader sees the model as it was
before the scope or as it is after it, never half of it. If the closure panics, the draft is
unregistered and discarded. Because a thread-local cannot be generic over the backend, the draft is
held type-erased and recovered by each operation; an operation that finds no draft, or a draft for
another backend, returns a failed `Q`. Scopes nest, and an operation builds into the innermost.

### 4. Static at the edges, dynamic inside, and everything nameable

The building API is typed. A quantity is a `Q<B, T>`, where `T` names its type (a manifold or a
discrete set), and distributions and operations are generic, so the compiler checks as much of a
model as it can while the model is written. Inside, the graph is type-erased: it holds quantities
of different types, added at run time from streams of data, and that needs dynamic dispatch. The
boundary between the two is deliberate, not accidental.

Everything the graph stores type-erased is *nameable*: each quantity type, operation and
distribution has a registered identifier and plain parameters. That is what lets a model be
serialized (an autonomous system can write its model out when it fails, to be inspected later),
and what will let Python bindings drive a fixed registry of distributions and operations while new
ones are written in Rust. Neither the format nor the bindings are part of the core; the principle
is, because retrofitting it is far costlier than respecting it from the start.

Quantities are nameable too. Each has an *address* chosen by its builder, such as "the position of
target 3 at tick 40", which stays the same across snapshots, branches and serialization, beside
the internal identifier the graph uses. Addresses are what let two branches agree that they hold
the same quantity, what random keys are derived from (position 10), and what a serialized model is
read back by.

**Addresses are hierarchical, and sub-models are functions.** An address is a path of segments,
each a name or an integer (tick `t` and measurement `j` are numbers, and stay numbers), displayed
as `40/target/3`. A *namespace* is a frame on the same stack as a building scope: inside
`scope::namespace(t, || ...)`, a quantity added as `x` is addressed `t/x`. A sub-model is then an
ordinary function of quantities that names its own quantities locally and knows nothing about where
it is placed; the code that calls it places it:

```rust
fn step<B: Backend>(prev: Q<B, Pose>, u: &Odometry, zs: &[Range]) -> Q<B, Pose> {
    let x = Normal::new(prev.compose(u.delta()), q).sample_as("x");
    for (j, z) in zs.iter().enumerate() {
        scope::namespace(j, || Normal::new(range(x, landmark(z.id)), sigma).sample_as("z"));
    }
    x
}

let mut x = writer.build(|| prior.sample_as("x"))?;
for (t, data) in stream.enumerate() {
    x = writer.build(|| scope::namespace(t, || step(x, &data.u, &data.zs)))?;
}
// "x", then "0/x", "0/0/z", "0/1/z", "1/x", "1/0/z", ...
```

Sub-models connect through their arguments and results, as Gen.jl's generative functions do, never
by looking each other's quantities up by name; looking a quantity up by address, relative to the
open namespaces, remains for code that resumes building from a snapshot. Namespaces nest freely.
They are also the natural unit of retirement: when tick 0 leaves a sliding window,
`writer.retire("0")` marginalizes everything under `0/` onto its Markov blanket and removes it,
which makes declarative the bookkeeping the archive did by hand on every tick. A namespace is part
of an address within one model, and is unrelated to branching: `1/x` names the same quantity in
every branch that contains it.

Three things are deliberately absent: code generated and compiled ahead of time (SymForce's
approach does not fit models built at run time), macro languages for models (hard to reason about
and unfriendly to editors), and panics across the public API (errors are values, which bindings
will need).

### 5. Derivatives come from composition, and their order is reported

Skoro differentiates for itself. burn's `Autodiff` backend was the obvious candidate and was ruled
out for two reasons: it has no second order (Hessian–vector products, which some samplers need,
require differentiating the backward pass), and it pulls back one cotangent vector per pass, while
a Jacobian-based optimizer wants a whole *frame* of them in one. Tangent-space derivatives on
non-Euclidean manifolds looked possible inside burn, but nothing guaranteed them deep in its
implementation.

A small set of *primitive* operations carries hand-written derivative maps (forward, JVP, VJP and
HVP, over frames), written once and tested hard. Everything else, distributions' log-densities and
manifolds' retractions included, is a *composite*: a subgraph of primitives, differentiated by the
chain rule. A composite is stored as a named, reusable subgraph and expanded when the model is
compiled, so a serialized model reflects the calls its user made. A hand-written fused version of a
composite is an optimization, tested against the composite, and some operations will need one for
numerical care (the logarithm of SO(3) near angle $\pi$, say).

Each operation states the highest order of derivative it supports, and a compiled program's order
is the least over its steps. A pass that needs more is refused before it runs, naming the operation
responsible. The case that motivates this is the bridge to neural networks: a model may contain an
operation that runs burn's `Autodiff` internally, translating between skoro's tensors and burn's at
its boundary. Such an operation supports first order only, and a second-order algorithm on that
model must fail with that explanation rather than with zeros.

### 6. Algorithms refuse by structure

The rule that runs through every position here: *what a model can do is read off its structure,
never declared by its user.* There are no capability flags and no capability traits. Instead,
each fact an algorithm depends on is recorded where it naturally lives, and the algorithm checks it
before it starts:

- the order of derivative each operation supports (above);
- whether a quantity is discrete, and whether it is observed (below);
- whether each density term carries the binding an algorithm reads (below);
- whether each distribution has a sampler, and whether that sampler is exact (below).

A refusal names what is missing and where. An algorithm that silently skips what it cannot read
(the archive's least-squares readers dropped any term without a residual) must be paired with a
check that nothing was skipped, since a well-conditioned solve of the wrong system is the worst
outcome an estimator can have.

### 7. Quantities: runtime dimensions, typed wrappers, two kinds, and products

**One implementation per idea.** Quantity types are dimensioned at run time (`Euclidean(n)`, a
matrix of `m` rows and `n` columns), and operations are written once against them. Fixed-size
forms such as `Euclidean<3>` are thin typed wrappers that convert to the runtime ones, so the
compiler catches shape errors where sizes are known and nothing is implemented twice. Shape
mismatches between runtime quantities are errors when the node is added.

**Two kinds of quantity.** A continuous quantity lives on a manifold: it has a tangent space, a
retraction, and an inverse retraction, and estimation moves it through its tangent space. A
discrete quantity takes values in a set and has none of these. Values of both kinds are stored as
tensors (floating-point and integer), and an operation's inputs may be of either kind. The rules
that follow:

- derivatives stop at discrete quantities: an operation is differentiated in its continuous inputs
  only, and a discrete quantity is never a coordinate of a gradient, Jacobian or Hessian;
- a gradient-based algorithm on a model with an unobserved discrete quantity refuses, saying so;
- *selection*, choosing among continuous values by a discrete index, is a primitive, differentiable
  in the values and not in the index, because mixtures, switching models and data association are
  built from it;
- `Categorical(n)` and an unbounded `Integer` are the discrete types, and `Boolean` is a typed name
  for `Categorical(2)`;
- a quantity with a continuous part and a discrete part, such as a pose with a mode, is two
  quantities.

Distributions over discrete quantities, and samplers that move them, arrive later through the
workflow. The kinds themselves are in the core because adding them later would touch everything.

**Products, provided the manifold interface builds subgraphs.** A product of quantity types is a
quantity type: its tangent space is the direct sum of its factors', and it retracts factor by
factor. Products make a joint distribution over several quantities ordinary (it scores one quantity
of a product type, and the quantities a user works with are projections of it), and they give a
marginal over a Markov blanket of run-time size a type to live on.

The archive could not build products, because a manifold *returned* its retraction as a typed
operation, which type erasure made opaque. Here a manifold *builds* its retraction into the graph:

```rust
trait DynManifold<B> {
    fn dimension(&self) -> usize;     // the tangent space's
    fn storage_size(&self) -> usize;  // the stored point's
    // Adds the nodes computing `point ⊕ tangent`, and returns the result.
    fn build_retraction(&self, g: &mut dyn GraphBuilder<B>, point: Node, tangent: Node) -> Node;
    fn build_inverse_retraction(&self, g: &mut dyn GraphBuilder<B>, a: Node, b: Node) -> Node;
}
```

A product's retraction is then a subgraph like any other: slice the point and the tangent into
blocks, let each factor build its retraction on its block, and concatenate. Its derivatives, of
both orders, follow from composition, and nothing about the factors' types needs to be recovered.
The position is accepted on that condition: the core's implementation must show that the
retraction of a runtime product of two Euclidean factors and a real line, and its first and second
derivatives, agree with the factor-wise computation. If it cannot, this position is reopened.

### 8. Terms have a fixed shape; roles are optional and arrive with their readers

A density term is the log-density node, the quantity it scores (a joint factor over existing
quantities scores none), and the quantities it is conditioned on. The model records all three when
the term is added, and readers use them for structural questions: which terms touch a quantity,
what its Markov blanket is, what retiring it disturbs.

Anything further a term offers is a *binding*: a labelled reference from the term to a node that
plays a *role* for it. A Gaussian term binds its whitened residual, its mean and its covariance
factor; a term with a closed-form expected information binds a root of it. A least-squares reader
finds the residuals, and a Fisher-scoring reader the roots, without the user saying anything: the
distribution declared them as it was added. Bindings count as references, so what a term binds
lives as long as the term.

The core defines no role at all. Each role arrives with the distribution or reader that gives it
meaning. A role is a typed key: it carries its name (its stable identifier) and the multiplicity it
expects (exactly one, at most one, or any number), and a lookup returns typed handles accordingly,
so the check that a term bound what it should happens once, in the lookup, rather than in every
reader.

### 9. A distribution has a density, and may have a sampler with a density of its own

A distribution provides the log-density of its term, and may provide a sampler. A sampler that
draws exactly from the distribution says so. One that draws from something else reports its own
log-density, which is how an improper prior (a flat density on a line) can still be generated
from, by a broad Gaussian say, with the mismatch accounted for. The importance weight of a
generated point is then a sum over terms:

$$
	\log w(x) = \sum_t \bigl(\log p_t(x) - \log q_t(x)\bigr), @tag(weight)
$$

where $q_t = p_t$, and the summand vanishes, for a term whose sampler is exact. A distribution with
no sampler at all is legal; generating from a model that contains one is refused, by structure.

This replaces the archive's two flags on each term ("in the target", "in the proposal"), and the
rule that a quantity generated but not scored had to be scored by a separate factor.

### 10. Randomness is explicit

Every sampling call takes a random key, which splits deterministically into independent keys, as in
JAX. One seed reproduces a whole run; each chain, particle set or thread has its own stream; no
hidden state couples them. The key a quantity's sampler receives is derived from the run's key and
the quantity's address, never from the order in which nodes were added, so that sampling one
quantity disturbs no other, retiring old states does not change the draws of new ones, and two
branches holding the same quantity draw it alike.

burn exposes no control over its random-number generation, so a sampler draws its uniform variates
from the key on the host and moves them to the device when it must. That costs a transfer, never
reproducibility.

### 11. burn behind a layer of skoro's own; precision chosen per query; solves on the host

burn was chosen for performant tensor operations that run on CPU and GPU from one Rust codebase,
with cubecl making custom kernels simple to write. The backend trait is sealed to burn's CPU backend
and its cubecl GPU backends: `Autodiff` never runs a model (it appears only inside the bridging
operation), and skoro supports only the backends it tests.

Operations reach burn through a small module of skoro's own, holding the tensor shapes skoro uses
(batched points, tangent frames, integer values) and the tensor operations its primitives need.
burn is not yet at version 1 and breaks between releases; an upgrade should touch that module and
the primitives' kernels, never a distribution or an estimator. Composition keeps that module small,
since only the primitives touch tensors.

**Precision belongs to the query.** burn's backend type does not fix float precision: the element
type it names is largely ignored, and precision is decided at run time, by a process-wide default
fixed at first use. Precision is also a choice that differs by estimator, not by model:
Levenberg–Marquardt wants double precision, because its conditioning demands it, while a sampler
on a GPU wants single precision, for speed. So:

- **A model commits to no precision.** Constants, observed values and initial points are plain
  data in the graph, stored on the host in double precision. They must be nameable and serializable
  anyway (position 4), and storing them as device tensors would fix a precision the moment the
  model is built.
- **The estimator chooses precision when it compiles or evaluates.** Compiling a program for a
  query takes a device and a precision, and that is when stored values become device tensors of
  the requested type. Compiled programs are cached by snapshot, targets, device and precision, so
  one snapshot can serve a double-precision solver on the CPU and a single-precision sampler on a
  GPU at once, as it serves different conditionings at once (position 1).
- **Skoro's tensor layer manages precision, and never defers to burn's default.** Every tensor
  skoro creates is created at an explicit precision, and every conversion is an explicit cast.
  Primitives' kernels are written for both precisions and tested at both. A device that cannot
  compute in double precision, as many GPUs through `wgpu` cannot, is reported when a query asks it
  to, before anything runs, like any other unmet requirement (position 6).
- **Random draws do not depend on precision.** A sampler draws its uniform variates from its key in
  double precision on the host and casts them, so the same key gives the same draw at either
  precision, up to rounding.

**The solves where conditioning lives run on the host, in double precision.** Dense
factorizations (the QR of a least-squares step, Cholesky, triangular solves, the elimination behind
marginalization) run on the host in double precision, through a dense linear-algebra library
(faer, as in the archive), whatever precision the model was evaluated at. burn has no QR
factorization, and the alternative, forming normal equations on the device, squares the condition
number. The arrangement also serves the mixed case: residuals and Jacobians evaluated in single
precision on a GPU, for speed, and the step solved in double precision on the host, where the
conditioning matters. Such a Jacobian arrives already rounded, which is the estimator's trade to
make knowingly; the solve itself is never the weak point. This is a principle for the estimators'
entries to build on; how each one uses it is theirs to argue.

### 12. Two crates and an umbrella

The core is two crates: `skoro-tensor`, the sealed backend and the layer over burn, and
`skoro-core`, everything that makes a model a model (identities and errors, quantity types and
manifolds, the primitives and the differentiation engine, the graph and its compiler, terms and
bindings, the writer and its snapshots, and the typed building API), separated by modules rather
than crates. The `skoro` crate re-exports both. Each later feature (a distribution, a manifold
such as SO(3), an estimator) is a crate of its own, built only on the core's public API, which
keeps that API honest: whatever a feature needs, the core must expose.

## Consequences

What becomes easy:

- running several algorithms, and several conditionings, on one model, concurrently;
- holding a model under many structural hypotheses at once, as cheap branches of one another;
- composing a model from sub-models written as functions, placed under namespaces, and retiring a
  whole namespace (a tick leaving a window) in one call;
- simulation-based checks of inference algorithms and consistency checks of estimators;
- adding a distribution or a manifold operation, which is mostly composition of existing pieces,
  with derivatives of both orders for free;
- writing an algorithm that discovers what it needs and refuses clearly when it is absent;
- reproducing any run that involved randomness, from one seed;
- writing a model out for later inspection, and, later, driving one from Python.

What becomes hard, or costs something:

- **Random structure is not automatic.** A model's structure is fixed; structure that is itself
  random needs either a bound (slots and discrete switches) or an algorithm over a family of
  branches, and those algorithms are later work. Programs whose shape changes as they run, in the
  manner of Gen.jl, are not supported.
- **Builders choose addresses.** Every quantity needs a meaningful, stable address, which is one
  more thing to name while building.
- **Compiled programs are longer.** Composition yields more, smaller steps than hand-written fused
  operations, until profiling earns a fusion.
- **Every reader takes an observation set**, and "is this quantity free?" becomes a question about
  the model and the query together.
- **Snapshots and branches are a new design**, untested in the archive.
- **Building scopes are implicit state.** Which writer an operation builds into is decided by the
  scope open on the current thread, not by anything in its signature; an operation outside a scope
  is refused when it runs, not when it compiles, and a failure inside one surfaces only when the
  scope ends.
- **Registration has a cost.** Every type-erased thing must be named and registered, built-in or
  not.
- **Products are conditional**, and the core must prove them before they are relied upon.
- **Explicit keys** add an argument to every sampling call.
- **Precision costs a conversion.** A stored value becomes a device tensor, at the precision a
  query asks for, once per compiled program; and every kernel must be correct, and tested, at two
  precisions.

## Alternatives

**Factors as the model** (GTSAM). Rejected for the reasons in the context: a factor is one reading
of a conditional density, serves one algorithm, and cannot be sampled from. Factors remain
available as what a least-squares reader *derives* from a generative model.

**Programs as the model** (Gen.jl). Gen's separation of modeling and inference is adopted; its
representation is not. A model that is a program re-run whenever a random choice changes its shape
needs efficient re-execution, which is hard, and reflection over typed programs, which Rust makes
impractical; and a program held as a closure cannot be serialized. Random structure is instead
handled over families of fixed-structure models held as branches (position 2), which keeps each
model cheap for algorithms to read.

**Random structure as out of scope.** Simpler, and tempting, since a graph built forward cannot
express it directly. Rejected because it would abandon multiple-hypothesis estimation, which
real-time estimation in cluttered environments depends on, when forkable snapshots and addresses
make it affordable.

**burn's `Autodiff`.** Rejected for the lack of second order and of frame pullbacks. Kept, through
the bridging operation, for neural networks inside a model.

**Hand-written derivatives for every operation** (the archive). Replaced by composition, which
removes most derivative code and the bugs that lived in it.

**Code generated ahead of time** (SymForce). Rejected: skoro's models are built at run time, from
streams of data, and must not need a compilation step.

**Conditioning by mutation, and one lock around a shared graph** (the archive). Replaced by
query-time conditioning and snapshots, for the reasons in positions 1 and 3.

**Compile-time dimensions only, or both kinds implemented separately** (the archive). Replaced by
runtime dimensions with typed wrappers, so nothing is implemented twice.

**Handles that carry their writer** (as the archive's carried its client). Operators work
anywhere a handle is, but a handle becomes bound to one thread and one branch: it cannot be sent to
an estimator, and building the same function into several hypotheses means rebinding every handle
to each branch. Building scopes keep the sugar and leave handles free.

**Lifetimes on handles instead of a thread-local** (`Q<'s, B, T>`, holding a reference to the
draft). Using a handle outside its scope would then fail to compile rather than fail when run, and
no thread-local would be needed. Declined, for now, because the lifetime appears in every function
a user writes, which is the burden building scopes exist to remove; it is the fallback if the
run-time checks prove error-prone.

**Importing a built model under a namespace.** Possible, since built models are nameable, but an
imported model arrives with its own priors fixed, while a sequential model needs each step's prior
to be the previous step's result. Sub-models as functions with arguments express that; imports do
not, and are left out until a use for them appears.

**An explicit builder in every signature** (`builder.add_mapped_quantity(...)`, or
`fn f(b: &mut Builder, x: Q) -> Q`). Rejected: it makes the code that describes a model read like
code that manipulates a graph, and an operator cannot take a builder at all.

**Precision fixed by the backend or the model** (the archive's precision policy). A backend-wide
policy cannot serve a double-precision solver and a single-precision sampler on one model, and a
model that stores device tensors fixes precision before any estimator has been chosen. Precision
was moved to the query.

**Factorizations on the device.** Rejected while burn has no QR and normal equations square the
condition number; dense solves stay on the host in double precision.

**Target and proposal flags on terms** (the archive). Replaced by samplers that carry their own
densities.

**Many small crates** (the archive). Replaced by two, since crate boundaries kept useful internals
apart without making the design any clearer.
