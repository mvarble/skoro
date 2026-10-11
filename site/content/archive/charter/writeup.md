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

> [!TODO] Operator sugar under a single writer
> In the archive, `a + b` added a node because every quantity handle carried the model. With a
> single writer there are two shapes, and the choice is open:
>
> - a handle holds the writer, so `a + b` keeps working, but only on the writing thread;
> - building goes through the writer explicitly (`w.add(&a, &b)`, or a closure handed a builder),
>   and a handle is a plain typed identifier that can be sent anywhere.
>
> The first keeps the ergonomics; the second makes the single-writer rule visible in the types.
> Both need a sketch of a small sequential model before one is chosen.

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

### 11. burn is the tensor layer, behind a layer of skoro's own

burn was chosen for performant tensor operations that run on CPU and GPU from one Rust codebase,
with cubecl making custom kernels simple to write. The backend trait is sealed to burn's CPU backend
and its cubecl GPU backends: `Autodiff` never runs a model (it appears only inside the bridging
operation), and skoro supports only the backends it tests.

Operations reach burn through a small module of skoro's own, holding the tensor shapes skoro uses
(batched points, tangent frames, integer values) and the tensor operations its primitives need.
burn is not yet at version 1 and breaks between releases; an upgrade should touch that module and
the primitives' kernels, never a distribution or an estimator. Composition keeps that module small,
since only the primitives touch tensors.

> [!TODO] Precision
> The archive ran each backend under a float precision policy, single precision by default and
> double precision where tests demanded it. Is precision a property of the backend, chosen once per
> model, or of individual quantities? And is there a user's story, not only a test's, for choosing
> double precision?

> [!TODO] Host linear algebra
> The archive solved its least-squares and marginalization problems on the host, in double
> precision, through faer, because burn has no QR factorization and forming normal equations
> squares the conditioning. Is that a position for the charter (dense factorizations belong on the
> host), or a choice for the estimators' own entries to make and argue?

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
- **Snapshots and branches are a new design**, untested in the archive, and their top-level API is
  not yet sketched.
- **Registration has a cost.** Every type-erased thing must be named and registered, built-in or
  not.
- **Products are conditional**, and the core must prove them before they are relied upon.
- **Explicit keys** add an argument to every sampling call.

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

**Target and proposal flags on terms** (the archive). Replaced by samplers that carry their own
densities.

**Many small crates** (the archive). Replaced by two, since crate boundaries kept useful internals
apart without making the design any clearer.
