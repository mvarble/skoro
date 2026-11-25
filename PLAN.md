# Plan

What skoro means to do next, kept here so the plan outlives any one session.
Each item returns as an entry of its own, through the workflow (`.agents/reference/mesearch/workflow.md`), and ends with the command that would open it; nothing here starts one.
The slug and the kind in each command are proposals, settled when the entry is opened.
Items that cite measurements take them from `EVALUATION.md`, the evaluation of the core of 2026-11-10, which says how each was made.
The archive is the branch `archive`, tagged `archive-2026-10-09`; read it with `git show archive-2026-10-09:<path>`.

## Immediate future

The goal is an initial release that shows what skoro does: a model built once, then estimated, filtered and sampled without being told its structure again.
The items are in the order they should be taken.

- **Estimation reads only Gaussian-shaped terms.** `Linearizer` refuses a model with any term that binds no residual (`NoResidual`), and the roles it reads are the normal's.
  The log-volume role was already redefined as "the varying part of a remainder" to carry the improper factor's linear part.
  One term that is not a normal, such as the half-Cauchy prior on the scale that eight schools usually carries, makes the whole model unlinearizable, and robust losses, which real pose-graph data needs, have no route either.
  The entry gives a term without a residual a place in a linearization, and weighs three routes (`EVALUATION.md`, the survey of inference algorithms).
  One is the term's gradient and Hessian products through the core's second-order passes, which is general and brings with it the Newton step and the observed information, a Laplace approximation that keeps the residual-weighted curvature Gauss–Newton drops.
  The other is a distribution that declares its expected information in its parents' coordinates, as the normal declares its information root, lifted by the parents' Jacobians: positive semidefinite, of the linearization's shape, and for a robust loss such as the Student-t's a constant multiple of the normal's rows, though its marginals do not know which measurements are outliers. A likelihood factor has no expected information, so it cannot take this route.
  The third is the one factor-graph libraries take for robust losses: the whitened residual reweighted by the loss at its current value, which keeps the term in the residual assembly and its Laplace approximation.
  The entry also damps Fisher scoring: `LevenbergMarquardt` refuses the Fisher assembly (`NotAnIntegral`), yet its gain ratio needs only the objective and a model with the objective's gradient, which the Fisher rows have; without it a model with a free scale, the calibration case, has no damped route to a mode.
  It comes before Hamiltonian Monte Carlo, whose example needs a mode and a Laplace approximation to start from.
  This is analysis in `EVALUATION.md`, not yet run.
  ```
  /wf-start terms-without-residuals feature A place in a linearization for a term that binds no residual, and damped Fisher scoring, revises linearization
  ```

- **Hamiltonian Monte Carlo.** The first estimator not based on a mode, probably a crate of its own.
  Most of what it needs exists: `Program::gradient` is batched, so the batch is a set of parallel chains; `Space::Positive` carries its log-volume, so the change of variables is already in the density; seeded sampling gives the momenta.
  The log-density gradient flattened by layout, which it needs, still lives in the tests (`gradient` in `crates/skoro-linearize/tests/common/mod.rs`) and moves into the library with it.
  Leapfrog through a retraction on SO(3) is not obviously volume-preserving or reversible, so the first entry refuses curved and discrete quantities by structure; curved quantities are the next item, and a metric that varies is in the eventual future.
  The scope is static HMC with dual-averaging step-size adaptation first, and NUTS as a follow-up; the Metropolis-adjusted Langevin algorithm, which is HMC with one leapfrog step, comes with it as the simplest sampler.
  The Laplace approximation serves as mass matrix and starting point: a mode, then Laplace, then HMC, all read off one model.
  Its diagnostics are R-hat and effective sample size, enough for the example to show its chains converged.
  Its example is eight schools (a positive scale τ with a half-Cauchy prior), the standard comparison with Stan.
  It is written non-centred, `θ_j = μ + τ η_j`: the centred form has no mode, since with every `θ_j = μ` its density in the logarithmic chart of τ grows like `τ^(-(J-1))` as `τ → 0`, so the mode and the Laplace approximation exist only for the non-centred form (`EVALUATION.md`, the survey of inference algorithms).
  Written so by hand, it has no funnel in the coordinates the chains move in; the centred form and its funnel are left to the item on a metric that varies.
  The archive's climate model (`08-example-climate-mcmc`) waited on seeded sampling, HMC and the positive reals, and can follow once its ODE is unrolled in the graph.
  ```
  /wf-start hmc feature Static Hamiltonian Monte Carlo on flat quantities, with eight schools as its example
  ```

- **Hamiltonian Monte Carlo on curved quantities.** Leapfrog through a retraction fails to be reversible only when the chart moves with the state.
  A global chart at one fixed anchor `a` covers the whole type but a set of measure zero, so `t ↦ a ⊕ t` makes the problem ordinary HMC on `ℝ^d`, with the chart's log-volume added to the log-density, and the first entry's sampler applies unchanged.
  It needs nothing new from the core: the gradient in the fixed chart is the gradient a pass gives in the point's own chart, carried through the Jacobian of the change of chart, one tangent pass over the type's retraction and inverse.
  The Cayley chart of the rotations covers each rotation once, so a posterior has no mirrored mode to visit, though a draw is stored in either sign and the diagnostics are computed on functions of the rotation rather than on raw quaternion components; the anchor is the mode, and the Laplace covariance in that chart is the mass matrix.
  The chart is not conformal: at a turn of θ from the anchor it shrinks lengths by `cos²(θ/2)` along the radius and by `cos(θ/2)` across it, both of which go to zero at the half-turns, so the chain is well conditioned only on a posterior that stays well inside a half-turn of its mode.
  The entry also weighs the alternative, an integrator along geodesics, which would have a type declare a geodesic step as it declares its volume, and which limits the mass matrix on a rotation to a multiple of the identity.
  This is analysis in `EVALUATION.md`, not yet run.
  ```
  /wf-start hmc-curved feature Hamiltonian Monte Carlo on curved quantities, in a fixed global chart
  ```

- **The umbrella re-exports every crate.** `skoro` re-exports only `skoro-core` and `skoro-tensor`, so a model that uses a distribution or an estimator also depends on `skoro-normal`, `skoro-uniform`, `skoro-spatial`, `skoro-linearize` or `skoro-elimination` by name, and the examples, the guide and the umbrella's own tests import them directly.
  By the release, `skoro` is the one crate a user depends on: it re-exports each feature crate under a module of its own (`skoro::normal`, `skoro::uniform`, `skoro::spatial`, `skoro::linearize`, `skoro::elimination`), every crate added before the release joins it, and the examples, the guide's snippets and the umbrella's tests are written against `skoro` alone, as `tests_use_the_public_interface_alone` already requires of the core.
  The feature crates depend on the core and never on `skoro`, so the umbrella can depend on all of them without a cycle; whether some of them sit behind cargo features is settled in the entry.
  ```
  /wf-start umbrella chore The skoro crate re-exports every crate of the workspace, and the examples and guide use it alone
  ```

- **The release, 0.1.0.** The README leads with what skoro does and a model of twenty lines or so that compiles, with the development process lower down, and is included in the `skoro` crate's documentation (`#![doc = include_str!("../README.md")]`) so that the snippet is doc-tested.
  It does not claim real-time use until the cost of a single row, below, has been measured again, and wgpu goes in it only if an example shows it paying off.
  Each crate gets rustdoc, a docs.rs build check, crates.io metadata, and `cargo publish --dry-run` in dependency order.
  A `CHANGELOG` and the tag finish it.
  ```
  /wf-start release-0-1-0 chore The README, the crates' documentation and metadata, the changelog and the tag
  ```

## Eventual future

These are in no order; each returns when something needs it.

- **A single row is slow.** In the `sliding_window` loop, with a window of five poses, one linearization took 10.5 ms for one row and 56.5 ms for a hundred, and one elimination with its step took 0.19 ms for one row.
  A first linearization on a new snapshot cost the same as a second, so the time is not compilation.
  The cost is mostly a fixed overhead of running each program's many small steps on the tensor backend, which a batch of rows shares and a single row pays in full; that reading is an inference from the timings, and the entry starts by profiling it.
  This suits many datasets or hypotheses at once, and does not yet suit one estimator running in real time, which the docs name as an audience.
  Fused kernels are the lever: `skoro-spatial` brought SO(3) and SE(3) back as embedded types whose maps are composites, and the archive's fused exponential and logarithm kernels for cubecl (`crates/skoro-so3/src/ops/exp/cube.rs`, `ops/log/cube.rs`) are the prior art.
  ```
  /wf-start fused-kernels feature The cost of a single row, and fused kernels for the maps that dominate it
  ```

- **A metric that varies: Riemannian manifold HMC.** For a posterior whose scale changes by orders of magnitude, such as the funnel of the centred eight schools, the sampler takes a metric `G` that changes from point to point.
  Its integrator needs `G`, its log-determinant, solves against it, and the derivative of `G`, which costs one order more than whatever `G` is built from.
  A metric built from the Hessian needs third derivatives, a fourth variant of `Kernel` and a third-order rule for every primitive, and is not the route.
  The Fisher linearization already assembles `G = JᵀJ` from first derivatives and the information roots, so its derivative along a direction `u` is a tangent pass and a Hessian pass seeded with `J u`, both of which exist, and `skoro-elimination` gives the log-determinant and the solves.
  What is new is outside the core: an implicit integrator whose fixed-point iterations stop row by row, and the derivative of the log-determinant, at one second-order pass per coordinate.
  The one change to the core to consider is a Hessian pass that takes a seed for each velocity column, so that this costs one pass.
  The entry weighs a cheaper cure for a funnel first: sampling in the whitened residual every normal binds, the non-centred parameterization read off the model rather than written by hand as the HMC example's is, which needs cuts, below, and no metric.
  Its example is the centred eight schools.
  This is analysis in `EVALUATION.md`, not yet run.
  ```
  /wf-start riemannian-hmc feature A metric that varies from point to point, from the Fisher linearization
  ```

- **A type has one chart.** A type's retraction is both how an optimizer steps and what every normal on the type means.
  A pose with noise in its own frame is `Pose`, one with position noise in a fixed frame is a product of a rotation and a vector, and noise given in the group's exponential coordinates has no type.
  A type whose natural chart is not global holds a measurement only as a likelihood factor (`factor_at`), which has no sampler, so that part of the model cannot be generated.
  This is acceptable while the three spatial types cover what is modelled.
  If it starts to bind, the entry lets `Normal` take its chart as an argument that defaults to the type's, and leaves the rule that the chart be global as it is.
  ```
  /wf-start normal-charts feature A chart given to the normal as an argument, revises normal
  ```

- **Draws are made on the host.** Every sampler runs on the host in double precision, with one chain of keyed hashes per row and quantity, and enters a generation program as a step with no derivative.
  This is deliberate, and gives reproducible draws at any precision.
  Its costs come later: particle methods on a GPU pay a round trip per drawn quantity, and no gradient passes through a draw, so a variational method that differentiates through its samples needs the noise as an explicit input.
  Sequential Monte Carlo, below, is the first client of the round trip; automatic-differentiation variational inference and its natural-gradient variants are the clients of the gradient, and wait on this entry.
  ```
  /wf-start device-sampling feature Draws on the target's device, and derivatives through a draw
  ```

- **Incremental elimination.** A model that keeps growing because nothing is retired, such as full smoothing with loop closures, is eliminated whole at every change, though each change adds only a few terms.
  The `Schedule`'s junction tree is the static half of iSAM2's Bayes tree (Kaess et al., 2012); the other half re-eliminates only the cliques on the paths from the blocks a change touches to the root, keeps the rest of the factorization, and relinearizes only the blocks whose step passed a threshold.
  The diff between snapshots is read off the ids of their terms and quantities, so the model is not told what changed.
  It does not address the cost of a single row: in the `sliding_window` loop elimination is 0.19 ms against 10.5 ms of linearization, and a retired window is bounded already.
  Relinearizing only some blocks would cut the linearization too, but needs the linearizer to compile and run programs over a subset of terms, which it cannot do today.
  This is analysis in `EVALUATION.md`, not yet run.
  ```
  /wf-start incremental-elimination feature Re-eliminating only the cliques a change of the model touches, revises linearization
  ```

- **Sequential Monte Carlo.** A bootstrap particle filter needs only draws of the transition and the observation's log-density, both of which the core gives; resampling permutes the rows of a batch.
  A sampler that tempers from the prior to the posterior with HMC moves (Del Moral, Doucet and Jasra, 2006) gives an unbiased estimate of the evidence for a posterior far from Gaussian, which nothing else in the plan does, and annealed importance sampling is its special case.
  A Rao–Blackwellized filter samples part of the state and carries the rest through the linearizer per particle.
  Each draw is a round trip from the device until draws on the device land, so the entry measures that cost first.
  This is analysis in `EVALUATION.md`, not yet run.
  ```
  /wf-start smc feature Particle filters and tempered samplers, with an estimate of the evidence
  ```

- **Sigma-point filtering.** The unscented transform propagates a Gaussian through a map by evaluating it at `2d + 1` points, with no derivative.
  It is the one Gaussian filter for a model with an operation that has no derivative rules, and it reads the same model as the linearizer.
  This is analysis in `EVALUATION.md`, not yet run.
  ```
  /wf-start sigma-points feature The unscented transform, for Gaussian filtering without derivatives
  ```

- **Not planned.** The survey of inference algorithms (`EVALUATION.md`) also covers the ensemble Kalman filter, Stein variational gradient descent, normalizing flows and controlled diffusions, and expectation propagation.
  None is planned: the first is for dimensions of a million, the second couples the rows of a batch that skoro keeps independent, the third needs trained networks, and the fourth asks each term for tilted moments, a declaration different in kind from any a distribution makes today.

- **The operation bridging burn's automatic differentiation.** From the charter: an operation that runs burn's `Autodiff` inside its kernel, translating between skoro's tensors and burn's at its boundary, for code that is easier to differentiate that way than to write as primitives.
  ```
  /wf-start autodiff-bridge feature An operation that runs burn's Autodiff inside its kernel
  ```

- **A serialization format.** From the charter: a model written out, by the names and parameters of its operations, and read back through the registry, so that an autonomous system can write out its model when it fails, to be inspected later.
  ```
  /wf-start serialization feature A model written out and read back through the registry
  ```

- **Python bindings.** From the charter: bindings that drive a fixed registry of distributions and operations.
  ```
  /wf-start python-bindings feature Python bindings over a fixed registry of distributions and operations
  ```

- **Spheres.** The core's acceptance test runs on a sphere declared in its tests (`crates/skoro/tests/common/sphere.rs`), which goes when the library's own spheres replace it.
  ```
  /wf-start spheres feature The library's own spheres, replacing the sphere of the core's tests
  ```

- **Pointwise operations the core does not have.** The absolute value, and the general division and reciprocal of Euclidean values, where the core offers them for positive values alone and division by a literal.
  The archive has them in `crates/skoro-pointwise/`.
  ```
  /wf-start pointwise feature The absolute value, and division and the reciprocal of Euclidean values
  ```

- **Compiling between cuts.** The archive's `Graph::compile_between` (`crates/skoro-graph/src/graph.rs`) stopped a program's backward walk at named interior nodes, each a differentiable leaf whose value the caller gave, so that a Jacobian could be taken at a mapped value.
  The core's programs have the quantities and constants their targets reach for roots, so a cut changes that.
  Linearization did without one, reading the tangent a normal binds as a node of its own; a cut returns with the first entry that needs one, as a method on `Query` that changes no existing query.
  Keep the archive's lesson: a cut is never folded through, since a value below it that looks fixed moves with the cut, and folding it would give the right values with the derivatives silently missing.
  ```
  /wf-start cuts feature A query compiled between cuts, revises core
  ```
