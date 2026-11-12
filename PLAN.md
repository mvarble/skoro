# Plan

What skoro means to do next, kept here so the plan outlives any one session.
Each item returns as an entry of its own, through the workflow (`.agents/reference/workflow.md`), and ends with the command that would open it; nothing here starts one.
The slug and the kind in each command are proposals, settled when the entry is opened.
Items that cite measurements take them from `EVALUATION.md`, the evaluation of the core of 2026-11-10, which says how each was made.
The archive is the branch `archive`, tagged `archive-2026-10-09`; read it with `git show archive-2026-10-09:<path>`.

## Immediate future

The goal is an initial release that shows what skoro does: a model built once, then estimated, filtered and sampled without being told its structure again.
The items are in the order they should be taken.

- **A broad normal on a rotation is not a weak prior.** Under the Cayley chart a tangent of length `r` is a turn by `2·arctan(r/2)`, and the density of a normal on `Rotation` against Haar measure is the Gaussian times `(1 + r²/4)²`, so its mean stops being a mode once its standard deviation passes 1 rad, and the mass then goes to the half-turns from the mean.
  Computed from the chart (not drawn), the median angle between a drawn rotation and the mean is 1.31 rad at σ = 1, 2.32 rad at σ = 3, 2.88 rad at σ = 10, 3.12 rad at σ = 100 and 3.14 rad at σ = 1000.
  So a very large standard deviation, the common way to write "unknown", gives a prior concentrated on the half-turns, where the chart is cut and a term is non-finite, and during optimization such a term pushes the estimate away from its mean.
  The entry brings the uniform distribution on the compact types, which is how "unknown" is said, and decides whether a constant scale above about one radian on a curved type is refused or documented.
  The measurements in `EVALUATION.md` were taken under the stereographic chart that `quaternion-signs` replaced, and the entry draws them again.
  ```
  /wf-start uniform feature The uniform distribution on compact types, and broad scales of a normal on a curved type, revises normal
  ```

- **Estimation reads only Gaussian-shaped terms.** `Linearizer` refuses a model with any term that binds no residual (`NoResidual`), and the roles it reads are the normal's.
  The log-volume role was already redefined as "the varying part of a remainder" to carry the improper factor's linear part.
  One term that is not a normal, such as the half-Cauchy prior on the scale that eight schools usually carries, makes the whole model unlinearizable, and robust losses, which real pose-graph data needs, have no route either.
  The entry gives a term without a residual a place in a linearization, such as its gradient and Hessian products through the core's second-order passes.
  It comes before Hamiltonian Monte Carlo, whose example needs a mode and a Laplace approximation to start from.
  ```
  /wf-start terms-without-residuals feature A place in a linearization for a term that binds no residual, revises linearization
  ```

- **Hamiltonian Monte Carlo.** The first estimator not based on a mode, probably a crate of its own.
  Most of what it needs exists: `Program::gradient` is batched, so the batch is a set of parallel chains; `Space::Positive` carries its log-volume, so the change of variables is already in the density; seeded sampling gives the momenta.
  The log-density gradient flattened by layout, which it needs, still lives in the tests (`gradient` in `crates/skoro-linearize/tests/common/mod.rs`) and moves into the library with it.
  Leapfrog through a retraction on SO(3) is not obviously volume-preserving or reversible, so the first entry refuses curved and discrete quantities by structure; curved quantities are the next item, and a metric that varies is in the eventual future.
  The scope is static HMC with dual-averaging step-size adaptation first, and NUTS as a follow-up.
  The Laplace approximation serves as mass matrix and starting point: a mode, then Laplace, then HMC, all read off one model.
  Its diagnostics are R-hat and effective sample size, enough for the example to show its chains converged.
  Its example is eight schools (a funnel, and a positive scale τ), the standard comparison with Stan.
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

- **A metric that varies: Riemannian manifold HMC.** For a posterior whose scale changes by orders of magnitude, such as the funnel of eight schools, the sampler takes a metric `G` that changes from point to point.
  Its integrator needs `G`, its log-determinant, solves against it, and the derivative of `G`, which costs one order more than whatever `G` is built from.
  A metric built from the Hessian needs third derivatives, a fourth variant of `Kernel` and a third-order rule for every primitive, and is not the route.
  The Fisher linearization already assembles `G = JᵀJ` from first derivatives and the information roots, so its derivative along a direction `u` is a tangent pass and a Hessian pass seeded with `J u`, both of which exist, and `skoro-elimination` gives the log-determinant and the solves.
  What is new is outside the core: an implicit integrator whose fixed-point iterations stop row by row, and the derivative of the log-determinant, at one second-order pass per coordinate.
  The one change to the core to consider is a Hessian pass that takes a seed for each velocity column, so that this costs one pass.
  The entry weighs a cheaper cure for a funnel first: sampling in the whitened residual every normal binds, the non-centred parameterization read off the model, which needs cuts, below, and no metric.
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
  ```
  /wf-start device-sampling feature Draws on the target's device, and derivatives through a draw
  ```

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
