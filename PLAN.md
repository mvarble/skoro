# Plan

What skoro means to do next, kept here so the plan outlives any one session. Each item returns as
an entry of its own, through the workflow (`.agents/reference/workflow.md`), with its mathematics
written down first; nothing here starts one. The archive is the branch `archive`, tagged
`archive-2026-10-09`; read it with `git show archive-2026-10-09:<path>`.

## Immediate future

The goal is an initial release that shows what skoro does: a model built once, then estimated,
filtered and sampled without being told its structure again. In this order:

1. **Examples that run.** The archive's example pages (`site/src/content/guide/07-example-*` to
   `10-example-*`) described problems it could not yet solve; `main` now can solve some of them.
   Each example is a real `cargo run --example` target, run in CI, and quoted by its guide page the
   way the guide quotes tests today.
   - *A pose graph on SE(3)*: MAP by the Gauss–Newton/Levenberg–Marquardt loop of
     `skoro-linearize`, then Laplace marginals from `schedule_keeping`; on a standard g2o dataset
     if loading one is cheap enough.
   - *A sliding-window filter by retirement*: retiring one tick after another, each replaced by
     its linearized improper factor (the archive's `09-example-sliding-window`).
   - Writing these is the test of the public API. Note what they need that only the tests have:
     the log-density gradient flattened by layout, for one, lives in
     `crates/skoro-linearize/tests/common/mod.rs` (`gradient`), and HMC needs it too.
2. **Hamiltonian Monte Carlo**, the first estimator not based on a mode, probably a crate of its
   own. Most of what it needs exists: `Program::gradient` is batched, so the batch is a set of
   parallel chains; `Space::Positive` carries its log-volume, so the change of variables is
   already in the density; seeded sampling gives the momenta. For its mathematics:
   - *Curved spaces.* Leapfrog through a retraction on SO(3) is not obviously volume-preserving or
     reversible. The first entry refuses curved and discrete quantities by structure; Riemannian
     HMC is a later entry.
   - *Scope.* Static HMC with dual-averaging step-size adaptation first; NUTS as a follow-up.
   - *Laplace as mass matrix and starting point*: MAP, then Laplace, then HMC, all read off one
     model.
   - *Diagnostics*: R-hat and effective sample size, enough for the example to show its chains
     converged.
   - *An example*: eight schools (a funnel, and a positive scale τ), the standard comparison
     with Stan. The archive's climate model (`08-example-climate-mcmc`) waited on seeded
     sampling, HMC and the positive reals; it can follow once its ODE is unrolled in the graph.
3. **The release, 0.1.0.**
   - *README*: lead with what skoro does and a model of twenty lines or so that compiles; the
     development process goes lower. Include it in the `skoro` crate's documentation
     (`#![doc = include_str!("../README.md")]`) so the snippet is doc-tested.
   - *Crates*: rustdoc for each crate, a docs.rs build check, crates.io metadata, and
     `cargo publish --dry-run` in dependency order.
   - A `CHANGELOG`, and the tag.
   - wgpu goes in the README only if an example shows it paying off.

## Eventual future

### From the charter

- **The operation bridging burn's automatic differentiation**: an operation that runs burn's
  `Autodiff` inside its kernel, translating between skoro's tensors and burn's at its boundary, for
  code that is easier to differentiate that way than to write as primitives.
- **A serialization format**, so that a model can be written out, by the names and parameters of
  its operations, and read back through the registry: an autonomous system writing out its model
  when it fails, to be inspected later.
- **Python bindings**, driving a fixed registry of distributions and operations.

### From the archive

- **Fused kernels for the rotations.** `skoro-spatial` brought SO(3) and SE(3) back as embedded
  types whose maps are composites; the archive's fused exponential and logarithm kernels for cubecl
  (`crates/skoro-so3/src/ops/exp/cube.rs`, `ops/log/cube.rs`) return if the composites prove slow.
- **Spheres.** The core's acceptance test runs on a sphere declared in its tests
  (`crates/skoro/tests/common/sphere.rs`), which goes when the library's own spheres replace it.
- **Pointwise operations the core does not have**: the absolute value, and the general division
  and reciprocal of Euclidean values, where the core offers them for positive values alone and
  division by a literal: `crates/skoro-pointwise/`.
- **Compiling between cuts**: `Graph::compile_between` (`crates/skoro-graph/src/graph.rs`) stopped
  a program's backward walk at named interior nodes, each a differentiable leaf whose value the
  caller gave, so that a Jacobian could be taken at a mapped value. The core's programs have the
  quantities and constants their targets reach for roots, so a cut changes that. Linearization did
  without one, reading the tangent a normal binds as a node of its own; a cut returns with the
  first entry that needs one, as a method on `Query` that changes no existing query. Keep the
  archive's lesson: a cut is never folded through, since a value below it that looks fixed moves
  with the cut, and folding it would give the right values with the derivatives silently missing.
