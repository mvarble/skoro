# What to revisit

The restart of 2026-10-09 brought back the core alone. This is what is left: what the archive had
that the core does not bring back, and what the charter names that no entry builds yet. Each
returns as an entry of its own, through the workflow, with its mathematics written down first. The
archive is the branch `archive`, tagged `archive-2026-10-09`; read it with
`git show archive-2026-10-09:<path>`.

## From the archive

- **The normal distribution**, with its joint form, the roots of its Fisher information, and the
  readers of its structure: `crates/skoro-normal/` (`distribution.rs`, `joint.rs`,
  `information.rs`, `structure.rs`). The core's tests keep a private normal of their own
  (`crates/skoro/tests/common/normal.rs`) until then.
- **SO(3) and SE(3)**, as embedded types, with their fused exponential and logarithm kernels for
  cubecl: `crates/skoro-so3/` (`ops/exp/cube.rs`, `ops/log/cube.rs`) and `crates/skoro-se3/`. The
  core's acceptance test runs on a sphere declared in its tests, which goes when a curved type of
  the library replaces it.
- **The linear algebra**: triangular matrices, their inverses and their products with vectors, the
  cross product, and norms: `crates/skoro-linalg/` (`tril/`, `matrix/`, `vector/`).
- **Pointwise operations the core does not have**: the absolute value, and the general division
  and reciprocal of Euclidean values, where the core offers division by a literal alone and
  `e^{-log b}` where `b` is positive: `crates/skoro-pointwise/`.
- **Least squares**: the least-squares readers, the assembly of Jacobians, linearization, and the
  coverage check, which names the terms an objective's readers skip for want of a role, so that an
  objective is known to cover the whole log-density: `crates/skoro-model/src/least_squares.rs`,
  `crates/skoro-model/src/information.rs`, `crates/skoro-model/src/coverage.rs` and
  `crates/skoro-linearize/`.
- **Elimination**, dense and block-sparse, with the marginalization a sliding window retires by:
  `crates/skoro-dense/` (`elimination.rs`, `marginal.rs`, `ordering.rs`). The core retires a
  namespace with a replacement the caller computes; computing one is this entry's.
- **Literal operands as primitives**: the archive gave an operation with a literal operand a
  kernel of its own (`crates/skoro-model/src/ops/`); the core makes a literal a constant node.
- **Compiling between cuts**: `Graph::compile_between` (`crates/skoro-graph/src/graph.rs`) stopped
  a program's backward walk at named interior nodes, each a differentiable leaf whose value the
  caller gave, so that a Jacobian could be taken at a mapped value. The core's programs have the
  quantities and constants their targets reach for roots, so a cut changes that; it returns with
  the first entry that needs one, likely linearization or Gauss–Newton, as a method on `Query` that
  changes no existing query. Keep the archive's lesson: a cut is never folded through, since a
  value below it that looks fixed moves with the cut, and folding it would give the right values
  with the derivatives silently missing.

## From the charter

- **The operation bridging burn's automatic differentiation**: an operation that runs burn's
  `Autodiff` inside its kernel, translating between skoro's tensors and burn's at its boundary, for
  code that is easier to differentiate that way than to write as primitives.
- **A serialization format**, so that a model can be written out, by the names and parameters of
  its operations, and read back through the registry: an autonomous system writing out its model
  when it fails, to be inspected later.
- **Python bindings**, driving a fixed registry of distributions and operations.
