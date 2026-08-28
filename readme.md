# skoro

Skoro is a probabilistic modeling library designed for speed, modularity, and extensibility.
Skoro offers a model-building API with descriptive intermediate representations which allow for a wide variety of inference algorithms.

## Design and Structure

Skoro is designed to bridge the gap between hierarchical probabilistic modeling, automatic differentiation (autodiff), and execution on high-performance `burn` backends.
It breaks this domain down into a few distinct crates:

- **`skoro-core`**: The foundational types and traits, including ID management (`Id`, `IdGenerator`), traits for packing user types into tensor backends (`Packable` / `NumComponents`), traits for sampling stochastic tensors (`Sampler`), and the indexed sequence abstraction (`SequenceMutator`).
- **`skoro-backend`**: A hardware abstraction layer built on top of `burn`.
  Skoro explicitly ignores `burn`'s automatic differentiation engine, using it strictly for high-performance tensor execution (CPU via `ndarray` and GPU via `cubecl`).
  It provides geometric type aliases like `Point<B>`, `TangentVector<B>`, and `TangentFrame<B>` representing batched elements via shape conventions over `burn`'s `Tensor`.
- **`skoro-operation`**: The custom automatic differentiation engine.
  Since probabilistic inference algorithms (like Hamiltonian Monte Carlo, Extended Kalman Filters, or MAP estimation with Fisher information) often require both first-order (Jacobian, Vector-Jacobian Product) and second-order (Hessian, Hessian-Vector Product) derivatives natively aware of non-linear manifolds, Skoro has a bespoke differentiation system.
- **`skoro-riemann`**: A differential geometry framework providing the `Manifold` / `DynManifold` traits along with flat manifold representations (`Real`, `Euclidean<D>` / `EuclideanD`, and `EuclideanMatrix`).
- **`skoro-graph`**: The probabilistic Bayesian belief DAG (Directed Acyclic Graph).
  This represents the hierarchical structure of a user's model through its cascade of conditional distributions.
  The graph allows us to sequence operations like random sampling and likelihood calculations.
  Evaluation is not limited to end-to-end state-to-density evaluations; subsets of the DAG can be selected, compiled into a sequential program, and evaluated.
- **`skoro-model`**: The high-level facade mapping user API concepts (like `ModelClient` and `Q` quantity proxies) down into the backend `Graph` execution logic.
  This offers type-gating on the operations we add to the graph as well as lifetimes for the quantities in the graph for garbage collection.
- **`skoro-linalg`**: Backend-agnostic linear algebra primitives (`Matmul`, `Transpose`, `CrossProduct`, vector norms) and the lower-triangular matrix manifold (`Tril`), used e.g. for covariance factors.
  These are _differentiable graph operations_; the dense factorizations an EKF or Levenberg-Marquardt iteration needs (Cholesky, triangular solve, QR least-squares) are a separate concern, and `docs/skoro.md` §3.3 explains why they run on the host rather than on the backend.
- **`skoro-so3`**: The SO(3) rotation group (`SO3`, `SO3Matrix`) with differentiable `exp` (Rodrigues' formula), `log`, retraction, compose, rotate/unrotate, and invert operations.
- **`skoro-se3`**: The SE(3) rigid-body transform group (`SE3`, `SE3Pose`), composing SO(3) rotation operations with translation.
- **`skoro-pointwise`**: Differentiable elementwise operations (`log`, `abs`).
- **`skoro-normal`**: A manifold-aware Gaussian distribution (`Normal<B, M>`) parameterized by a `mean: Q<B, M>` and a lower-triangular factor `std_dev: Q<B, Tril>`.
  It also supplies the _structural_ claims an inference algorithm bounds on — `GaussianFactor` on a single distribution, `GaussianStructure` on a whole model — alongside the `GaussianModelClient` builder that accumulates them.
- **`skoro-dense`**: Host-resident dense linear algebra in `f64`, one matrix at a time, backed by `faer`: Cholesky, triangular and SPD solves, and a column-pivoted-QR least squares.
  A leaf crate depending on nothing else in the workspace, so only the estimators pay for it. `docs/skoro.md` §3.3 explains why this runs on the host rather than on the backend.
- **`skoro`**: The top-level umbrella crate aggregating the workspace and re-exporting the sub-crates (`skoro::dense`, `skoro::prelude`).
