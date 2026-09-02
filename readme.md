# skoro

Skoro is a probabilistic modeling library designed for speed, modularity, and extensibility.
Skoro offers a model-building API with descriptive intermediate representations which allow for a wide variety of inference algorithms.

## Design and Structure

Skoro is designed to bridge the gap between hierarchical probabilistic modeling, automatic differentiation (autodiff), and execution on high-performance `burn` backends.
It breaks this domain down into a few distinct crates:

- **`skoro-backend`**: A hardware abstraction layer built on top of `burn`.
  Skoro explicitly ignores `burn`'s automatic differentiation engine, using it strictly for high-performance tensor execution (CPU via `ndarray` and GPU via `cubecl`).
  It provides geometric type aliases like `Point<B>`, `TangentVector<B>`, and `TangentFrame<B>` representing batched elements via shape conventions over `burn`'s `Tensor`.

- **`skoro-core`**: The foundational types and traits, including ID management (`Id`, `IdGenerator`), traits for packing user types into tensor backends (`Packable` / `NumComponents` / `Cardinality`), traits for sampling stochastic tensors (`Sampler`), and the indexed sequence abstraction (`SequenceMutator`).

- **`skoro-operation`**: The custom automatic differentiation engine.
  Since probabilistic inference algorithms (like Hamiltonian Monte Carlo, Extended Kalman Filters, or MAP estimation with Fisher information) often require both first-order (Jacobian, Vector-Jacobian Product) and second-order (Hessian, Hessian-Vector Product) derivatives natively aware of non-linear manifolds, Skoro has a bespoke differentiation system.

- **`skoro-riemann`**: A differential geometry framework providing the `Manifold` / `DynManifold` traits along with flat manifold representations (`Real`, `Euclidean<D>` / `EuclideanD`, and `EuclideanMatrix`).
  Types which implement `Manifold` provide the context for retraction/inverse-retraction operations, so that optimization over non-linear manifolds may occur.
  The types also act as tags which allow us to type-gate our operations, which under the hood operate over untyped batched tensors.

- **`skoro-graph`**: The probabilistic Bayesian belief DAG (Directed Acyclic Graph).
  This represents the hierarchical structure of a user's model through its cascade of conditional distributions.
  The graph allows us to sequence operations like random sampling and likelihood calculations.
  Evaluation is not limited to end-to-end state-to-density evaluations; subsets of the DAG can be selected, compiled into a sequential program, and evaluated.

- **`skoro-model`**: The high-level facade mapping user API concepts (like `ModelClient` and `Q` quantity proxies) down into the backend `Graph` execution logic.
  This offers type-gating on the operations we add to the graph as well as lifetimes for the quantities in the graph for garbage collection.
  The idea is that with a `ModelClient`, the user can add constant/parameter/stochastic nodes to the graph and obtain an instance `x` of type `Q<T>` where `T` is the manifold/discrete-set which supports the quantity.
  From there, they may apply operations `let y: Q<S> = f(x);` to get mapped nodes.
  The API here is as intuitive as if `x` and `y` were structs with concrete data, despite the fact that they represent nodes in a graph for _realizing_ our random quantities.
  A `Q` is the ownership token: the collector is a mark-and-sweep whose roots are the live handles, the model's likelihood terms, and the nodes a recorded factor names, so a quantity nothing reaches is reclaimed whatever kind of node it is.
  A bare `Id` is a weak reference by comparison — it may name nothing after a `cleanup()`.
  Bindings are where a builder writes down what it knows about a least-squares factor — labelled reference edges from a term's density to its residual, its location, its whitening, and whatever its normalizer depends on — for an estimator to read back at solve time through `ModelClientOps::{factors, bound}`; a binding is validated against the graph when it is made, refuses to let its target be removed out from under it, and goes with the term in the same locked call as any mutation that retires it.

- **`skoro-linalg`**: Backend-agnostic linear algebra primitives (`Matmul`, `Transpose`, `CrossProduct`, vector norms) and the lower-triangular matrix manifold (`Tril`), used e.g. for covariance factors.
  These are _differentiable graph operations_; the dense factorizations an EKF or Levenberg-Marquardt iteration needs (Cholesky, triangular solve, QR least-squares) are a separate concern, and `docs/skoro.md` §3.3 explains why they run on the host rather than on the backend.

- **`skoro-so3`**: The SO(3) rotation group (`SO3`, `SO3Matrix`) with differentiable `exp` (Rodrigues' formula), `log`, retraction, compose, rotate/unrotate, and invert operations.

- **`skoro-se3`**: The SE(3) rigid-body transform group (`SE3`, `SE3Pose`), composing SO(3) rotation operations with translation.
- **`skoro-pointwise`**: Differentiable elementwise operations (`log`, `abs`).

- **`skoro-normal`**: A manifold-aware Gaussian distribution (`Normal<B, M>`) parameterized by a `mean: Q<B, M>` and a lower-triangular factor `std_dev: Q<B, Tril>`.
  `add_quantity(Normal::new(μ, L))` is how a Gaussian conditional enters a model: it adds the quantity, and `Normal`'s factor builds the whitened residual `r = L⁻¹(x ⊖ μ)` and binds it alongside `μ` and `L` under their own roles, so a filter can read `H = ∂μ/∂x` and `R = L Lᵀ` apart. `GaussianBlock` is the read side of those bindings.

- **`skoro-dense`**: Host-resident dense linear algebra in `f64`, one matrix at a time, backed by `faer`: Cholesky, triangular and SPD solves, and a column-pivoted-QR least squares.
  A leaf crate depending on nothing else in the workspace, so only the estimators pay for it. `docs/skoro.md` §3.3 explains why this runs on the host rather than on the backend.

- **`skoro`**: The top-level umbrella crate aggregating the workspace and re-exporting the sub-crates (`skoro::dense`, `skoro::prelude`).
