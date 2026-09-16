# skoro

Skoro is a probabilistic modeling library designed for speed, modularity, and extensibility.
Skoro offers a model-building API with descriptive intermediate representations which allow for a wide variety of inference algorithms.

## Design and Structure

Skoro is designed to bridge the gap between hierarchical probabilistic modeling, automatic differentiation (autodiff), and execution on high-performance `burn` backends.
It breaks this domain down into a few distinct crates:

- **`skoro-backend`**: A hardware abstraction layer built on top of `burn`.
  Skoro explicitly ignores `burn`'s automatic differentiation engine, using it strictly for high-performance tensor execution (CPU via `burn-flex` and GPU via `cubecl`).
  It provides geometric type aliases like `Point<B>`, `TangentVector<B>`, and `TangentFrame<B>` representing batched elements via shape conventions over `burn`'s `Tensor`.

- **`skoro-core`**: The foundational types and traits, including ID management (`Id`, `IdGenerator`), traits for packing user types into tensor backends (`Packable` / `NumComponents` / `Cardinality`), traits for sampling stochastic tensors (`Sampler`), the indexed sequence abstraction (`SequenceMutator`), and the binding label `Role` — which lives here rather than beside the graph's edges so that `SkoroError` can name one.

- **`skoro-operation`**: The custom automatic differentiation engine.
  Since probabilistic inference algorithms (like Hamiltonian Monte Carlo, Extended Kalman Filters, or MAP estimation with Fisher information) often require both first-order (Jacobian, Vector-Jacobian Product) and second-order (Hessian, Hessian-Vector Product) derivatives natively aware of non-linear manifolds, Skoro has a bespoke differentiation system.

- **`skoro-riemann`**: A differential geometry framework providing the `Manifold` / `DynManifold` traits along with flat manifold representations (`Real`, `Euclidean<D>` / `EuclideanD`, and `EuclideanMatrix`).
  Types which implement `Manifold` provide the context for retraction/inverse-retraction operations, so that optimization over non-linear manifolds may occur.
  The types also act as tags which allow us to type-gate our operations, which under the hood operate over untyped batched tensors.

- **`skoro-graph`**: The probabilistic Bayesian belief DAG (Directed Acyclic Graph).
  This represents the hierarchical structure of a user's model through its cascade of conditional distributions.
  The graph allows us to sequence operations like random sampling and likelihood calculations.
  Evaluation is not limited to end-to-end state-to-density evaluations; subsets of the DAG can be selected, compiled into a sequential program, and evaluated.
  Beside the parent edges, the graph stores **bindings**: labelled reference edges `(from, role, to)` from a log-density term to the nodes that play a role for it.
  The compiler never reads a binding, but removal and garbage collection treat one as a reference: `remove` refuses to take a bound node from under its term, and `remove_closure` takes the term with it.
  A graph carries two densities over the same nodes: the posterior `p(x|z)` it scores, and the generative model `q(x|z)` it samples from.
  Each term declares whether it is a summand of each — membership in `p` is stored on the term, membership in `q` is read off whether the quantity it scores is still being generated — so a *proposal-only* term can generate a quantity the posterior does not score.
  That is what lets a Monte-Carlo algorithm work over a distribution whose density is known but which offers no sampler: propose from something you can draw from, and let the log-weight `p − q` correct.

- **`skoro-model`**: The high-level facade mapping user API concepts (like `ModelClient` and `Q` quantity proxies) down into the backend `Graph` execution logic.
  This offers type-gating on the operations we add to the graph as well as lifetimes for the quantities in the graph for garbage collection.
  The idea is that with a `ModelClient`, the user can add constant/parameter/stochastic nodes to the graph and obtain an instance `x` of type `Q<T>` where `T` is the manifold/discrete-set which supports the quantity.
  From there, they may apply operations `let y: Q<S> = f(x);` to get mapped nodes.
  The API here is as intuitive as if `x` and `y` were structs with concrete data, despite the fact that they represent nodes in a graph for _realizing_ our random quantities.
  A `Q` is the ownership token: the collector is a mark-and-sweep whose roots are the live handles and the model's likelihood terms, following parent edges and bindings, so a quantity nothing reaches is reclaimed whatever kind of node it is.
  A bare `Id` is a weak reference by comparison — it may name nothing after a `cleanup()`.
  A `Distribution` returns a `Factor`: its log-density plus the bindings it declares under the roles in `roles` (`ARGUMENT`, `RESIDUAL`, `LOCATION`, `WHITENING`, `NORMALIZER_PARENT`, `FISHER_ROOT`), recorded once when the quantity is added; a free-standing factor is attached the same way with `add_factor`.
  Inference algorithms read back the bindings they need: `least_squares::residual_layout` gathers each term's `RESIDUAL` as the rows of `J`, and `information` assembles the expected (Fisher) information from each term's `FISHER_ROOT`, a matrix node `C` with `CᵀC` equal to the term's expected information over its arguments.
  These readers — and `skoro-normal`'s `GaussianBlock::all` — skip a term that bound nothing under the role they want — a distribution need not offer a Fisher root, and one with no closed-form expected information should not — so what they assemble can cover a subset of the log-density without saying so.
  `coverage::terms_missing` names those terms and `coverage::require_role` refuses over them, which is how a consumer that pairs a curvature with a gradient keeps the two over the same terms.
  `add_quantity` adds a quantity the model both generates and scores; `add_proposal` adds one it generates without scoring, leaving the posterior's account of it to a separate `add_factor`.
  `ModelClientOps::{log_density, log_proposal, log_weight}` evaluate `p`, `q` and their difference, and each `FactorInfo` reports which of the two it belongs to — objective-assembling readers such as `residual_layout` and `information_blocks` use that to skip the terms `p` does not contain.

- **`skoro-linalg`**: Backend-agnostic linear algebra primitives (`Matmul`, `Transpose`, `CrossProduct`, vector norms) and the lower-triangular matrix manifold (`Tril`), used e.g. for covariance factors.
  `FixedLinearMap` / `linear_map` apply a host-supplied matrix as a differentiable operation, which is how a builder without access to constants — a distribution — puts a known matrix into the graph; the matrix follows its input to whatever device that lives on.
  `SparseLinearMap` / `sparse_linear_map` do the same for a large, sparse matrix, storing only its nonzero entries and building its indices on the input's device; the Gaussian Fisher root is built this way.
  These are _differentiable graph operations_; the dense factorizations an EKF or Levenberg-Marquardt iteration needs (Cholesky, triangular solve, QR least-squares) are a separate concern, and `docs/skoro.md` §4.2 explains why they run on the host rather than on the backend.

- **`skoro-so3`**: The SO(3) rotation group (`SO3`, `SO3Matrix`) with differentiable `exp` (Rodrigues' formula), `log`, retraction, compose, rotate/unrotate, and invert operations.

- **`skoro-se3`**: The SE(3) rigid-body transform group (`SE3`, `SE3Pose`), composing SO(3) rotation operations with translation.
- **`skoro-pointwise`**: Differentiable elementwise operations (`log`, `abs`).

- **`skoro-normal`**: A manifold-aware Gaussian distribution (`Normal<B, M>`) parameterized by a `mean: Q<B, M>` and a lower-triangular factor `std_dev: Q<B, Tril>`.
  A Gaussian conditional enters a model through `client.add_quantity(Normal::new(mean, std_dev))`. Its factor binds the whitened residual `r = L⁻¹(x ⊖ μ)`, binds `μ` and `L` separately so a filter can read `H = ∂μ/∂x` and `R = L Lᵀ` apart, and binds a Fisher information root over `[x, μ, L]`. `GaussianBlock` is the read side of those bindings, and `JointNormal` / `add_joint_factor` attach a joint factor over quantities the model already holds.

- **`skoro-dense`**: Host-resident dense linear algebra in `f64`, one matrix at a time, backed by `faer`: Cholesky, triangular and SPD solves, and a column-pivoted-QR least squares.
  A leaf crate depending on nothing else in the workspace, so only the estimators pay for it. `docs/skoro.md` §4.2 explains why this runs on the host rather than on the backend.

- **`skoro`**: The top-level umbrella crate aggregating the workspace and re-exporting the sub-crates (`skoro::dense`, `skoro::prelude`).
