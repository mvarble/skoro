# skoro

Skoro is a probabilistic modeling library designed for speed, modularity, and extensibility.
By implementing abstract intermediate representations, Skoro offers a model-building API which allows users to experiment with a wide variety of inference algorithms.

## Design and Structure

Skoro is designed to bridge the gap between hierarchical probabilistic modeling, automatic differentiation (autodiff), and execution on high-performance `burn` backends.
It breaks this domain down into a few distinct crates:

- **`skoro-core`**: The foundational types and traits, including ID management (`Id`, `IdGenerator`), traits for packing user types into tensor backends (`Packable` / `NumComponents`), traits for sampling stochastic tensors (`Sampler`), and the indexed sequence abstraction (`SequenceMutator`).
- **`skoro-backend`**: A hardware abstraction layer built on top of `burn`.
  Skoro explicitly ignores `burn`'s automatic differentiation engine, using it strictly for high-performance tensor execution (CPU via `ndarray` and GPU via `cubecl`).
  It provides geometric type aliases like `Point<B>`, `TangentVector<B>`, and `TangentFrame<B>` representing batched elements via shape conventions over `burn`'s `Tensor`.
- **`skoro-diff`**: The custom automatic differentiation engine.
  Since probabilistic inference algorithms (like Hamiltonian Monte Carlo, Extended Kalman Filters, or Gauss-Newton optimization) often require both first-order (Jacobian, Vector-Jacobian Product) and second-order (Hessian, Hessian-Vector Product) derivatives natively aware of non-linear manifolds, Skoro has a bespoke differentiation system.
- **`skoro-riemann`**: A differential geometry framework providing the `Manifold` / `DynManifold` traits along with flat manifold representations (`Real`, `Euclidean<D>` / `EuclideanD`, and `EuclideanMatrix`).
- **`skoro-graph`**: The probabilistic DAG (Directed Acyclic Graph).
  This represents the logical structure of a user's model before it gets flattened and executed.
- **`skoro-model`**: The high-level facade mapping user API concepts (like `ModelClient` and `Q` quantity proxies) down into the backend `Graph` execution logic.
- **`skoro-linalg`**: Backend-agnostic linear algebra primitives (`Matmul`, `Transpose`, `CrossProduct`, vector norms) and the lower-triangular matrix manifold (`Tril`), used e.g. for covariance factors.
  These are *differentiable graph operations*; the dense factorizations an EKF or Gauss-Newton iteration needs (Cholesky, triangular solve, QR least-squares) are a separate concern, and `docs/estimator-linear-algebra.md` explains why they run on the host rather than on the backend.
- **`skoro-so3`**: The SO(3) rotation group (`SO3`, `SO3Matrix`) with differentiable `exp` (Rodrigues' formula), `log`, retraction, compose, rotate/unrotate, and invert operations.
- **`skoro-se3`**: The SE(3) rigid-body transform group (`SE3`, `SE3Pose`), composing SO(3) rotation operations with translation.
- **`skoro-pointwise`**: Differentiable elementwise operations (`log`, `abs`).
- **`skoro-normal`**: A manifold-aware Gaussian distribution (`Normal<B, M>`) parameterized by a `mean: Q<B, M>` and a lower-triangular factor `std_dev: Q<B, Tril>`.
- **`skoro`**: The top-level umbrella crate aggregating the workspace (currently a stub that depends on the sub-crates without re-exporting).

---

### The `ModelServer` and `ModelClient`

The `skoro-model` crate acts as the user-facing interface, mapping high-level statistical definitions to the underlying `Graph` and evaluation engines.
The core architecture is split into a server-client model:

- **`ModelServer<B>`**: The stateful engine that wraps and manages the underlying `skoro_graph::Graph<B>`.
- **`ModelClient<B>`**: The primary user-facing structure. It holds a thread-safe reference (e.g., `Arc<RwLock<ModelServer<B>>>`) to the server, acting as a public dispatch interface to add variables, map functions, and execute inference passes without requiring mutable borrows of the whole model.

---

### The `Q` Proxy

When adding constants, parameters, or stochastic variables to the `ModelClient`, it returns a lightweight proxy object: **`Q<B, M>`**.
This quantity handle retains the node's `Id`, a reference to the `ModelClient`, its differentiability state, and phantom data for its underlying manifold `M`.
It allows users to ergonomically perform operations on variables directly.
For instance, conditioning the model for posterior inference is as simple as calling `y1.observe(1.0)`, which delegates back to the underlying `ModelServer`.

---

### Evaluation Modes

The `ModelClient` (and the underlying `ModelServer`) provide methods to evaluate the graph and generate analytical data for inference algorithms. These come in first-order (`_ord1`) and second-order (`_ord2`) flavors, plus the lower-level `_jvp` / `_vjp` / `_hvp` building blocks:

- **`log_density_ord1(points)`**: Triggers compilation of the graph's log-densities, performs the forward pass to evaluate probabilities, and walks the `Tape` backward to return the first-order gradients (scores) for all differentiable quantities.
- **`log_density_ord2(points, tangents)`**: Similar to first-order, but evaluates Hessian-Vector Products to provide both scores and their sensitivities.
- **`evaluate_ids_ord1` / `evaluate_ids_ord2`**: The same first- and second-order passes, but targeting an arbitrary set of node `Id`s rather than the log-densities.

The `Q<B, M>` proxy mirrors this surface with `evaluate`, `evaluate_jvp` / `evaluate_vjp` / `evaluate_hvp`, and `evaluate_ord1` / `evaluate_ord2` for evaluating an individual quantity.

---

### Mapped Functions and Distributions

To build relationships between variables, Skoro maps user logic into the `Graph`:

- **`Distribution<B, M>` Trait**: An interface for registering statistical distributions to the model. It supplies a `quantity_type`, a `sampler`, the relevant `parent_ids`, and a `log_density`, which under the hood generate the required `Stochastic` and `LogDensity` nodes in the `Graph`.
- **Mapped Functions (`register_map1` … `register_map12`)**: A family of utilities that take generic user functions over one to twelve `Q` inputs and lower them into deterministic `Mapped` nodes in the `Graph`, ensuring they are backed by a valid `DiffOperation` (via `GraphOperation`) for autodiff. The arithmetic operator overloads on `Q` (`QAdd`, `QSub`, `QMul`, `QDiv`, and their scalar-literal variants) are implemented on top of these.

---

### The `Graph`

At the heart of Skoro's structure is the `Graph<B>` (in `skoro-graph`).
It is an arena holding the probabilistic variables and relationships in a model.

The graph is composed of **`Node`** variants:

- **`Constant` / `Parameter`**: Leaf nodes with fixed or optimizable data.
- **`Stochastic`**: A conditional random variable associated with a specific probability distribution.
- **`Mapped`**: A deterministic functional transformation of other nodes.
- **`LogDensity`**: A mapped node representing evaluated probabilities.
- **`Observed`**: A stochastic node bound to a concrete value (conditioning the model for posterior inference).

**Compilation**: Evaluating a HashMap-based graph is slow for tight numerical loops.
To fix this, `Graph` provides a `compile_backward_from(targets)` method.
This performs a topological sort and lowers the graph into a **`CompiledSequence`** of **`CompiledStep`**s.
The sequence flattens the graph into a list of steps where HashMap IDs are replaced with fast contiguous array indices, and where each `Mapped`/`LogDensity` node lowers (via `GraphOperation`) into a type-erased `Box<dyn DynSequenceOperation>`.

---

### The `DiffOperation`

Autodiff in Skoro is divided into stateless mathematical logic and stateful execution tracking.

The **`DiffOperation`** (in `skoro-diff`) is the stateless mathematical representation of an operation.
Operations must implement `forward`, `vjp`, and `hvp`.

**The Forward Pass (`forward`)**
The `forward` pass always calculates the forward operation _eagerly_, but its behavior branches depending on the active `DiffModeState` (and returns the matching variant of the `DiffOperationForward` result enum):

- **`ForwardOnly`**: Computes only the eager forward operation without any derivative tracking.
- **`Jvp` (Jacobian-Vector Product)**: Calculates the eager forward map and simultaneously computes the first-order forward map on a tangent _frame_ of vectors, represented as a tensor of shape `[batches, dimension, frames]`.
- **`Vjp` (Vector-Jacobian Product)**: Calculates the eager forward operation and computes/memoizes a state required for the first-order cotangent pullback.
- **`Hvp` (Hessian-Vector Product)**: Calculates the eager forward operation and computes/memoizes states for _both_ the first-order cotangent pullback and the second-order sensitivity with respect to the input tangent _vector_.

**The Backward Passes (`vjp` and `hvp`)**
When the `Tape` is walked in reverse, it invokes the backward passes using the memoized states:

- **`vjp`**: Performs the first-order cotangent pullback.
- **`hvp`**: Performs the second-order backward derivative.

Crucially, both `vjp` and `hvp` operate on a _frame_ of cotangent vectors.
These cotangent frames are represented as tensors of shape `[batches, dimension, frames]`, allowing the system to efficiently propagate multiple gradient signals (or sensitivities) simultaneously across batches.

When the `Graph` is compiled, its operation nodes lower into `DynSequenceOperation`s.
A `SequenceOperation` holds a `DiffOperation` alongside its execution indices.

During an autodiff pass (like `log_density_ord1`), the program iterates sequentially through the `CompiledSequence`.
Each operation calls `DiffOperation::forward` over its input array indices, writes the result to its output index, and records its memoized state as a backward step onto a **`Tape`** (Wengert list, generic over `VjpStep` / `HvpStep`).
To backpropagate, the `Tape` is walked in reverse, calling `DiffOperation::vjp` (and `DiffOperation::hvp` for second order) to fan gradients back to the parameters.
