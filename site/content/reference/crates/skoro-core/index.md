`skoro-core` is everything that makes a model a model. It holds the [spaces](../../../concepts/spaces/)
quantities live in (the Euclidean and positive types, products, the discrete types, and the embedded
types that later crates declare from a retraction and its inverse) and the typed handles code holds
them by, and gives every continuous space a reference measure through the log-volume of its chart:
Lebesgue measure on its own types, and on an embedded type that declares its chart's volume, the
measure its retraction normalizes. A space without one has no densities, and the core refuses to
score, propose or integrate out a quantity of it. It holds [operations](../../../concepts/operations/): the trait every operation implements,
the kernels they are lowered to with derivatives up to second order, the primitives, composites, and
the registry that makes an operation again from its name and parameters.

It holds the model itself: the writer that [builds](../../../concepts/building/) it in scopes and
publishes snapshots, branches and readers, addresses and namespaces, and the
[terms](../../../concepts/terms/) that make the probabilistic graph, with their samplers, proposals
and roles; the `Distribution` trait, by which a distribution scores a quantity; and `sample`, which
makes a quantity and scores it by a distribution in one call. It compiles [queries](../../../concepts/queries/) for a target into programs, whose passes
give values, tangents, gradients and Hessian-vector products in tangent charts, for values of any
number of entries, none included, as a `Vector(0)` or a `Matrix(0, n)`, and it holds
generation, which draws a model's free quantities from explicit random keys and weighs the draw, and
retirement, which removes the quantities at a collection of addresses, named one by one or as
the namespaces they lie under, and keeps what they said about the rest as a likelihood factor over
their Markov blanket, which the caller builds with the roles it binds, and whose handle the
retirement gives back.

An algorithm in another crate reads what it needs through the same interface: a query's terms and
observations, and the union of two sets of observations; the quantities any node is reached from;
whether a composite reads an input, and so whether a space's chart volume depends on its point; and
a constant of a space known only at run time. [The linearization](../../../concepts/linearization/)
is built on these alone.

Its `testing` module is public, so that a later crate checks its own operations and embedded types
the way the core checks its primitives: against finite differences, at both precisions, with the
`precision_tests!` macro, and a declared chart volume against the area formula. Its wgpu target
is admitted before any test uses it: `Admission` measures the adapter's sine and cosine against
double precision, and refuses an adapter whose absolute error exceeds what the removable
singularities' cutoffs are designed for, as [Accuracy on the
GPU](../../../archive/wgpu-accuracy/) works out.
