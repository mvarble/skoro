`skoro-core` is everything that makes a model a model. It holds the [spaces](../../../concepts/spaces/)
quantities live in (the Euclidean and positive types, products, the discrete types, and the embedded
types that later crates declare from a retraction and its inverse) and the typed handles code holds
them by. It holds [operations](../../../concepts/operations/): the trait every operation implements,
the kernels they are lowered to with derivatives up to second order, the primitives, composites, and
the registry that makes an operation again from its name and parameters.

It holds the model itself: the writer that [builds](../../../concepts/building/) it in scopes and
publishes snapshots, branches and readers, addresses and namespaces, and the
[terms](../../../concepts/terms/) that make the probabilistic graph, with their samplers, proposals
and roles. It compiles [queries](../../../concepts/queries/) for a target into programs, whose passes
give values, tangents, gradients and Hessian-vector products in tangent charts, and it holds
generation, which draws a model's free quantities from explicit random keys and weighs the draw, and
retirement, which removes the quantities at a collection of addresses, named one by one or as
the namespaces they lie under, and keeps what they said about the rest as a likelihood factor over
their Markov blanket, which the caller builds.

Its `testing` module is public, so that a later crate checks its own operations and embedded types
the way the core checks its primitives: against finite differences, at both precisions, with the
`precision_tests!` macro.
