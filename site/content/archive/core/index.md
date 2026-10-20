---
title: "The core"
pages:
    - start/how-skoro-is-developed
    - guide/first-model
    - concepts/spaces
    - concepts/operations
    - concepts/building
    - concepts/terms
    - concepts/queries
    - reference/crates
crates: [skoro, skoro-core, skoro-tensor]
---

The core of skoro, built to [the charter](../charter/): three crates in which a model can be built,
queried and differentiated, but no distribution exists yet. `skoro-tensor` is the layer over burn,
whose tensors are made at an explicit precision on the Flex device or a wgpu device. `skoro-core`
holds the spaces of quantities (the Euclidean and positive types, products, the discrete types, and
the embedded types later crates declare), the primitive operations with kernels of up to second
order, composites, the writer with its snapshots, branches and building scopes, addresses, terms
and the probabilistic graph, explicit random keys, queries and their compiler, the passes of a
program in tangent charts, generation weighed by importance, and retirement under a namespace.
`skoro` re-exports both. Much of it was ported from the `archive` branch and reshaped to the
charter, and `REVISIT.md` lists what the archive had that the core does not bring back, each to
return as an entry of its own. Along the way the workflow gained the reading orders of the crates,
which the end of a build reports and an entry links to.
