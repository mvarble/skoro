---
title: "The core"
pages: []
---

The core of skoro, built to [the charter](../charter/): the API for building models, with
quantity types (the Euclidean manifolds, products, and the discrete types), the primitive
operations and their derivatives of first and second order, composition, the compilation graph
and its compiler, the probabilistic graph, terms and bindings, the writer with its snapshots,
branches and building scopes, addresses, explicit random keys, and the layer over burn. It ends where models can be built and
differentiated but no distribution exists yet, except private ones in tests. Much of it is ported
from the `archive` branch, reshaped to the charter. When it is done, `REVISIT.md` lists every
feature of the archive not yet brought back, each to return as an entry of its own.
