---
title: "Normal distributions"
pages: []
crates: []
---

Support for normal distributions in skoro. [The core](../core/) can build, query and
differentiate a model, but no distribution exists in it yet, and the normal is the first that
`REVISIT.md` lists to return from the `archive` branch; until it does, the core's tests keep a
private normal of their own. This entry works out the mathematics of the normal distribution and
brings it into the library, as a crate built on the core.
