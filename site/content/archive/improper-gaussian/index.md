---
title: "Improper Gaussian replacements"
pages: []
crates: []
revises: [normal]
---

A retirement's replacement is a likelihood factor over its blanket, and the normal writes it only
as a Gaussian of full rank, its `Stacked` replacement. When the retired terms do not determine the
whole blanket, as for a landmark seen by bearing from one retired pose, the exact replacement is an
improper Gaussian in the blanket's tangent coordinates at fixed anchors: squared rows of any number,
none included, a right-hand side, a linear term and a constant, which no normalization makes a
density. This entry gives skoro that factor and the roles it binds, so that a model can marginalize
what it learns about a quantity it cannot yet estimate, accumulate such factors, and estimate the
quantity once enough is known. Computing the factor from a linearized retirement is the work of the
`linearization` entry, which needs it.
