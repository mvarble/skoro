---
title: "Reference measures and chart volumes of spaces"
pages: []
crates: []
revises: [core]
---

[The core](../core/) fixes a reference measure for each of its own types (Lebesgue measure on the
flat and positive types, counting measure on the discrete ones, product measures on products) and
states every density, importance weight and retirement against them, but it leaves the embedded,
curved types without one. Every distribution with a density on such a type needs that measure, and
so does the normal entry's tangent normal, the law of μ ⊕ ε for a normal ε on any space whose
chart's volume is known. This entry gives every continuous space of the core a reference measure
and the log-volume of its retraction's chart, once and for all types. An embedded type may declare
the log-volume of its chart, an operation on Euclidean values with derivatives to second order, and
the chart's domain, global or the set of tangent vectors on which the chart is a diffeomorphism;
its reference measure is then the one its retraction normalizes, whose charts all have volume 1 at
their centre: Haar measure on the groups and surface measure on the sphere, whatever the storage.
The core's own types get theirs (0 on the Euclidean types, the sum of the logarithms of the point's
coordinates on the positive orthant), a product's is the sum of its factors' over the stacked
tangent, and a type that declares none has no densities, so that a distribution on it can only be
a likelihood factor over data. The log-volume is declared rather than computed from the
retraction, since the area formula would put a Jacobian of the retraction into the forward value
and third derivatives into the Hessian; the formula is instead what the tests check each
declaration against, on the core's acceptance-test sphere and its stereographic chart among
others. The mathematics starts from the note "Densities in tangent charts", written on the `normal`
branch, and the entry merges into `main` before the normal entry's build.
