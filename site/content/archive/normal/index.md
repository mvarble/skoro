---
title: "Normal distributions"
pages:
    - concepts/the-normal
    - guide/first-model
    - reference/crates
crates: [skoro, skoro-normal]
---

skoro's first distribution, the normal, returns as one distribution for every space whose chart is
global and whose volume is known: the law of the mean moved by normal noise through its chart, so
that a measurement of a rotation, a direction or a pose is held as what was measured, observed in
one query and simulated in another. Its term is the Gaussian log-density of the tangent at the mean
less the chart's log-volume, which makes it an exact density against the type's reference measure
from [the core's measures](../measures/); on the real line and vectors it is the familiar normal,
and on the positive orthant the log-normal. Its sampler draws exactly by the space's own retraction;
its scale is given in four forms; it binds the roles a least-squares or Fisher-scoring reader needs,
the root of its expected information among them, exact on the flat spaces and stated with its error
on curved ones; and it refuses, with the core's own errors, a chart that is not global, where no
closed form keeps density and sampler exact. A measurement on a type without a volume or a global
chart is a likelihood factor over data, and the stacked tangent gives a retirement's Gaussian
replacement its form. The new crate `skoro-normal` holds it, tested on the core's two test spheres.
