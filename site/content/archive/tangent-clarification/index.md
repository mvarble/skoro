---
title: "Smooth maps before retractions"
pages:
    - concepts/queries
crates: []
revises: [mathematics-in-order]
amends:
    - archive/core/implementation
    - archive/core/math
    - archive/rotations/statements/first-order
---

The Mathematics section used to define a retraction by its derivative at the origin before it said
what a derivative on a manifold is. This entry reordered that foundation. The first note, now
[Smooth manifolds, smooth maps, and retractions](../../math/retraction/), recalls smooth
manifolds, smooth maps, tangent vectors and the tangent bundle from Lee. It then fixes a basis of
each tangent space by a convention that asks no regularity and may be defined only on an open
subset. That turns every differential into a matrix, and only after it does the note define a
retraction, on the tangent bundle. The inverse function theorem makes the retraction's local
inverses charts, and the chain rule shows that derivatives read through it are those the bases
define. The note ends with an index of every abuse of notation the later notes make when they treat
tangent vectors as columns, so that no later note speaks of coordinate-free objects. The third
note keeps the gradient, the chart Hessian and the passes as
[Gradients and Hessians in tangent charts](../../math/derivatives-in-charts/). The results that
compare coordinates at nearby points now say that they assume $\oplus$ is smooth.
