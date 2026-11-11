---
title: "Free the constants a retirement leaves with no consumer"
pages: []
crates: []
revises: [core]
---

A retirement removes the retired quantities and every node computed from them, but a constant,
or a value folded from constants, is computed from no quantity, so it is never removed: the scales,
the anchors, and each replacement's matrix and right-hand side stay in the model after the factor
that used them is gone. In the loop of the `sliding_window` example the model held 854 nodes after
tick 9 and 4,121 after tick 79, while its quantities and terms stayed between 33 and 50. The cost is
memory rather than time, but it grows without bound, so a model that runs indefinitely is not yet
supported. This entry has a retirement also drop the fixed nodes it leaves with no consumer, and
decides whether a handle the user still holds keeps such a constant alive.
