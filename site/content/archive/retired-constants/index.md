---
title: "Free the constants a retirement leaves with no consumer"
pages:
    - concepts/building
    - concepts/linearization
    - concepts/spaces
    - concepts/terms
    - concepts/the-normal
    - guide/sliding-window
    - reference/crates
    - start/what-is-skoro
crates: [skoro, skoro-core]
revises: [core]
---

A retirement removed the retired quantities and every node computed from them, but never a
constant, or a value folded from constants, since those are computed from no quantity: the scales,
the anchors, and each replacement's matrix and right-hand side stayed in the model after the factor
that used them was gone, and the `sliding_window` example's model grew from 854 nodes after tick 9
to 4,121 after tick 79. This entry has a retirement also free every node it leaves needed by
nothing, one that is no quantity, has no consumer, and is held by no term, binding, sampler,
proposal or pin, by reference counting from what the change touched. [The mathematics](math.svx)
shows that this removes nothing a term, sampler or proposal reads and changes no value of the
model, and that a model's size is then bounded by what its current quantities, terms and pins need.
It decided that a handle the user holds does not keep its node, and added `scope::pin` and
`scope::unpin` for the nodes that must be kept, such as a constant made once for a whole run. The
sliding window now holds at most 680 nodes, and checks that the count stopped growing.
