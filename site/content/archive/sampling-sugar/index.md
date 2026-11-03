---
title: "Sampling a quantity from a distribution"
pages:
    - concepts/building
    - concepts/terms
    - concepts/the-normal
    - guide/first-model
    - reference/crates
crates: [skoro-core, skoro-normal]
revises: [normal]
---

`sample("x", T, distribution)` makes a quantity at an address, scores it by a distribution and
returns it, in place of `quantity("x", T)` followed by the distribution's `score`, and `sample_in`
does the same for a space given as a value. The core gained the `Distribution` trait for it, whose
one method adds a distribution's term at a quantity, with an implementation for references so that
a distribution is passed by value or borrowed; `Normal` implements it by its own `score`, which
stays, so that a caller scoring by a normal it names imports nothing. A refusal by either step is
recorded at the line that called `sample`. It returns the quantity and not the term: the term is
found from the quantity, which is scored at most once, and a model that needs the term's handle
writes the two statements. The sugar adds no mathematics: `sample` builds exactly what those two
statements build.
