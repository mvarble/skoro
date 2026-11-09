---
title: "Improper Gaussian replacements"
pages:
    - concepts/the-normal
    - reference/crates
crates: [skoro, skoro-core, skoro-normal, skoro-tensor]
revises: [normal]
---

A retirement whose terms do not determine its whole blanket, as for a landmark seen by bearing from
one retired pose, now has a replacement that keeps exactly what those terms said: the improper
Gaussian factor $-\tfrac12 \lVert A \delta - b \rVert^2 - \eta\transpose \delta + c$ in the
blanket's stacked tangent at constant anchors, of any number of rows, none included, which no
normalization need make a density. `Stacked::improper` builds it in `skoro-normal`, with its
numbers shared or given row by row of a batch, as a retirement's `Replacement` or a likelihood
factor. Where its rows have full rank it is the normal's `Stacked` replacement, and factors over a
quantity accumulate as terms, which the model reads as their sum, until they determine it. It binds
its residual, and its linear part under the log-volume role, whose description now says what every
least-squares reader takes from it: the varying part of a term's remainder, not the volume of a
chart. A factor of no rows has a residual of no components, which the core could not carry: burn's
`expand` refuses a dimension of zero and its `reshape` reads one as the source's size, so the tensor
layer gained a `broadcast` and a `reshape` that give the tensor of no entries there, and every pass
of the core now runs on values of no entries, on the host and on wgpu. The sum of factors, and a
reduction of their rows, were left to a later entry.
