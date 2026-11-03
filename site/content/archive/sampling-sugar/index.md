---
title: "Sampling a quantity from a distribution"
pages: []
crates: []
revises: [normal]
---

A model declares a quantity and scores it in two statements, `let x = quantity("x", T);` and
`normal.score(x);`, and nearly every quantity drawn from a distribution is written that way. This
entry adds `sample("x", T, distribution)`, which declares the quantity at the address, scores it by
the distribution and returns it, so that the pair reads as the one act it is. `Normal` is the only
distribution so far and its `score` is its own method, so `sample` needs a trait that a
distribution implements to be scored; `Normal` is its first implementation. The sugar adds no
mathematics: what `sample` builds is exactly what the two statements build.
