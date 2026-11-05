---
title: "Linearization and elimination"
pages: []
crates: []
---

This entry brings back two pieces that many later methods share. Linearization replaces a model
with its first-order approximation at a point. Elimination then solves the linear problem this
gives, or removes some of its variables from it. Gauss–Newton and Levenberg–Marquardt both solve
such a linear system at every step. A Laplace approximation reads a normal approximation off
the system at the mode. An extended Kalman filter linearizes each step and eliminates the state
it leaves behind. The entry works out the mathematics of both pieces and builds them. The methods
themselves are left to later entries, which will use these two pieces. The `archive` branch's
`skoro-linearize` crate and the elimination in its `skoro-dense` crate are prior art for this
entry, and do not bind it.
