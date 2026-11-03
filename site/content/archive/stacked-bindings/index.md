---
title: "Bindings of the stacked tangent's replacement"
pages:
    - concepts/the-normal
    - reference/crates
crates: [skoro, skoro-normal]
revises: [normal]
---

`Stacked` now gives a retirement its Gaussian replacement whole. `Stacked::replacement` returns a
`Replacement` holding $\ell_{\mathcal N}(\delta; m, W) + c$, no proposals, and the whitened
residual, the mean and the scale bound under the normal's roles, which the retirement checks and
binds to the factor it adds, as [the core's replacement bindings](../replacement-bindings/) allow.
`Stacked::factor` and `Stacked::replacement` bind through one helper, so that a replacement binds
exactly the roles the same factor added in a scope binds, and a least-squares or Fisher-scoring
reader finds the residual of a retired window as it finds any other. `Stacked::log_density` stays,
for a caller that wants the log-density alone; changing it to return a `Replacement` was set aside,
since it would break its callers and its name would no longer say what it gives.
