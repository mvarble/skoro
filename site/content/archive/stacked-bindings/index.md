---
title: "Bindings of the stacked tangent's replacement"
pages: []
crates: []
revises: [normal]
---

The normal's `Stacked` writes a retirement's Gaussian replacement, the normal on $\R^D$ at the
stacked tangent of the blanket's quantities at constant anchors. Added as a factor with
`Stacked::factor`, it binds its whitened residual, its mean and its scale, as the normal's roles
ask; written for a retirement with `Stacked::log_density`, it gives the log-density alone, and
binds nothing. Since [the core's replacement bindings](../replacement-bindings/), a
`Replacement` carries the roles its factor binds, checked and bound by the retirement in the
snapshot it publishes. This chore has `Stacked` supply those bindings for the replacements it
writes, so that a replacement built with it binds the same roles as the factor `Stacked::factor`
adds, and a least-squares or Fisher-scoring reader finds the residual of a retired window as it
finds any other. It needs no new mathematics.
