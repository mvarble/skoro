---
title: "Bindings on a retirement's replacement"
pages:
    - concepts/the-normal
    - reference/crates
crates: [skoro, skoro-core, skoro-normal]
revises: [core]
---

A retirement's replacement can now bind roles. `Replacement` carries the roles its factor binds,
added with `Replacement::bind` as `TermHandle::bind` adds them on a term in a scope, and
`Retirement::replace` checks each node, refusing one reached from a quantity outside the factor's
with `BindingOutsideTerm` before anything is removed, binds them to the factor it adds in the one
snapshot it publishes, and gives back that factor's handle, or none for a retirement of nothing.
Binding afterwards in a scope of its own was set aside, since it would publish a replacement with no
roles before the one that binds them. The normal's `Stacked`, which writes a retirement's Gaussian
replacement, now says that its caller binds the replacement's residual, mean and scale this way; its
supplying those bindings itself is the work this opens.
