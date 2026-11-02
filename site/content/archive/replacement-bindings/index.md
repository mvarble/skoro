---
title: "Bindings on a retirement's replacement"
pages: []
crates: []
revises: [core]
---

`Writer::retire` gives the caller a plan, and `Retirement::replace` adds the caller's replacement
as a likelihood factor over the blanket itself, giving back no handle on the term it made. Whatever
the caller builds for the replacement therefore stays out of reach once the core has added it: the
term cannot bind roles, so a least-squares or Fisher-scoring reader finds no residual on a
replacement, even when it is a Gaussian whose residual, mean and scale the caller built. The normal
entry documented this gap on `Stacked::log_density`, which writes a retirement's Gaussian
replacement, and left it for the core to close. This chore makes what the core constructs from a
replacement transparent to the caller, so that a replacement's term can carry bindings as a factor
added with `scope::factor` can, with no new mathematics. Binding the normal's residual, mean and
scale on its replacements is the work this opens, and follows it.
