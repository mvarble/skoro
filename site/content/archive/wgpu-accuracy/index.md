---
title: "Accuracy on the GPU"
pages:
    - concepts/queries
    - concepts/rotations-and-motions
    - guide/first-model
    - reference/crates
crates: [skoro, skoro-core, skoro-spatial]
revises: [rotations, measures]
---

Three checks of single against double precision passed on the CPU and failed on a wgpu adapter,
where CI cannot run them: two of the pose's left Jacobian in the rotations' tests, and one of the
exponential test sphere's log-volume in the core's. The cause was the GPU's sine and cosine, which
are accurate only to an absolute error of about $4 \cdot 10^{-7}$. Near the identity, a closed form
whose numerator cancels to a power of the angle divides that error by the power, and its derivatives
by higher ones. The [mathematics](math.md) measures the adapter's elementary functions, derives how
the error reaches each closed form and its derivatives, and designs the cutoffs for a stated error
of $2^{-20}$, which the tests check the adapter against before they run. The rotations' cutoff moved
from $\theta^2 = 0.1$ to $1$, with series long enough for double precision there. The test sphere's
exponential and log-volume moved to $r^2 = 0.1$. Every test passes on the adapter, with no tolerance
changed.
