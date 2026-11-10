---
title: "Linearization and elimination"
pages:
    - concepts/linearization
    - concepts/queries
    - concepts/spaces
    - concepts/terms
    - concepts/the-normal
    - guide/least-squares
    - reference/crates
    - reference/crates/skoro-core
    - reference/crates/skoro-elimination
    - reference/crates/skoro-linearize
    - reference/crates/skoro-normal
crates: [skoro-core, skoro-elimination, skoro-linearize, skoro-normal, skoro-spatial]
---

This entry brought back the two pieces every least-squares method rests on, as two crates.
`skoro-elimination` is the note on elimination in square-root form as code: factored problems over
blocks of variables, scheduled once by COLAMD and a junction tree, eliminated clique by clique by
faer's unpivoted QR on the host in double precision, and answering minimizers, integrals, draws,
marginals and the factor a replacement leaves. `skoro-linearize` linearizes a density query at a
batch of points into such problems, one per row: by each term's whitened residual, with the curved
types' log-volumes folded into the right-hand sides and the others in a linear term read off the
log-volumes themselves, or, for a free scale, by the normal's information root and a scale residual
that carries its score. It refuses by structure what it cannot linearize, seeds one tangent pass by
a colouring of the factor structure, and gives Gauss–Newton and Levenberg–Marquardt steps, marginals
that name their curved quantities, and the Laplace approximation's evidence and draws in the chart.
A retirement's replacement is now computed rather than written: the linearized integral of the
retired terms, built as the normal's improper factor, which is the Kalman filter on linear-Gaussian
terms and an iterated extended Kalman filter's step at their mode. The core gained the accessors an
algorithm in another crate needs, and the normal binds its tangent under a role of its own. The
methods themselves, an optimizer's loop, a damping schedule and a filter, are left to later entries.
