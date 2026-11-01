---
title: "Reference measures and chart volumes of spaces"
pages:
    - concepts/spaces
    - concepts/terms
    - reference/crates
crates: [skoro, skoro-core]
revises: [core]
---

Every continuous space of [the core](../core/) now has a reference measure, given through the
log-volume of its retraction's chart, and the core's appendix no longer leaves curved types without
one. `Space::log_volume` gives the log-volume as a composite beside the retraction: zero on the
Euclidean types, the logarithm of the moved point's coordinates on the positive orthant (both
against Lebesgue measure), and the sum of the factors' on a product. An embedded type may declare
its chart's volume with `Embedded::with_volume`: a log-volume written as an operation on Euclidean
values, against the measure its retraction normalizes (Haar measure on a group, surface measure on
a sphere, whatever the storage), and a domain, global or given by a discretization, on which the
chart is a diffeomorphism onto all of the type but a set of measure zero. A type that declares none
has no densities: the core refuses to score, propose or integrate out a quantity of it, and a
measurement of it is a likelihood factor over data. The volume is declared rather than computed
from the retraction, which would need third derivatives in a term's Hessian, and
`testing::check_volume` holds each declaration to the area formula instead. The mathematics rests
on the note [densities in tangent charts](../../math/chart-volume/), brought from the `normal`
branch, which this entry corrected and extended with the measure a retraction normalizes in its
intrinsic form and the volume of a product chart. The acceptance test's sphere declares its
stereographic chart's volume, and an exponential sphere is the tests' chart with a bounded domain.
The normal entry's tangent normal builds on this, and its branch drops its own copy of the note
once this entry is merged.
