---
title: "Rotations and rigid motions"
pages: []
crates: []
---

Support for the rotations $SO(3)$ and the rigid motions $SE(3)$ and $SE_2(3)$ as curved types of
skoro, declared as embedded types on the core's public interface, so that attitudes, poses and
extended poses (a pose with a velocity) can be estimated states and observables, each carrying the
normal. The only curved type skoro has so far is the sphere of the core's tests. The `archive`
branch had `SO(3)` and `SE(3)` with exponential charts and fused kernels for their exponential and
logarithm; since then [the normal](../normal/) has come to require a global chart, and
[the core's measures](../measures/) a declared volume, which point the rotations at the Cayley
chart, rational and free of trigonometry, whose retraction, inverse and log-volume can be
composites. This entry works out each type's storage, its chart and the side it perturbs on, its
reference measure and declared volume, a global chart for the rigid motions held as groups, and how
the three are tested, with no earth frame assumed: frames are the user's to compose.
