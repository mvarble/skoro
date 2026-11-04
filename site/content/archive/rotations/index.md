---
title: "Rotations and rigid motions"
pages:
    - concepts/rotations-and-motions
    - concepts/the-normal
    - guide/rotations-and-poses
    - reference/crates
crates: [skoro-spatial]
---

skoro's first curved types of its own: the rotations $SO(3)$, the rigid motions $SE(3)$ and the
extended poses $SE_2(3)$, a pose with a velocity, held as the unit quaternions and their semidirect
products with one and two copies of $\R^3$, double covers of the three groups. The new crate
`skoro-spatial` declares them as embedded types on the core's public interface, each with a global
chart, stereographic in the rotation and moving the translations in the point's own frame, and a
volume declared against Haar measure, so that an attitude, a pose or an extended pose is scored by
[the normal](../normal/), observed and simulated like any other quantity. Every usual map is a
composite of the core's operations, with no kernel and no derivative of its own: compositions and
actions, matrices and adjoints, exponentials and logarithms, the Jacobians and their inverses with
their removable singularities computed by series, interpolations and the angle between rotations,
and the constructions from axes and angles, pairs of directions, matrices and Euler angles in all
twelve conventions. The `archive` branch's matrix storage, exponential charts and fused kernels are
not kept: the normal's need for a global chart decided the storage and the chart. The mathematics
was reopened once, after the build, to treat the Euler angles at both of their singularities, the
lock and the wrap at $\pm\pi$, which it proves cannot be removed; the residual of angles on the
circle and a lock threshold set by its cost came of it, and the implementation was revised to say
what each map gives within rounding of a row where it is undefined, rather than promise non-finite
values there. No earth frame is assumed: frames are the user's to compose.
