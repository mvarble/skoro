---
title: "Observed quaternions of arbitrary sign on outside data"
pages:
    - concepts/spaces
    - concepts/linearization
    - concepts/rotations-and-motions
    - guide/rotations-and-poses
    - reference/crates
crates: [skoro, skoro-core, skoro-linearize, skoro-spatial]
revises: [rotations, core, measures, linearization]
---

The rotations, the rigid motions and the extended poses were declared as the unit quaternions and
their products, with the stereographic chart, which tells $q$ from $-q$: an observed quaternion had
to be given the sign of its prediction by its caller, and one given with the other sign was read as
nearly a full turn away, and could wind a turn into a pose graph's estimate without notice. This
entry makes them the groups themselves, $SO(3)$, $SE(3)$ and $SE_2(3)$, still stored as quaternions
but up to their sign, under the Cayley chart, global on each group and blind to the sign, so that
the normal on them is the law of the rotation and a measurement is observed in whatever sign it
comes. The core learned to declare a type stored up to a finite group of signed permutations of its
storage, and to check it, with none of its derivatives changed; every map of the rotations became a
map of the group, the logarithm, its axis and angle and the interpolations those of the shorter
turn; and the linearizer refuses a point at which a factor is not finite, which is how a value a
half-turn from its mean, the Cayley chart's cut, is reported. The mathematics first worked out a
sign-free mixture of two normals on the quaternions, and turned to the chart instead when the
mixture would have needed a second law and a correction on every term. A reader of g2o's pose
graphs, an example estimating one in arbitrary signs, and tests on a loop whose closure starts on
the far side of its prediction show it on outside data.
