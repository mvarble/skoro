---
title: "The uniform distribution on compact types, and broad scales of a normal on a curved type"
pages: []
crates: []
revises: [normal]
---

A broad normal on a rotation is not a weak prior. Under the Cayley chart a tangent of length $r$ is
a turn by $2 \arctan(r/2)$, and the density of [the normal](../normal/) on the rotations against
Haar measure is the Gaussian times $(1 + r^2/4)^2$, so its mean stops being a mode once its standard
deviation passes one radian, and the mass then goes to the half-turns from the mean: computed from
the chart, the median angle between a drawn rotation and the mean is 1.31 rad at $\sigma = 1$, 2.32
rad at $\sigma = 3$, 2.88 rad at $\sigma = 10$ and 3.14 rad at $\sigma = 1000$. A very large
standard deviation, the common way to write "unknown", therefore gives a prior concentrated where
the chart is cut and a term is not finite, and during optimization such a term pushes the estimate
away from its mean. This entry sets out to bring the uniform distribution on the compact types,
which is how "unknown" is said, and to decide whether a constant scale above about one radian on a
curved type is refused or documented. The evaluation of the core measured the effect under the
stereographic chart that [the quaternions' signs](../quaternion-signs/) replaced, and the entry
draws those measurements again under the Cayley chart.
