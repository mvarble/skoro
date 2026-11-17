---
title: "The uniform distribution on compact types, and broad scales of a normal on a curved type"
pages:
    [
        concepts/spaces,
        concepts/the-normal,
        concepts/rotations-and-motions,
        concepts/the-uniform,
        concepts/linearization,
        guide/rotations-and-poses,
        reference/crates,
    ]
crates: [skoro, skoro-core, skoro-linearize, skoro-normal, skoro-spatial, skoro-uniform]
revises: [normal]
---

A broad normal on a rotation is not a weak prior. Under the Cayley chart a tangent of length $r$ is
a turn by $2 \arctan(r/2)$, and once the standard deviation of [the normal](../normal/) passes one
radian its mean stops being a mode and its mass goes to the half-turns from the mean, where the
chart is cut: the median angle between a draw and the mean is 1.31 rad at $\sigma = 1$, 2.88 rad at
$\sigma = 10$ and 3.12 rad at $\sigma = 100$. This entry brought the uniform distribution, which is
how "unknown" is said. A type of finite mass now declares its log-mass and a uniform draw in the
core, beside its volume, and a categorical space has its mass without declaring it; the rotations
declare $8\pi^2$ and Shoemake's draw, and the test spheres $4\pi$ and Archimedes'. A new crate,
`skoro-uniform`, scores a quantity of such a space with the constant $-\log \nu(\manifold)$, draws
it exactly on the host, and binds a residual of no components, so that a linearization reads it as
a factor with no rows and needed no change. The normal was left as it was, exact at every scale:
what a large scale does on a curved type is documented, with the table of median angles drawn from
its own sampler, under the Cayley chart that [the quaternions' signs](../quaternion-signs/) put in
place of the stereographic one the core's evaluation measured.
