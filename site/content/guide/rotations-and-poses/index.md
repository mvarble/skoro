---
title: Rotations and poses
summary: Score an attitude or a pose with the normal, draw it, and read Euler angles off it, with the crate skoro-spatial, from its own tests.
order: 2
---

The crate `skoro-spatial` gives a model its rotations, rigid motions and extended poses. This page
shows what a user writes with them: a quantity of one of these types, scored by
[the normal](../../concepts/the-normal/) and drawn, a pose with its noise in its own frame, and
Euler angles read off a rotation and compared. Each step quotes the crate's tests, so every line
shown compiles and runs. The [concepts page](../../concepts/rotations-and-motions/) explains how
the types are made and what each map computes.

## A rotation in a model

A rotation is a quantity of the marker `Rotation`, and the normal takes it like any other space:
its mean is a rotation, here a constant unit quaternion $(w, x, y, z)$, and its scale is given in
the rotation's tangent coordinates, the rotation vector in the frame of the mean, in radians.
Generating the model draws the attitude exactly, one row per draw:

```rust
{../../../../crates/skoro-spatial/tests/normal.rs:175..186}
```

`generated.values.get(q)` holds the draws as unit quaternions, four numbers to a row. A quaternion
and its negation are the same rotation; a quaternion observed from outside the model is brought to
the side of the model's prediction by `host::align` before it is observed.

## A pose

A pose is a rotation and a position, made from host values as a pair: here `mean`, the quaternion
`[0.3, 0.5, -0.7, 0.4]`, normalized when the constant is made, and the position `[1.0, -2.0, 0.5]`.
Its tangent coordinates are the rotation vector and then the displacement, both in the pose's own
frame, so a normal on `Pose` is noise in the frame of the body, as odometry gives it. Here a pose is
moved from its mean through its chart by the tangent `q[0]`, and the normal's log-density is taken
there, with a full covariance over the six coordinates:

```rust
{../../../../crates/skoro-spatial/tests/normal.rs:147..150}
```

With the chart's log-volume added back, as here, this is the Gaussian log-density of the tangent,
which the test checks against the host's.

`*` composes poses and applies them to points, and `PoseMaps` gives a pose's rotation, position,
inverse and relative motion, among others.

## Euler angles

`euler` reads the angles of a convention off a rotation, and `rotation::from_euler` and
`host::from_euler` make a rotation from them. The first and last angles jump by $2\pi$ at $\pm\pi$,
so a model compares them with `angle_residual`, which takes their difference on the circle. Here the
yaw of a rotation about $z$ by just under and just over a half-turn jumps from near $\pi$ to near
$-\pi$, while its residual against a reference yaw of $3$ changes by $2 \cdot 10^{-5}$:

```rust
{../../../../crates/skoro-spatial/tests/euler.rs:336..348}
```

Near gimbal lock the separate angles are ill-conditioned, so a model's noise belongs in the
rotation's chart rather than in its Euler angles; the
[concepts page](../../concepts/rotations-and-motions/#euler-angles) says why.
