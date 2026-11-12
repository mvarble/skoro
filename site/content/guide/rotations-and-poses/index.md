---
title: Rotations and poses
summary: Score an attitude or a pose with the normal, draw it, observe one given from outside in whatever sign, and read Euler angles off it, with the crate skoro-spatial, from its own tests and examples.
order: 2
---

The crate `skoro-spatial` gives a model its rotations, rigid motions and extended poses. This page
shows what a user writes with them: a quantity of one of these types, scored by
[the normal](../../concepts/the-normal/) and drawn, a pose with its noise in its own frame,
measured poses read from a file and observed as they come, and Euler angles read off a rotation and
compared. Each step quotes the crate's tests, so every line
shown compiles and runs. The [concepts page](../../concepts/rotations-and-motions/) explains how
the types are made and what each map computes.

## A rotation in a model

A rotation is a quantity of the marker `Rotation`, and the normal takes it like any other space:
its mean is a rotation, here a constant unit quaternion $(w, x, y, z)$, and its scale is given in
the rotation's tangent coordinates, the rotation vector in the frame of the mean, in radians.
Generating the model draws the attitude exactly, one row per draw:

```rust
{../../../../crates/skoro-spatial/tests/normal.rs:176..187}
```

`generated.values.get(q)` holds the draws as unit quaternions, four numbers to a row. A quaternion
and its negation are the same rotation, and the types know it: every density, derivative and map is
the same whichever of the two a value is stored as, so a draw may carry either sign, and a
quaternion observed from outside the model is observed as it comes. To report one sign, take
`host::canonical`, the quaternion with $w \ge 0$.

## A pose

A pose is a rotation and a position, made from host values as a pair: here `mean`, the quaternion
`[0.3, 0.5, -0.7, 0.4]`, normalized when the constant is made, and the position `[1.0, -2.0, 0.5]`.
Its tangent coordinates are the rotation vector and then the displacement, both in the pose's own
frame, so a normal on `Pose` is noise in the frame of the body, as odometry gives it. Here a pose is
moved from its mean through its chart by the tangent `q[0]`, and the normal's log-density is taken
there, with a full covariance over the six coordinates:

```rust
{../../../../crates/skoro-spatial/tests/normal.rs:148..151}
```

With the chart's log-volume added back, as here, this is the Gaussian log-density of the tangent,
which the test checks against the host's.

`*` composes poses and applies them to points, and `PoseMaps` gives a pose's rotation, position,
inverse and relative motion, among others.

## Poses measured outside the model

Pose graphs are usually given as files in the format of g2o: each vertex an initial pose, each edge
a measured motion between two vertices with its information matrix, the quaternions in whatever
sign the program that wrote them chose. The example `g2o` reads one, by a reader in its common
module that converts each edge's information to the precision of skoro's residual, and writes the
model: the first vertex held at its pose, every other a parameter, and each edge a measured pose
scored by the normal of the relative motion between its vertices. The measurements are observed as
they were read, and the vertices start where the file puts them:

```rust
{../../../../crates/skoro/examples/g2o.rs:46..84}
```

Without a file, the example writes a loop of a hundred poses that turns once, whose vertices are
composed from the odometry, so that the last one's quaternion is near $-1$ while the loop closure is
measured near the identity: the measurement and its prediction are the same rotation in opposite
signs, and the estimate converges all the same.

```sh
cargo run --release -p skoro --example g2o                # its own loop
cargo run --release -p skoro --example g2o -- garage.g2o  # any 3D pose graph g2o writes
```

A measurement given exactly a half-turn from its prediction is the one value a rotation's chart
cannot read: the estimator refuses it, naming the term, so a start is chosen away from a
half-turn from the data. The [concepts page](../../concepts/rotations-and-motions/#where-a-map-is-undefined)
says why.

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
