---
title: Rotations and rigid motions
summary: The crate skoro-spatial declares the rotations, the rigid motions and the extended poses as embedded types with global charts and declared volumes, so that they carry the normal, and writes their maps as composites whose derivatives are the core's.
order: 7
depends_on: [concepts/spaces, concepts/the-normal, retraction, chart-volume]
---

Attitudes, poses and the states of a navigation filter live on curved spaces: the rotations
$SO(3)$, the rigid motions $SE(3)$, and the extended poses $SE_2(3)$, a pose with a velocity. The
crate `skoro-spatial` declares each as an [embedded type](../spaces/#measures-and-chart-volumes) on
the core's public interface, with a chart that is global and a volume that is declared, so that a
quantity of any of them can be scored by [the normal](../the-normal/), observed and simulated like
any other. Everything a model computes with them, compositions, actions on vectors, exponentials,
Jacobians, Euler angles, is a composite of the core's operations, and its derivatives are the
core's. This page shows how the types are made, the maps on them, and where those maps stop being
smooth.

## The three types

A rotation is stored as a unit quaternion $q = (w, x, y, z)$, a rigid motion as $(q, p)$ with a
position $p$, and an extended pose as $(q, v, p)$ with a velocity before the position: storage $4$,
$7$ and $10$, dimension $3$, $6$ and $9$. The tangent coordinates are the rotation's $\phi$ first,
then one vector for each translation, all in the frame of the point, so that noise on a pose is
noise in its own frame, as odometry has it.

The rotation's chart is the stereographic projection of the unit quaternions from the antipode of
the base point,

$$
	q \oplus \tangent = q \otimes \frac{\bigl(1 - s,\ \tangent / 2\bigr)}{1 + s}, \qquad s = \tfrac1{16}
	\lVert \tangent \rVert^2,
$$

rational, free of trigonometry, and a diffeomorphism of $\R^3$ onto every unit quaternion but $-q$.
To first order $\tangent$ is the rotation vector, and the angle of the turn $1 \oplus \tangent$ is
$4 \arctan(\lVert \tangent \rVert / 4)$, so the chart reaches every rotation, the half-turns at
$\lVert \tangent \rVert = 4$. Its log-volume, against Haar measure, is $\lambda(\tangent) = -3
\log(1 + \lVert \tangent \rVert^2 / 16)$. A motion moves its rotation by this chart and its
translations in its own frame, $(q, p) \oplus (\phi, \rho) = (q \oplus \phi,\ p + R \rho)$ with $R$
the rotation's matrix, which is global too, with the same volume. One generic declaration makes all
three, for $k$ translations:

```rust
{../../../../crates/skoro-spatial/src/types.rs:92..126}
```

The retraction normalizes its base point. On the type this changes nothing; off it, it keeps a
point that a long run of retractions has carried off the sphere by rounding from passing that error
on to every later step. The markers name the three declarations:

```rust
{../../../../crates/skoro-spatial/src/types.rs:134..152}
```

> [!DECISION] Quaternions, and a rational chart
> A rotation could be stored as its matrix, with the exponential chart. The normal requires a
> chart that is global, and the exponential chart is a diffeomorphism only on the ball of radius
> $\pi$; the stereographic chart is global, its retraction, inverse and volume are rational, and a
> quaternion is four numbers rather than nine, with one constraint rather than six. See
> [the mathematics of the rotations](../../archive/rotations/math.svx#matrices-or-quaternions).

$q$ and $-q$ are the same rotation, and the types are their **double cover**: a quantity of
`Rotation` is a unit quaternion, and two quaternions a model compares must be on the same side.
A constant made from a host value is normalized there, and refused when it is zero or not finite;
an observed quaternion, from a sensor or a file, is aligned with the model's prediction by
`host::align` before it is observed, and a quaternion the model computes by `aligned_with`. The
normal on $S^3$ is a law on the rotations as well: [the mathematics](../../archive/rotations/math.svx#the-double-cover)
shows that the rotation it gives has the density $\density(q) + \density(-q)$.

## Computing with them

`*` composes, and acts on vectors, as the mathematics writes it:

```rust
{../../../../crates/skoro-spatial/src/rotation.rs:257..291}
```

The other maps of a rotation are methods of its handle, through an extension trait, since the core
owns `Q`:

```rust
{../../../../crates/skoro-spatial/src/rotation.rs:293..338}
```

and the rotations are made by free functions in `rotation`, from a rotation vector, an axis and an
angle, a pair of directions, Euler angles or a matrix:

```rust
{../../../../crates/skoro-spatial/src/rotation.rs:415..464}
```

`so3` holds the Euclidean maps of rotation vectors: `hat` and `vee`, and the left and right
Jacobians of the exponential and their inverses. A rigid motion composes, acts on points, and has
its parts, its homogeneous matrix and its adjoint, its exponential and logarithm in body twists,
and interpolations along the screw or along each part:

```rust
{../../../../crates/skoro-spatial/src/pose.rs:358..420}
```

with `pose::identity`, `from_parts`, `from_product`, `exp`, and the Jacobians `jl`, `jr`, `jl_inv`
and `jr_inv` in the order $(\phi, \rho)$. The extended poses have the same maps through
`ExtendedPoseMaps` and `extended_pose`, with the velocity carried as the position is. `as_product`
reads a pose as a value of the core's product $S^3 \times \R^3$, whose chart moves the position in
a fixed frame: a pose whose noise is in the world's frame, a GPS fix for one, is a quantity of that
product instead.

No frame is assumed: an earth frame, a gravity or a sensor's mounting is a rotation or a motion
the model composes, or a constant it holds.

## Derivatives

No map writes a derivative. Each is a composite of the core's primitives, so its derivatives of
both orders are those of its parts, and in the charts they are the Jacobians of the textbooks: the
tangent pass of `rotation::exp` at $\omega$, read in the chart at its value, is `so3::jr(ω)`, and
likewise for the motions, as [the mathematics](../../archive/rotations/math.svx#maps-of-the-rotations)
proves and the crate's tests check. The second derivatives of a map of a rotation agree with those
taken in exponential coordinates; on a pose they differ by a term the mathematics names, since the
pose's chart is not its exponential.

The exponential, the logarithm and the Jacobians are smooth at the identity, but their closed forms
divide by the angle there. Each coefficient is computed by a truncated series below a cutoff and by
its closed form above it, selected by the core's selection, with each formula given a harmless
argument on the rows where the other is chosen, so that every pass is finite at the identity too.
The cutoffs are one for both precisions, since a composite does not know the precision it will run
at.

## Euler angles

`Euler` names a convention: one of the twelve sequences of axes and whether the turns are about the
axes as the previous turns left them (intrinsic) or about the fixed axes (extrinsic).
`Euler::YAW_PITCH_ROLL` is the aerospace convention, intrinsic about $(z, y, x)$.
`rotation::from_euler` is smooth everywhere. The way back, `euler`, is not, and no formula could
make it so: [the mathematics](../../archive/rotations/math.svx#constructions-and-conversions)
proves that it has two singularities.

**The wrap.** The first and last angles are returned in $(-\pi, \pi]$, and jump by $2\pi$ where
they cross $\pm\pi$. Any real-valued angle must be cut somewhere; the jump is in the value alone,
and the derivatives of every order are the same on both sides. What breaks is a difference of
angles, which is taken on the circle instead:

```rust
{../../../../crates/skoro-spatial/src/euler.rs:319..328}
```

**Gimbal lock.** Where the middle angle is $0$ or $\pi$ for a proper sequence, or $\pm\pi/2$ for a
Tait–Bryan one, only the sum or the difference of the other two is determined. The composite
returns the last angle as $0$ there, where one of two norms vanishes to within $10^{-34}$ of the
quaternion's, which changes the rotation the angles give back by less than double precision's
rounding. Near the lock the first and last angles alone are ill-conditioned, their derivatives
growing like $1 / \sin(\beta / 2)$, while the middle angle and the determined combination stay
accurate.

Euler angles are therefore a way into and out of a model, from a configuration file or to a
display, and a model's noise is put in the charts.

## Where a map is undefined

Some maps are undefined at a point the mathematics names: the inverse retraction at $-q$, the
logarithm at $-1$, the inverses of the Jacobians at an angle of $2\pi$, `from_two_vectors` at
opposite directions. At a row exactly there, the composite's values or passes are non-finite. At a
row that rounding leaves just off it, they are finite: large, where the map grows without bound,
and where only its conditioning does, as in the direction of the rotation carrying a vector to its
opposite, meaningless. Each map's documentation says which.

> [!DECISION] Singular rows are undefined, not refused
> A guard could make every row within some distance of a singular one non-finite. A composite does
> not know its precision, so one distance either refuses rows that double precision computes well
> or lets single precision's meaningless rows through; the documentation tells a model what to
> keep away from instead. See
> [the implementation](../../archive/rotations/implementation.md#errors-and-domains).

## On the host

`host` does in double precision what a model does before its values are loaded: aligning an
observed quaternion with a reference, composing, converting between quaternions, matrices, Euler
angles and rotation vectors, for constants and observations given in those forms.
