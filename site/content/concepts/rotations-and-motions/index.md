---
title: Rotations and rigid motions
summary: The crate skoro-spatial declares the rotations, the rigid motions and the extended poses as embedded types stored as quaternions up to sign, with the Cayley chart, global on each group, and declared volumes, so that they carry the normal in whatever sign their quaternions are given, and writes their maps as composites whose derivatives are the core's.
order: 7
depends_on:
    [
        concepts/spaces,
        concepts/the-normal,
        rotations,
        rigid-motions,
        removable-singularities,
        chart-volume,
    ]
---

Attitudes, poses and the states of a navigation filter live on curved spaces: the rotations
$SO(3)$, the rigid motions $SE(3)$, and the extended poses $SE_2(3)$, a pose with a velocity. The
crate `skoro-spatial` declares each as an [embedded type](../spaces/#measures-and-chart-volumes) on
the core's public interface, stored as unit quaternions up to their sign, with a chart that is
global on the group and a volume that is declared, so that a quantity of any of them can be scored
by [the normal](../the-normal/), observed in whatever sign it is given, and simulated like any
other. Everything a model computes with them, compositions, actions on vectors, exponentials,
Jacobians, Euler angles, is a composite of the core's operations, and its derivatives are the
core's. This page shows how the types are made, the maps on them, and where those maps stop being
smooth. The mathematics is in three notes: [rotations and unit quaternions](../../math/rotations/),
[rigid motions](../../math/rigid-motions/), and
[removable singularities in floating point](../../math/removable-singularities/).

## The three types

A rotation is stored as a unit quaternion $q = (w, x, y, z)$, a rigid motion as $(q, p)$ with a
position $p$, and an extended pose as $(q, v, p)$ with a velocity before the position: storage $4$,
$7$ and $10$, dimension $3$, $6$ and $9$. The tangent coordinates are the rotation's $\phi$ first,
then one vector for each translation, all in the frame of the point, so that noise on a pose is
noise in its own frame, as odometry has it.

A quaternion and its negation are the same rotation, and the types are the groups themselves,
stored up to that sign: each declares that it is [stored up to a group](../spaces/#types-stored-up-to-a-group),
here the one that negates the quaternion and keeps the translations, so that $q$ and $-q$ store one
point. The rotation's chart is the Cayley chart, the first-order quaternion update, normalized,

$$
	q \oplus \tangent = q \otimes \frac{(1,\ \tangent / 2)}{\sqrt{1 + \lvert \tangent \rvert^2 / 4}},
	\qquad
	p \ominus q = \frac{2\, w\, \operatorname{vec}(r)}{w^2}, \quad r = \bar q \otimes p,\ w = w(r),
$$

rational but for one square root, and the same for $q$ and $-q$: negating either quaternion negates
$r$, whose vector part and scalar part change sign together. To first order $\tangent$ is the
rotation vector, and the angle of the turn $1 \oplus \tangent$ is $2 \arctan(\lvert \tangent \rvert
/ 2)$, so the chart reaches every rotation by less than a half-turn from its base point, which is
all of $SO(3)$ but a set of measure zero. Its log-volume, against Haar measure, is $\lambda(\tangent)
= -2 \log(1 + \lvert \tangent \rvert^2 / 4)$. The inverse is written over $w^2$ so that the only
reciprocal it takes is of a positive number, the one kind the core's primitives take. A motion
moves its rotation by this chart and its translations in its own frame, $(q, p) \oplus (\phi,
\rho) = (q \oplus \phi,\ p + R \rho)$ with $R$ the rotation's matrix, which is global too, with the
same volume. One generic declaration makes all three, for $k$ translations:

```rust
{../../../../crates/skoro-spatial/src/types.rs:96..138}
```

The retraction normalizes its base point. On the type this changes nothing; off it, it keeps a
point that a long run of retractions has carried off the sphere by rounding from passing that error
on to every later step. The markers name the three declarations:

```rust
{../../../../crates/skoro-spatial/src/types.rs:148..168}
```

> [!DECISION] Quaternions up to sign, and the Cayley chart
> A rotation could be stored as its matrix, with no sign to choose, but the projection onto the
> matrices has no composite smooth at the rotations themselves, while the quaternions' projection
> is a normalization. Stored as a quaternion, a rotation has two names, and a chart of the
> quaternions tells them apart: under the stereographic chart this crate used first, a measurement
> given with the other sign than its prediction was read as nearly a full turn away, and could wind
> a turn into an estimate unnoticed. The Cayley chart is a chart of the rotations, not of the
> quaternions, global on $SO(3)$ and blind to the sign. See
> [the mathematics of the rotations](../../archive/rotations/math.svx#matrices-or-quaternions) and
> [of observed quaternions](../../archive/quaternion-signs/math.svx#a-chart-that-cannot-tell-the-signs-apart).

So a quantity of `Rotation` is a rotation, and its value is stored as either of its quaternions.
A constant made from a host value is normalized there, and refused when it is zero or not finite,
its sign kept; an observed quaternion, from a sensor or a file, is observed in whatever sign it
comes in; and every term, every derivative and every map below gives the same at either. Which
quaternion a value carries is decided by how it was made: a retraction keeps the side of its base
point, so a draw of the normal carries its mean's sign, a composition carries the product's, and an
observation the sign it was given.

## Computing with them

`*` composes, and acts on vectors, as the mathematics writes it:

```rust
{../../../../crates/skoro-spatial/src/rotation.rs:268..302}
```

The other maps of a rotation are methods of its handle, through an extension trait, since the core
owns `Q`:

```rust
{../../../../crates/skoro-spatial/src/rotation.rs:304..349}
```

Every one of them is a map of the rotations, not of the quaternions that store them: a value out of
the type is the same whichever sign its inputs are stored in, and a rotation into it is the same
rotation. The logarithm is the rotation vector of the shorter turn, of length at most $\pi$, its
axis and angle are that vector's, and the interpolation goes along the shorter turn; each of these
three is a discretization at the half-turns, where the two turns are equally short. `aligned_with`
chooses which quaternion stores a rotation, and is the identity on the rotations.

and the rotations are made by free functions in `rotation`, from a rotation vector, an axis and an
angle, a pair of directions, Euler angles or a matrix:

```rust
{../../../../crates/skoro-spatial/src/rotation.rs:414..464}
```

`so3` holds the Euclidean maps of rotation vectors: `hat` and `vee`, and the left and right
Jacobians of the exponential and their inverses. A rigid motion composes, acts on points, and has
its parts, its homogeneous matrix and its adjoint, its exponential and logarithm in body twists,
and interpolations along the screw or along each part, all of the shorter turn:

```rust
{../../../../crates/skoro-spatial/src/pose.rs:342..399}
```

with `pose::identity`, `from_parts`, `from_product`, `exp`, and the Jacobians `jl`, `jr`, `jl_inv`
and `jr_inv` in the order $(\phi, \rho)$. The extended poses have the same maps through
`ExtendedPoseMaps` and `extended_pose`, with the velocity carried as the position is. `as_product`
reads a pose as a value of the core's product $S^3 \times \R^3$, whose chart moves the position in
a fixed frame: a pose whose noise is in the world's frame, a GPS fix for one, is a quantity of that
product instead.

No frame is assumed: an earth frame, a gravity or a sensor's mounting is a rotation or a motion
the model composes, or a constant it holds.

Some of these maps are needed before a model runs, or after it, on values that are not in it: a
start for an estimator computed from the measurements, or a quaternion to report. `host` holds
them, in double precision on arrays: the alignment and `canonical`, the representative with
$w \ge 0$ to report, the composition of quaternions, the matrix, the exponential and logarithm, the
Euler angles, and a motion's composition and action, each computing what the graph's map computes:

```rust
{../../../../crates/skoro-spatial/src/host.rs:32..44}
```

## Derivatives

No map writes a derivative. Each is a composite of the core's primitives, so its derivatives of
both orders are those of its parts, and in the charts they are the Jacobians of the textbooks: the
tangent pass of `rotation::exp` at $\omega$, read in the chart at its value, is `so3::jr(ω)`, and
likewise for the motions, as [the note](../../math/rotations/#agreement-with-the-exponential)
proves and the crate's tests check. The second derivatives of a map of a rotation agree with those
taken in exponential coordinates; on a pose they differ by
[a term of second order](../../math/rigid-motions/#the-exponential-and-the-logarithm), since the
pose's chart is not its exponential.

The exponential, the logarithm and the Jacobians are smooth at the identity, but their closed forms
divide by the angle there. Each coefficient is computed by a truncated series below a cutoff and by
its closed form above it, selected by the core's selection, with each formula given a harmless
argument on the rows where the other is chosen, so that every pass is finite at the identity too.
The cutoff is one for both precisions and for every device, since a composite does not know where
it will run, and [the note](../../math/removable-singularities/#where-to-put-the-cutoff) gives the
two conditions that place it:

```rust
{../../../../crates/skoro-spatial/src/series.rs:16..19}
```

At $\theta^2 = 1$, the closed forms above the cutoff agree with double precision in single
precision on a GPU, whose sine is accurate only to an absolute error (see
[queries](../queries/#a-target)). Their derivatives agree too, to second order. The series below
it keep as many terms as double precision needs there, up to $\theta^{22}$ for the inverse
Jacobians' coefficient.

> [!DECISION] One cutoff, high enough for the GPU
> The closed forms whose numerators hold a sine, those of the Jacobians of the motions, lose a factor
> of ten to a GPU's sine just above a low cutoff. Moving the cutoff costs a few more terms of each
> series on every device, and saves choosing a cutoff by precision, which a composite cannot do.
> [Accuracy on the GPU](../../archive/wgpu-accuracy/) derives the conditions and measures them.

## Euler angles

`Euler` names a convention: one of the twelve sequences of axes and whether the turns are about the
axes as the previous turns left them (intrinsic) or about the fixed axes (extrinsic).
`Euler::YAW_PITCH_ROLL` is the aerospace convention, intrinsic about $(z, y, x)$.
`rotation::from_euler` is smooth everywhere. The way back, `euler`, is not, and no formula could
make it so: [the note](../../math/rotations/#angles-and-directions) proves that it has two
singularities.

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

Some maps are undefined at a point the mathematics names: the inverse retraction at the half-turns
from its base point, the inverses of the Jacobians at an angle of $2\pi$, `from_two_vectors` at
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

The inverse retraction's cut matters most, since every term on these types reads its value through
it. A value given exactly a half-turn from its mean, a measurement a half-turn from its prediction
or a start a half-turn from a prior's mean, makes the term non-finite, and the
[linearization](../linearization/#linearizing-a-query) refuses that point with `NotFinite`,
naming the term and the row; near the cut the residual is large, and an estimator started there
crawls away from it. A half-turn between a value and its mean is the largest error a rotation can
have, so an estimate that fits its data never meets the cut: it is a fact about a start, and a
start is chosen away from it.

> [!DECISION] A half-turn is left non-finite
> A finite residual selected at the cut would let an evaluation at a bad start succeed only by
> hiding an error of a half-turn, and refusing a value within some angle of the cut would need an
> angle that no model's noise sets. See
> [the mathematics](../../archive/quaternion-signs/math.svx#the-rotations-the-rigid-motions-and-the-extended-poses).

## On the host

`host` does in double precision what a model does before its values are loaded or after it has
run: composing, converting between quaternions, matrices, Euler angles and rotation vectors, for
constants and observations given in those forms, and choosing the quaternion with $w \ge 0$ to
report.
