---
summary: The uniform distribution on a type of finite mass, its constant term and its sampler; and what a normal with a large scale is on a curved type, with the ways of refusing it, bounding its scale, replacing its law and documenting it weighed against each other.
depends_on: [uniform, chart-volume, normal, archive/normal/math, archive/measures/math, archive/quaternion-signs/math]
---

## The problem

A model often has to say that nothing is known of a rotation or a direction before the data are
seen: the heading of a robot at start-up, the attitude of a camera, the first pose of a graph. On
a Euclidean type the habit is a normal with a very large standard deviation, and the habit carries
over to curved types unexamined. There it says something else. [The normal](../normal/math.svx) of
a curved type is the law of its mean moved by Gaussian noise through a global chart, and under the
Cayley chart that [the rotations now have](../quaternion-signs/math.svx) a large tangent vector is
a turn of nearly $\pi$: the larger the scale, the more surely the draw is a half-turn from the
mean, where the chart is cut and a term is not finite.

Two things are worked out here. The first is the distribution that does say "unknown", the uniform
distribution on a type whose reference measure is finite: its term, its sampler, and what an
estimator reads of it. The second is what to do about a normal with a large scale on a curved
type, whether its scale is a constant someone wrote or a random quantity that is large with small
probability. Four ways are weighed: refusing it, bounding the scale by a saturating function,
replacing the normal's law by one that tends to the uniform, and documenting it.

The general mathematics is in the note on
[the uniform distribution](../../math/uniform/), written with this entry. What follows is
skoro's own: which types, which choices, and why.

## Setting

A continuous type $\manifold$ has, or declares, a reference measure $\nu$ and the log-volume
$\Lambda(\mu, \tangent)$ of its chart ([the measures](../measures/math.svx)). The types at issue
are those of finite mass $\nu(\manifold)$:

- the rotations `Rotation`, $SO(3)$ stored as unit quaternions up to their sign, with the Cayley
  chart, global and full, and Haar measure of mass $8\pi^2$;
- the core's two test spheres, $S^2$ with surface measure of mass $4\pi$, one with the
  stereographic chart, global and full, and one with the exponential chart, full and not global;
- products of these, with the product of their masses.

The rigid motions `Pose` and the extended poses `ExtendedPose` have translations and so infinite
mass, as have the Euclidean and positive types. The masses are those of
[the note's table](../../math/uniform/#the-definition).

The normal $\mathcal N(\mu, \Sigma)$ is that of [its note](../../math/normal/), with the
log-density [](eq:math/normal/log-density), the whitening $W$ and the precision $\Lambda =
W\transpose W$. An isotropic scale is $\Sigma = \sigma^2 I$, and on a curved type $\sigma$ is in
the coordinates of the chart, which are radians to first order.

## The uniform distribution

On a type of finite mass the uniform distribution is the law $\nu / \nu(\manifold)$, with the
log-density [](eq:math/uniform/log-density),

$$
	\logterm_{\mathcal U}(\point) = -\log \nu(\manifold) , @tag(term)
$$

which is $-\log 8\pi^2$ on the rotations and $-\log 4\pi$ on a sphere, and on a product the sum of
its factors'. It has no parameters, and it is defined by the measure alone: unlike the normal it
asks nothing of the chart, so the exponential test sphere, on which the normal is refused, has a
uniform distribution like any other type of finite mass.

**The mass is declared.** A type declares its log-volume because computing it would cost
derivatives the core does not have ([Declared, not computed](../measures/math.svx#declared-not-computed)).
The mass is a number, the integral of that volume over the chart's domain,
[](eq:archive/measures/math/total-mass), and that identity is already among the checks of a
declared volume, where the test supplies the mass. The type now declares it, $\log \nu(\manifold)$,
and the same check compares the declaration with the integral. A type that declares no mass has no
uniform distribution, and a product has one when every factor has.

**What the term is.** The term [](eq:term) is a constant, so its gradient and its Hessian product
vanish in every chart, and it changes neither a mode nor a step towards one. It is not nothing: it
is the term that makes the model's density a density, and it enters the evidence and every
importance weight, where a model with a uniform prior and one with a normal prior are compared at
their true normalizations. As a proposal, the uniform distribution has the same constant density,
and its weight is the model's density times the mass.

**The uniform among the normals.** By [%full](statement:math/uniform/tangent), a uniform rotation
is $\mu \oplus \varepsilon$ at any $\mu$, with $\varepsilon$ a Cauchy vector of scale $2$ in the
Cayley chart, and so, by [](eq:math/uniform/mixture), a mixture of the normals $\mathcal N(\mu,
\sigma^2 I)$ over the scale $\sigma = 2 / \lvert g \rvert$, with $g$ standard normal. "Unknown" is
therefore not a normal with one large scale. It is the normals of every scale together, most of
them broad, with weights that only the chart's volume could have named.

## Drawing a uniform point

The normal's sampler draws on the host, in double precision, from $\dimension$ uniform variates of
the row's key, and its draw is a fixed function of them
([the normal's sampler](../normal/math.svx)). The uniform distribution has two samplers of the
same kind.

*Through the chart.* Draw the tangent vector of [%full](statement:math/uniform/tangent) and
retract a fixed point by it: on the rotations, $\varepsilon = 2 \zeta / g$ from four normal
variates. It needs only the type's retraction and a sampler of the tangent law, but that law is
the type's own (a Cauchy for one chart, a Student t of two or three degrees of freedom for others),
it spends a variate more than the dimension or inverts a distribution function with no closed
form, and its largest draws are retracted where the chart is steepest.

*Directly.* A map from the unit cube of $\dimension$ variates into storage that carries Lebesgue
measure to the uniform distribution: [%full](statement:math/uniform/draws) gives one for $S^2$
and one for the rotations,

$$
	q = \bigl(\sqrt{1 - u_1} \cos 2\pi u_2,\ \sqrt{1 - u_1} \sin 2\pi u_2,\ \sqrt{u_1} \cos 2\pi u_3,\ \sqrt{u_1} \sin 2\pi u_3\bigr) , @tag(rotation-draw)
$$

a unit quaternion of a uniform rotation, in whichever order storage keeps its parts, since the
uniform distribution on $S^3$ is invariant under a permutation of coordinates. Its sign is
immaterial to a type stored up to it.

> [!DECISION] A type of finite mass declares its uniform draw
> The direct map is the sampler. It spends exactly $\dimension$ variates, as the normal's does; it
> is a closed form of square roots, sines and cosines; it needs no base point; and nothing in it
> comes near a chart's cut. It is declared by the type, beside the mass, because no formula gives
> it from the retraction alone. A product draws factor by factor from consecutive variates. The
> chart's route stays as the test: the coordinates $\point \ominus \mu$ of the draws, at a $\mu$ of
> the test's choosing, must follow the law of [the note's table](../../math/uniform/#in-a-chart).
> On the rotations their direction is uniform and their length $r$ gives an angle $\theta = 2
> \arctan(r/2)$ with the distribution function $(\theta - \sin \theta) / \pi$.

## A normal with a large scale

[%full](statement:math/uniform/broad) says what the habit of a large scale does on a compact type
with a global chart: as the scale grows, the normal's mass leaves every compact part of the chart's
image, its total variation distance from the uniform distribution tends to $1$, and its limit sits
on the chart's cut. On the rotations with an isotropic scale the limit is the mean turned by a
half-turn about a uniform axis. The median angle between a draw and the mean, computed from the
chart, is

| $\sigma$ | $0.5$ | $1$ | $3$ | $10$ | $100$ | $1000$ | uniform |
| --- | --- | --- | --- | --- | --- | --- | --- |
| median angle, rad | $0.73$ | $1.31$ | $2.32$ | $2.88$ | $3.12$ | $3.14$ | $2.31$ |

These replace the figures [the normal's entry](../normal/math.svx) was evaluated with, which were
drawn under the stereographic chart of the quaternions, where a broad normal returned to its mean
on the far side of the sphere. Under the Cayley chart it does not return: at $\sigma = 100$ a draw
is within $0.1$ rad of a half-turn with probability $0.98$, where the uniform distribution puts
$0.064$. The mean is a mode only while every eigenvalue of $\Sigma$ is below $1$
([Where the modes are](../../math/uniform/#a-normal-with-a-large-scale)); beyond that the density
against Haar measure rises away from the mean, to a shell at the angle $2 \arccos(1 / \sigma)$ for
an isotropic scale, and inside that shell the term's gradient pushes an estimate away from the
mean it was meant to stay near. No isotropic normal is closer to uniform than $0.37$ in total
variation.

None of this is an error in the normal. Its density and its sampler are exact at every scale, and
a scale of $1.2$ rad is a legitimate, if unusual, noise. What is wrong is a reading: that a large
scale means little is known. Four ways of meeting it follow.

**Refusing.** A normal on a curved type could refuse a constant scale with an eigenvalue of
$\Sigma$ above $1$ on a curved block. It cannot refuse a scale that is a quantity or computed from
one: a scale with a prior of its own is large with some probability, a refusal at one value of a
random scale is not a law, and a term that turned non-finite there would cut the scale's prior
without saying so. The rule would therefore hold for constants and not for the same number
arriving any other way, and the threshold would be a property of the chart's volume that the
normal would have to be told of each type. It would also refuse laws that are exact and may be
meant.

**Bounding the scale.** A saturating function could be put between the scale given and the scale
used, such as $\sigma \mapsto \sigma_{\max} \tanh(\sigma / \sigma_{\max})$ with $\sigma_{\max} \le
1$, so that the mean stays a mode and no mass reaches the cut. Its costs are these.

- It does not say "unknown" either. Every large scale becomes $\mathcal N(\mu, \sigma_{\max}^2 I)$,
  which at $\sigma_{\max} = 1$ has a median angle of $1.31$ rad against the uniform's $2.31$, and
  is $0.65$ from it in total variation. A prior written with $\sigma = 1000$ would be a confident
  statement that the rotation is within about $75°$ of its mean, and nothing non-finite would ever
  reveal it. No function of the scale can do better, because the family has no member near the
  uniform distribution at any scale.
- It changes what every scale means. A smooth saturation is not the identity anywhere: the one
  above turns $0.1$ rad into $0.0997$. A scale would no longer be the standard deviation of the
  noise in the chart, which is the one thing the normal promises of it, and a function that is the
  identity below a threshold and saturates above it is not smooth where they join.
- It makes a large scale unidentifiable. The log-density's derivative in $\log \sigma$ is
  multiplied by the saturation's logarithmic slope, which tends to zero, and the scale's block of
  the expected information, exact today ([the normal's note](../../math/normal/#the-expected-information)),
  by its square. An estimated scale that wanders past the bound stops being pulled back by the
  data.
- It has no form for a full covariance. The condition is on the eigenvalues of $\Sigma$, the
  normal reads its scale through a triangular factor, and a saturation of eigenvalues needs a
  decomposition the normal otherwise never computes. On a product, such as a rotation with a
  position, it would have to act on the curved block of a covariance that correlates the two.

**Replacing the law.** A law could be chosen that tends to the uniform distribution as its scale
grows. The natural one is [](eq:math/uniform/against-measure), the Gaussian of the residual as a
density against the reference measure: its mode is its mean at every scale, it is the law that
least squares of the whitened residual estimates exactly, and its limit is the uniform
distribution. It is not the normal, and making the normal into it would undo what the normal's
entry chose it for. Its normalizer is a function of the scale with a closed form only in special
cases, [](eq:math/uniform/normalizer) for an isotropic scale on the rotations, so a full or an
estimated scale has no exact term; and its draw is by rejection, with a random number of variates
and an acceptance that falls to $0.07$ at $\sigma = 3$ from a normal proposal, where the normal's
draw is a fixed smooth function of its variates. On the flat types it is the normal, so it would
be a second law on curved types only, beside the first. As a distribution of its own, with an
isotropic scale, it is a candidate for a later entry, and Student t noise in the chart, of which
the uniform is one member, is another.

**Documenting.** The normal stays as it is, exact at every scale, and the site says what a large
scale is: that the mean is a mode only below a scale of about one radian in the chart, where the
mass goes beyond it, and that the uniform distribution is how "unknown" is said.

> [!DECISION] Large scales are documented, and the uniform distribution is the remedy
> The normal neither refuses nor alters a scale. Refusal cannot reach a random scale and would
> split one number into two rules; a bound on the scale silently replaces one wrong prior by
> another and costs the scale its meaning and its information; and a law that tends to the
> uniform is a different distribution, with a normalizer and a sampler the normal was chosen to
> avoid. What misled was the absence of a way to say "unknown", and this entry supplies it. The
> pages of the normal and of the rotations state the threshold, the table above and the remedy,
> where a reader choosing a scale will meet them, and the tests draw the table's medians from the
> normal's sampler, so that the documented behaviour is checked and not only described.

A random scale needs no rule. A prior on a scale that reaches past one radian with small
probability gives those draws the law [%full](statement:math/uniform/broad) describes, with small
probability, and the model remains exactly what was written. A model that means "the noise is
either small or the measurement says nothing" is a mixture of a normal and the uniform
distribution, which this entry makes it possible to write.

## In skoro's terms

- **A type of finite mass declares** $\log \nu(\manifold)$ and a uniform draw, a map from
  $\dimension$ variates to its storage. `Rotation` declares $\log 8\pi^2$ and
  [](eq:rotation-draw); a sphere $\log 4\pi$ and the first map of
  [%full](statement:math/uniform/draws); a product has the sum of its factors' log-masses and
  their draws side by side. The mass is checked against the integral of the declared volume,
  [](eq:archive/measures/math/total-mass), and the draw against the law of its coordinates in a
  chart.
- **The uniform distribution is a term with no parameters.** It scores one random quantity of a
  type of finite mass with the constant [](eq:term). Nothing is differentiated: its gradient and
  Hessian products with respect to the quantity are zero, in every chart, and it has no scale and
  no mean to differentiate with respect to.
- **The quantity it scores is random**, drawn in a generation by the type's uniform draw on the
  host, from the variates of its key, as the normal's quantity is drawn by the retraction. It may
  be observed, and then the term is a constant of the data.
- **In a linearization it is a factor with no rows.** The term is the
  [improper Gaussian factor](../../math/improper-gaussian/) with a matrix of no rows, no linear
  part and the constant $c = -\log \nu(\manifold)$: flat in every direction, exactly, with nothing
  approximated. It adds nothing to the rows a step is solved from, and its constant enters the
  evidence of a Laplace approximation. A quantity that no other term bears on has then no rows at
  all, and the problem is singular in its block, as it should be, since its posterior is uniform
  and has no mode.
- **The normal is unchanged.** It accepts every scale on every type it accepted, its term, its
  sampler and its roles are as [its entry](../normal/math.svx) left them, and what this entry
  revises is what its pages say of a large scale on a curved type.
- **The tests** draw from the normal on `Rotation` at the scales of the table and compare the
  median angle with $2 \arctan(\sigma\, m / 2)$, $m = 1.538$ the median length of a standard normal
  vector of $\R^3$, and draw from the uniform distribution and compare the angle from a fixed
  rotation with its distribution function $(\theta - \sin \theta) / \pi$.

## Open questions

> [!TODO] The finite discrete types
> A categorical type with counting measure has finite mass, the number of its values, and the
> uniform distribution on it is the term $-\log n$ with a draw from one variate. It fits the
> definition and costs little, but skoro has no distribution on a discrete type yet, and the first
> one may deserve its own entry. Is it part of this one?

> [!TODO] The Gaussian against the reference measure
> Whether [](eq:math/uniform/against-measure), with an isotropic scale, should return as a
> distribution of its own is left to a later entry here. If it is wanted now, as the prior that
> goes smoothly from concentrated to unknown, this entry would have to settle its normalizer and a
> sampler that spends a random number of variates.
