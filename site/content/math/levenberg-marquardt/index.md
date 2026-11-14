---
title: The Levenberg–Marquardt method
part: estimation
order: 4
summary: A Gauss–Newton step is trusted only as far as the linearization is; damping shortens it, the ratio of the decrease obtained to the decrease promised says whether to trust it more or less, and a stopping rule on the promise alone tells convergence from rounding.
depends_on: [linearization, square-root-elimination]
---

A [Gauss–Newton step](../linearization/) minimizes a model of the function, and the model is good
only near the point where it was made. From a poor start the step overshoots, and the function
rises. The Levenberg–Marquardt method [](cite:levenberg1944) [](cite:marquardt1963) keeps the step
and limits its length, by an amount it adapts as it learns how far the model can be trusted. This
note states the method as four choices: the damping, the measure of trust, the schedule that
adapts it, and the rule that stops it.

Throughout, $f$ is the sum of squares and log-volumes [](eq:math/linearization/objective), $m$ its
linearization at the current point $\hat \point$, $J$ the stacked Jacobian of the residuals there,
and $g$ the gradient.

## The damping

A step minimizes the linearization plus a penalty on its length, $m(\tangent) + \tfrac12 \lambda\,
\tangent\transpose D \tangent$, for a $\lambda \ge 0$ and a positive diagonal $D$. The penalty is
[rows of its own](../square-root-elimination/#damping), one block for each variable, so damping
changes neither the structure of the problem nor the order of its elimination, and it makes every
direction determined for $\lambda > 0$. The damped minimizer solves

$$
	(J\transpose J + \lambda D)\, \tangent = -g . @tag(damped-step)
$$

At $\lambda = 0$ it is the Gauss–Newton step. As $\lambda$ grows it turns toward $-D^{-1} g /
\lambda$, a short step down the gradient.

The diagonal is Marquardt's, the squared norms of the columns of $J$,

$$
	D_{ii} = \max\bigl( \lvert J_{:, i} \rvert^2,\ \delta \bigr) , @tag(marquardt)
$$

so that the damping does not depend on the units of the coordinates: a rotation in radians and a
position in metres are damped alike, and rescaling a coordinate rescales its column and its
damping together. The floor $\delta > 0$ keeps $D$ positive where a column vanishes. $D$ is
recomputed at every linearization. With it, $\lambda$ is dimensionless, a multiple of the curvature
along each coordinate.

## The gain ratio

The damped step $\tangent$ promises a decrease of the undamped linearization,

$$
	\Delta(\tangent) = m(0) - m(\tangent) = \tfrac12 \lvert J \tangent \rvert^2 + \lambda\, \tangent\transpose D \tangent , @tag(promised)
$$

since $m(0) - m(\tangent) = -g\transpose \tangent - \tfrac12 \tangent\transpose J\transpose J
\tangent$ and $-g\transpose \tangent = \tangent\transpose (J\transpose J + \lambda D) \tangent$ by
[](eq:damped-step). What it obtains is $f(\hat \point) - f(\hat \point \oplus \tangent)$. Their
ratio

$$
	\varrho_\lambda = \frac{f(\hat \point) - f(\hat \point \oplus \tangent)}{\Delta(\tangent)} @tag(gain)
$$

is one where the linearization is exact, and falls as it stops being trustworthy over the step's
length. Here $f$ is the function itself, log-volumes included, evaluated at the trial point. The
linearization's constant is not needed.

## The schedule

The schedule is Nielsen's [](cite:nielsen1999), with $\lambda_0 = 10^{-3}$ and $\beta = 2$ at the
start:

$$
	\begin{gathered}
		\varrho_\lambda > 0 : \quad \hat \point \leftarrow \hat \point \oplus \tangent, \quad \lambda \leftarrow \lambda \max\bigl(\tfrac13,\ 1 - (2 \varrho_\lambda - 1)^3\bigr), \quad \beta \leftarrow 2 , \\
		\varrho_\lambda \le 0 : \quad \lambda \leftarrow \beta \lambda, \quad \beta \leftarrow 2 \beta .
	\end{gathered} @tag(schedule)
$$

An accepted step with $\varrho_\lambda$ near one divides $\lambda$ by three, one near zero keeps
it, and a rejected step multiplies it by a factor that doubles with each rejection in a row. The
update is continuous in $\varrho_\lambda$, which avoids the oscillation between two values of
$\lambda$ that a rule with thresholds shows. With $\lambda_0$ a small multiple of the curvature, a
start near the minimum takes nearly a Gauss–Newton step.

## The stopping rule

The method stops when a step promises less than a tolerance, $\Delta(\tangent) \le \epsilon$,
measured in the units of $f$ itself, nats for a log-density, whether or not the step is accepted:
the step is taken if its gain is positive, and the method stops either way. It reports failure,
and not convergence, when $\lambda$ has grown past a bound $\lambda_{\max}$ or the iterations past
theirs.

The promise is the right measure. It is invariant to the coordinates of the chart, and near the
minimum, where $\lambda$ is small, it approaches the Gauss–Newton decrement $\tfrac12 g\transpose
(J\transpose J)^{-1} g$, the decrease of $f$ that a full Gauss–Newton step would still give. A
gradient's norm would depend on the units of the coordinates, and a step's on the chart.

**The tolerance is limited by the arithmetic.** The obtained decrease in [](eq:gain) is a
difference of two values of $f$, each with an error of about $u \lvert f \rvert$ at a unit roundoff
$u$, while $\Delta$ is computed from the rows themselves. For the ratio to mean anything, $\Delta$
must stay well above $u \lvert f \rvert$, and so must $\epsilon$. With $f$ of the order of the
number of residual components, a few thousand say, double precision allows $\epsilon = 10^{-9}$,
and single precision nothing below about $10^{-3}$.

**Why the rule does not wait for an acceptance.** Once $\Delta$ is of the order of $u \lvert f
\rvert$, the obtained decrease is rounding and the gain has a random sign; a step that promises
nothing has the gain $0/0$. A rule that stopped only on an accepted step would meet such steps
just after convergence, since Gauss–Newton contracts fast near a minimum and the step after one
promising slightly more than $\epsilon$ promises far less. Each rejection multiplies $\lambda$ by
$\beta$ and doubles $\beta$, so a run of some ten of them, about one chance in a thousand, would
carry $\lambda$ past $\lambda_{\max}$ and report a converged run as one that failed. Stopping on
the promise alone forms no gain from rounding, and a run is never failed for having converged.

## What is checked at the end

The stopping rule bounds the damped step, not the undamped one. The damped promise is at least
$\tfrac12 g\transpose (J\transpose J + \lambda D)^{-1} g$, which bounds the Gauss–Newton decrement
when $\lambda$ is small and not when it is large, as it may be at a stop that followed rejections.
So convergence is checked apart from the rule: at the final point, with $\lambda = 0$, one
elimination gives the Gauss–Newton step $\tangent_0$ and its decrement $\tfrac12 \lvert J \tangent_0
\rvert^2$, which must be at most a small multiple of $\epsilon$. That is the statement that the
point is stationary for $f$, since when the rows have full rank $\tangent_0 = 0$ exactly where $g =
0$ ([](eq:math/linearization/step)).
