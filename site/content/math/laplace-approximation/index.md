---
title: The Laplace approximation
part: estimation
order: 5
summary: A Laplace approximation replaces a density near a point by a Gaussian in the chart there; taken in a chart it approximates the density times the chart's volume, its mean, marginals, draws and evidence are computations on the linearization's rows, and its covariance omits curvature of the order of the noise's variance on a curved space.
depends_on: [linearization, square-root-elimination, chart-volume]
---

Once a mode has been found, the question is how sure the answer is: the spread of the density
around the mode, the spread of a few of its coordinates with the rest integrated out, the total
mass under it. Each is an integral that has no closed form for anything but a Gaussian. A _Laplace
approximation_ replaces the density by a Gaussian fitted at a point, and reads the answers off the
Gaussian. This note fits it from the [linearization](../linearization/), says what each answer
costs to compute, and measures what the fitted covariance leaves out when the space is curved.

Throughout, $f$ is the negative log-density [](eq:math/linearization/objective) of a point $\point$
of a product of manifolds, against the product $\nu$ of their reference measures, $\hat \point$ is
the point of approximation, and $J$ is the stacked Jacobian of the residuals there.

## The integral, in a chart

The integrals in question are against $\nu$, and a Gaussian lives on a vector space. The chart at
$\hat \point$ connects them, and it brings its [volume](../chart-volume/) with it. By
[](eq:math/chart-volume/chart-integral), for a chart that is full,

$$
	\int e^{-f} \, \dd \nu = \int_{U_{\hat \point}} e^{-f(\hat \point \oplus \tangent) + \lambda_{\hat \point}(\tangent)} \, \dd \tangent , @tag(chart-integral)
$$

with $\lambda_{\hat \point}$ the log-volume of the chart at $\hat \point$. What is approximated by a
Gaussian in $\tangent$ is therefore the _chart objective_ $f(\hat \point \oplus \tangent) -
\lambda_{\hat \point}(\tangent)$, and not $f$ alone. Its linearization, the _chart linearization_,
has the rows of the linearization of $f$, with $\eta - \nabla \lambda_{\hat \point}(0)$ and $c_0 -
\lambda_{\hat \point}(0)$ in place of $\eta$ and $c_0$. The difference shows in two places.

- **The centre.** The Gaussian is centred at $\tangent = 0$ when $\hat \point$ is a stationary
  point of the chart objective, where $g = \nabla \lambda_{\hat \point}(0)$, and not of $f$. On a
  Euclidean space $\lambda = 0$. On a group whose chart translates the identity's, as the Cayley
  chart of the rotations does, the log-volume is the same at every point and even in $\tangent$,
  as the stereographic chart's of the sphere is, so its gradient at $0$ is zero: on these spaces
  the centre is a mode of $f$. On a positive space $\nabla \lambda_{\hat \point}(0) = 1$ in each
  coordinate, and the centre is the mode in the logarithmic chart. For a log-normal of median
  $\mu$ it is $\mu$ itself, where $\log \randomPoint$ is centred, while the mode with respect to
  Lebesgue measure is $\mu\, e^{-\sigma^2}$, below it. A Gaussian in the tangent coordinates is a
  Gaussian in the logarithms, and is centred where they are.
- **The constant.** $\lambda_{\hat \point}(0)$ is $\sum_i \log \hat \point_i$ on a positive space
  and zero where the reference measure is the one the retraction normalizes: the volume of the
  reference measure near $\hat \point$.

## What it gives

When the rows of the chart linearization $m$ have full rank, $e^{-m}$ is a constant times a normal
density on $\R^\dimension$, and everything asked of the approximation is a computation on the rows
by [elimination](../square-root-elimination/#what-the-eliminated-problem-answers), with $R$ the
triangular root the elimination leaves and $\tangent^*$ the minimizer of $m$.

| asked for                                              | computed by                                                                                                                                                            |
| ------------------------------------------------------ | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| the mean, in the tangent at $\hat \point$              | the minimizer $\tangent^*$, zero at the centre                                                                                                                         |
| the covariance, and its marginal on some variables $K$ | partial elimination of the others ([%full](statement:math/square-root-elimination/marginal)), and the triangular root $W$ of what is left on $K$: $W^{-1} W^{-\top}$   |
| a draw                                                 | $\hat \point \oplus (\tangent^* + R^{-1} z)$, with $z$ standard normal                                                                                                 |
| the evidence                                           | $\log \int e^{-f} \dd \nu \approx \tfrac{\dimension}{2} \log 2\pi - m(\tangent^*) - \sum_i \log \lvert R_{ii} \rvert$, by [](eq:math/square-root-elimination/integral) |

No covariance matrix is formed for any of them. A marginal on one variable is a normal in the
chart at that variable's estimate, with a whitening of its own, and costs one partial elimination.
A draw lies in the chart's domain on every space whose chart is global. Where the chart is not
global, the Gaussian's mass outside its domain is an error of
[its own](../chart-volume/#a-density-pushed-through-a-chart), apart from those below.

## What the covariance leaves out

The Gaussian's precision is the rows', $J\transpose J$. The Hessian of the chart objective at the
centre is that and

$$
	D = \sum_t \sum_i (r_t)_i\, \nabla^2 (r_t)_i + \sum_t \nabla^2 \lambda_t - \nabla^2 \lambda_{\hat \point}(0) , @tag(omitted)
$$

the curvature of the residuals, of the terms' log-volumes and of the chart's log-volume: the part
of [](eq:math/linearization/hessian) that the linearization omits, with the chart's term added.
For linear-Gaussian terms $D = 0$ ([%full](statement:math/linearization/exact)), and the
approximation is exact. Otherwise, on Euclidean and positive spaces, only the first sum remains,
the curvature that every Gauss–Newton method leaves out. On a curved space all three parts are of
the same order. Each is of order one, while $J\transpose J$ is of the order of the terms'
precisions, so each changes the covariance by a relative amount of the order of the noise's
variance, $\sigma^2$.

- **The chart's part is positive.** For the Cayley chart of the rotations and the stereographic
  chart of the sphere, the log-volume is $-2 \log(1 + \lvert \tangent \rvert^2 / 4)$, whose Hessian
  at zero is $-I$. So the chart adds $+I$ on each such variable that is integrated over.
- **The terms' part is negative.** Let $G_t$ be the derivative of a term's tangent coordinate
  $\point \ominus \mu$ with respect to $\tangent$. The term's log-volume adds about
  $-G_t\transpose G_t$, against the $G_t\transpose \Lambda_t G_t$ that its rows keep: the
  replacement of $\Lambda$ by $\Lambda - I$ that
  [a measurement on a curved space](../normal/#what-the-parameters-mean) shows.
- **The residuals' part is not small, even at the mode.** Along the retraction, the second
  derivative of $\point \ominus \mu$ is of first order in the residual. The first sum of
  [](eq:omitted) is therefore quadratic in the residuals, of order one like the others, and it
  vanishes only where the residuals do. For a rotation measured in the Cayley chart, it averages
  about $+\tfrac12 I$ over the noise, against the log-volume's $-I$, as computed numerically from
  4000 draws at $\sigma = 0.1$.

**The sign is not fixed.** One covariance is smaller than another exactly when its precision is
larger, and then every marginal of it is smaller too. So the rows' covariance is smaller than the
second-order covariance $(J\transpose J + D)^{-1}$ in every direction when $D$ is negative
semidefinite, and larger in every direction when $D$ is positive semidefinite. Neither holds in
general: the chart's part is positive, the terms' log-volumes negative, and the residuals' part
can be of either sign.

What a covariance is meant to describe, though, is the density itself, and the second-order
approximation is itself in error on a curved space, at the same relative order $\sigma^2$. Two
examples on the rotations, in the Cayley chart with an isotropic $\sigma$, show how the three
covariances compare with the true one. Both were computed numerically: the true covariance in the
chart at the mode by importance sampling, or exactly from the chain that generates it.

The first example is a rotation with a prior of fixed mean and $k$ measurements of it, approximated
at the mode. The table gives the trace of each covariance divided by the true one, averaged over
six data sets.

| $\sigma$ | $k$ | rows    | rows and log-volumes | full Hessian |
| -------- | --- | ------- | -------------------- | ------------ |
| $0.3$    | $1$ | $0.976$ | $1.021$              | $1.004$      |
| $0.3$    | $3$ | $0.971$ | $1.038$              | $1.003$      |
| $0.15$   | $1$ | $0.993$ | $1.004$              | $1.000$      |
| $0.15$   | $3$ | $0.992$ | $1.008$              | $0.999$      |

The rows are over-confident by about half of the $\sigma^2 k / (k + 1)$ that the log-volumes alone
would predict. Adding the log-volumes' curvature overshoots, to an under-confidence of about the
same size, because the residuals' curvature cancels about half of it. The full Hessian is close to
the truth.

The second example is a chain: $\randomPoint_0$ with a prior of fixed mean, and $\randomPoint_1 =
\randomPoint_0 \oplus \varepsilon$, with nothing observed. It is approximated at its mode, where
every residual is zero. The residuals' curvature therefore vanishes, and the rows with the
log-volumes' curvature are the full Hessian. The variance of $\randomPoint_1$, per axis, is as
follows.

| $\sigma$ | rows     | full Hessian | true     |
| -------- | -------- | ------------ | -------- |
| $0.3$    | $0.180$  | $0.159$      | $0.194$  |
| $0.15$   | $0.0450$ | $0.0435$     | $0.0457$ |

Both approximations are over-confident here, and the rows are the closer. They are exact on
$\randomPoint_0$, where the full Hessian gives $0.083$ against $0.09$ at $\sigma = 0.3$. The
second-order approximation matches the curvature at the mode exactly, and that does not match the
spread: the law is a Gaussian in the chart of $\randomPoint_0$ and in that of the transition, and
not in the joint chart at the mode.

So, on a curved variable:

- **Confidence.** Compared with the second-order approximation, the rows' covariance may be over-
  or under-confident, depending on the direction. Compared with the truth, it was slightly
  over-confident in every case computed here: by $1$ to $2\,\%$ at $\sigma = 0.15$ rad and $3$ to
  $7\,\%$ at $\sigma = 0.3$ rad. No argument shows that it must be. On Euclidean and positive
  spaces with linear-Gaussian terms it is exact.
- **Consistency.** Adding the log-volumes' curvature to a marginal is not the consistent choice.
  It adds one of three parts of the same order and leaves out the residuals', which in the first
  example cancels about half of it. Added on the kept variables alone, it also leaves out the
  curvature on the eliminated ones, which reaches the marginal through their conditionals. To
  first order in $D$, the marginal precision on $K$ changes by $P\transpose D P$, not by $D_{KK}$,
  where $P$ lifts a tangent vector on $K$ to the whole tangent by the conditionals of the
  eliminated variables $E$, with the right-hand sides set to zero: with $\Lambda = J\transpose J$
  in blocks, $P = (I, -\Lambda_{EE}^{-1} \Lambda_{EK})$. The consistent second-order approximation
  is the full Hessian. It needs second derivatives, it can be indefinite, and in the second
  example it is no closer to the truth.
- **Definiteness.** The rows' precision is positive definite whenever the rows determine every
  direction. With the log-volumes' curvature, the terms' and the chart's parts sum to about
  $\sum_t G_t\transpose (\Lambda_t - I) G_t + I$ on the curved blocks, which is positive definite
  whenever every $\Lambda_t - I$ is, that is, when every curved term's noise is less than one in
  its chart's coordinates. This is the condition under which a normal's mean is a mode of its
  density. Definiteness is lost only where the normals themselves are broad, and the residuals'
  part can still move this boundary.

The rows' precision is thus a sum of squares, positive definite by construction, obtained from
first derivatives and one elimination, and in error on a curved variable by a relative amount of
order $\sigma^2$. Adding part of the omitted curvature buys no consistency that the rows lack, and
the full second-order approximation is not reliably closer. A covariance of a curved variable
reported from the rows should be reported with the order of its error.

## A marginal kept as a Gaussian

Integrating variables out by elimination of the linearization leaves an
[improper Gaussian factor](../improper-gaussian/#what-elimination-leaves) on the variables that
remain, anchored at the point of linearization. For linear-Gaussian terms it is the exact marginal.
Otherwise it approximates the logarithm of the marginal density, and it differs from the marginal
of a Laplace approximation in two ways. The chart's part of [](eq:omitted) is there only for the
variables integrated over, so nothing offsets the terms' log-volumes on the variables kept. And
the factor is a function on the manifold, not a Gaussian in a chart of it, so whoever integrates
against it later applies the chart's volume to it again.

The chain of the second example shows both, read as integrating $\randomPoint_0$ out and keeping
$\randomPoint_1$. The exact factor is the logarithm of the marginal density of $\randomPoint_1$,
whose curvature at the anchor was computed numerically from draws of $\randomPoint_0$. These are
precisions per axis.

| $\sigma$ | rows    | with [](eq:omitted) | with [](eq:omitted) and the log-determinant | exact   |
| -------- | ------- | ------------------- | ------------------------------------------- | ------- |
| $0.3$    | $5.556$ | $5.294$             | $5.457$                                     | $5.455$ |
| $0.15$   | $22.22$ | $21.97$             | $22.10$                                     | $22.10$ |

The rows are over-confident at the anchor, by $2\,\%$ at $\sigma = 0.3$ and $0.6\,\%$ at $0.15$.
The second-order Schur complement overshoots, by about as much, the other way. The exact curvature
needs one more term: how the log-determinant of the eliminated block's Hessian varies with the
variables kept, the correction of Laplace's method of the next order.

The larger cost lies elsewhere. The exact marginal is not a Gaussian in the anchor's chart,
whatever its curvature, and the departure is of the order of the marginal's own variance. With the
exact curvature, the factor alone gives $\randomPoint_1$ the variance $0.159$ per axis, against the
$0.193$ of the exact marginal at $\sigma = 0.3$, and $0.0435$ against $0.0458$ at $\sigma = 0.15$.
A filter that predicts without measuring integrates one state out after another, and its factor's
variance grows. Its error grows with it. At $\sigma = 0.15$, after $k$ such steps along the chain,
the variance that the factor gives the newest state, divided by the exact one, is as follows.

| $k$                 | $1$    | $2$    | $3$    | $4$    | $5$    |
| ------------------- | ------ | ------ | ------ | ------ | ------ |
| rows                | $0.94$ | $0.91$ | $0.88$ | $0.85$ | $0.82$ |
| with [](eq:omitted) | $0.94$ | $0.90$ | $0.84$ | $0.77$ | $0.71$ |

This is the over-confidence of an extended Kalman filter on a curved state. It is a cost of
keeping a marginal as a Gaussian in one chart, and not of the omitted curvature: adding the
curvature makes it worse after the first step. Measurements bound the factor's variance, and with
it this error.
