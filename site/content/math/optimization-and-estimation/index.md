---
title: Optimization and estimation, term by term
part: estimation
order: 14
summary: The notes on least squares and Gaussian approximations name most things twice, once as an optimizer would and once as a state estimator would; this table sets the two names side by side, row by row, with the note that defines each.
depends_on:
    [
        improper-gaussian,
        trust-regions,
        kalman-filter,
        levenberg-marquardt,
        laplace-approximation,
        fisher-scoring,
    ]
---

The notes of this part read one computation in two vocabularies. An optimizer sees a function to
be minimized by steps that each minimize a model of it. A state estimator sees a posterior to be
approximated by Gaussians, combined by adding factors and reduced by integrating quantities out.
The [linearization](../linearization/) and the [improper Gaussian factor](../improper-gaussian/)
are where the two meet: one quadratic, read once as a model in a trust region and once as a
Gaussian on the increment from the point of linearization. Most of what follows from either
reading has a name in both, and this note is the table of those names.

Each row pairs an optimizer's term with an estimator's and names the note that defines them. The
pairs are exact under the condition the row states, and where none is stated they are exact
throughout this part: a model built forward, its objective $f$ the negative logarithm of its joint
density at the observations, and its terms read in the charts at the current point $\hat
\point$.

## The problem

| optimization                                                   | estimation                                                                                                         | where                                                                                 |
| -------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------ | ------------------------------------------------------------------------------------- |
| the objective $f$                                              | the negative log-density of the free quantities at the observations, the negative log-posterior up to the evidence | [maximum a posteriori estimation](../map-estimation/)                                 |
| a minimizer of $f$                                             | a mode of the posterior, the maximum a posteriori estimate; the maximum likelihood estimate when there is no prior | [maximum a posteriori estimation](../map-estimation/)                                 |
| a term of $f$                                                  | the negative log-density of one kernel, likelihood factor or score                                                 | [maximum a posteriori estimation](../map-estimation/#the-objective-is-a-sum-of-terms) |
| a regularizer, a penalty on the variables                      | a prior, the term of a free quantity's own kernel                                                                  | [maximum a posteriori estimation](../map-estimation/)                                 |
| a residual, weighted by the inverse root of its covariance     | a whitened residual, the standardized error of a measurement under its normal                                      | [models of normals as least squares](../nonlinear-least-squares/)                     |
| the cost, half the sum of squared residuals                    | the negative log-likelihood of normals with fixed scales, up to a constant and the log-volumes                     | [models of normals as least squares](../nonlinear-least-squares/)                     |
| a change of variables, which leaves the minimizer where it was | a change of reference measure, which moves the mode                                                                | [maximum a posteriori estimation](../map-estimation/#a-mode-depends-on-the-measure)   |

## The model and its curvature

| optimization                                                                       | estimation                                                                                             | where                                                                                       |
| ---------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------ | ------------------------------------------------------------------------------------------- |
| the gradient $g$                                                                   | the negative of the score, the gradient of the log-density                                             | [Newton's method and trust regions](../trust-regions/)                                      |
| the Hessian of $f$                                                                 | the observed information, of the posterior when the model has priors                                   | [Newton's method and trust regions](../trust-regions/#newton-s-method)                      |
| the Gauss–Newton matrix $\sum_t J_t\transpose J_t$                                 | the precision of the linearization's Gaussian; for normals with fixed scales, the expected information | [linearization](../linearization/), [Fisher scoring](../fisher-scoring/#two-linearizations) |
| the curvature of a quasi-Newton model                                              | a choice of information: observed, expected, or the rows'                                              | [Newton's method and trust regions](../trust-regions/#any-curvature)                        |
| the quadratic model $m$ of $f$ at $\hat \point$                                    | the Gaussian factor $e^{-m}$ on the increment $\point \ominus \hat \point$                             | [the improper Gaussian factor](../improper-gaussian/#the-linearization-is-a-factor)         |
| the model's minimizer, the Gauss–Newton step                                       | the mean of the increment's Gaussian                                                                   | [the improper Gaussian factor](../improper-gaussian/#the-linearization-is-a-factor)         |
| a direction in which the curvature is singular, along which the step is not unique | a direction the factor does not determine, along which it is flat and has no covariance                | [the improper Gaussian factor](../improper-gaussian/)                                       |
| a damping $\lambda D$, a trust region's radius                                     | a normal prior on the increment, with mean zero and precision $\lambda D$                              | [the improper Gaussian factor](../improper-gaussian/#the-linearization-is-a-factor)         |
| the damped step, the Levenberg–Marquardt step                                      | the mean of the increment under that prior                                                             | [the Levenberg–Marquardt method](../levenberg-marquardt/)                                   |
| the gain ratio                                                                     | the fit of the Gaussian factor to the log-density over the step                                        | [Newton's method and trust regions](../trust-regions/#the-gain-ratio)                       |

## Solving

| optimization                                                              | estimation                                                                      | where                                                                                              |
| ------------------------------------------------------------------------- | ------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------- |
| the normal equations $A\transpose A\, \tangent = A\transpose b - \eta$    | the information form of a Gaussian                                              | [the normal equations](../normal-equations/)                                                       |
| the matrix $A\transpose A$                                                | the precision, or information matrix                                            | [the normal equations](../normal-equations/)                                                       |
| its inverse                                                               | the covariance                                                                  | [the normal equations](../normal-equations/)                                                       |
| the right-hand side $A\transpose b - \eta$, the negative gradient at zero | the information vector, the precision times the mean                            | [the normal equations](../normal-equations/)                                                       |
| the triangular root $R$, with $R\transpose R = A\transpose A$             | the square-root information matrix, a whitening of the precision                | [elimination in square-root form](../square-root-elimination/#the-triangular-root)                 |
| $\log \lvert \det R \rvert$                                               | half the log-determinant of the precision, or minus half that of the covariance | [elimination in square-root form](../square-root-elimination/#what-the-eliminated-problem-answers) |

## Eliminating and integrating

| optimization                                          | estimation                                                                             | where                                                                                              |
| ----------------------------------------------------- | -------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------- |
| minimizing over a block, keeping the rest             | maximizing the joint density over a quantity: its conditional mode given its separator | [elimination in square-root form](../square-root-elimination/#eliminating-one-block)               |
| the same elimination, by its other part               | integrating the quantity out: its marginal on the rest                                 | [elimination in square-root form](../square-root-elimination/#eliminating-one-block)               |
| the Schur complement                                  | the marginal precision                                                                 | [elimination in square-root form](../square-root-elimination/#eliminating-one-block)               |
| the reduced problem on the blocks kept                | the improper Gaussian factor on the Markov blanket                                     | [what elimination leaves](../integrating-out/)                                                     |
| back substitution through the elimination tree        | the conditional means, from the roots down, given the values above                     | [elimination in square-root form](../square-root-elimination/#what-the-eliminated-problem-answers) |
| eliminating a chain of blocks in order                | filtering: the prediction of each state from the measurements before it                | [the Kalman filter](../kalman-filter/)                                                             |
| the residual of a measurement at the current estimate | the innovation, at the prediction                                                      | [the Kalman filter](../kalman-filter/#the-filter)                                                  |
| the weight a step gives that residual                 | the Kalman gain                                                                        | [the Kalman filter](../kalman-filter/#the-filter)                                                  |
| eliminating the chain, then back substituting         | smoothing: each state's mean given every measurement                                   | [the Kalman filter](../kalman-filter/#the-filter-as-elimination)                                   |

## Approximations

| optimization                                                        | estimation                                                                       | where                                                                                 |
| ------------------------------------------------------------------- | -------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------- |
| the inverse of the model's curvature at the minimizer               | the covariance of the Laplace approximation                                      | [the Laplace approximation](../laplace-approximation/)                                |
| the minimum of the model, with the log-determinant of its curvature | the evidence, the log of the marginal density of the observations, approximately | [the Laplace approximation](../laplace-approximation/#what-it-gives)                  |
| the curvature the Gauss–Newton matrix omits                         | what the Laplace approximation's covariance leaves out                           | [the Laplace approximation](../laplace-approximation/#what-the-covariance-leaves-out) |
| the inverse of Fisher scoring's curvature at its limit              | the asymptotic covariance of a maximum-likelihood estimate of parameters         | [Fisher scoring](../fisher-scoring/#two-linearizations)                               |

## Where the vocabularies part

Two rows look like pairs and are not.

- **A change of variables against a change of measure.** An optimizer may reparametrize freely:
  the minimizer of $f$ moves with the variables and is the same point. A density is a density
  against a measure, and its mode is not: the log-normal's mode against Lebesgue measure is
  $e^{m - s^2}$, and against $\dd \point / \point$ it is $e^m$. Every method of this part reads its
  derivatives in charts and still minimizes $f$, the density against the spaces' own measures; a
  Gaussian fitted in a chart is fitted to the chart objective
  [](eq:math/map-estimation/chart-objective), with the chart's volume.
- **The Hessian against the expected information.** An optimizer has one Hessian. An estimator has
  two informations, the observed, which is that Hessian, and the expected, which is its expectation
  over the data. For normals with fixed scales the expected information is the Gauss–Newton matrix,
  and not the Hessian; with a free scale it is neither.
