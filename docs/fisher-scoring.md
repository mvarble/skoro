# Fisher Scoring

This note explains the method that skoro's `information` and `information_root` functions exist to
support. It begins with Newton's method and the reason it fails far from an optimum, introduces the
Fisher information and the identities that make it useful, and then shows why replacing the Hessian
with the information matrix produces a reliable algorithm. Two small models are worked through
numerically. The final sections connect the mathematics to skoro's `FISHER_ROOT` bindings, derive
the rule by which the information of a many-term model is assembled, and state the algorithm and its
limitations.

Throughout, $\theta \in \mathbb R^D$ denotes the free parameters, $y$ denotes data, and
$\ell(\theta) = -\log p(y, \theta)$ denotes the negative log-density that we wish to minimize. For
maximum a posteriori estimation, $\ell$ includes the negative log-prior. Gradients and Hessians are
taken with respect to $\theta$ unless stated otherwise.

## 1. Newton's Method

The estimate we seek is the minimizer $\hat\theta = \arg\min_\theta \ell(\theta)$. Newton's method
approaches it by replacing $\ell$ near the current iterate with its second-order Taylor expansion,

$$
\ell(\theta + \delta) \approx \ell(\theta) + g^\top \delta + \tfrac12\, \delta^\top H\, \delta,
\qquad g = \nabla \ell(\theta), \quad H = \nabla^2 \ell(\theta),
$$

and moving to the minimizer of that quadratic. Setting the derivative of the right-hand side with
respect to $\delta$ to zero gives the Newton step

$$
H\, \delta = -g, \qquad \theta \leftarrow \theta + \delta.
$$

When the iterate is already close to $\hat\theta$ and $H$ is positive definite there, the method
converges quadratically: the number of correct digits roughly doubles with each step. Its difficulty
lies elsewhere. The quadratic model has a minimizer only when $H$ is positive definite, and the step
decreases $\ell$ to first order only when

$$
g^\top \delta = -g^\top H^{-1} g < 0 ,
$$

which is guaranteed for every $g$ exactly when $H \succ 0$. Far from the optimum the Hessian of a
nonlinear model is frequently indefinite, and the Newton step then points toward a maximum or a
saddle point instead of a minimum. Computing $H$ also requires second derivatives of the entire model,
and an indefinite matrix cannot serve as a metric, which some methods (Section 8) require. Fisher
scoring addresses the first and last of these problems by replacing $H$ with a matrix that is
positive semidefinite by construction.

## 2. The Fisher Information

Consider a statistical model $p(y \mid \theta)$ for data $y$. Its **score** is the gradient of the
log-likelihood,

$$
s(y; \theta) = \nabla_\theta \log p(y \mid \theta),
$$

and its **Fisher information** is the matrix

$$
\mathcal I(\theta) = \mathbb E_{y \sim p(\cdot \mid \theta)}\!\left[ s(y;\theta)\, s(y;\theta)^\top \right].
$$

The expectation is taken over data drawn from the model itself at the parameter value $\theta$, not
over the data actually observed. Two identities follow from the single fact that a density integrates
to one.

**Proposition 1.** Under the usual regularity conditions permitting differentiation under the
integral sign,

$$
\mathbb E[s] = 0
\qquad\text{and}\qquad
\mathcal I(\theta) = \mathbb E\!\left[-\nabla_\theta^2 \log p(y \mid \theta)\right].
$$

*Proof.* Differentiating $\int p(y \mid \theta)\, dy = 1$ with respect to $\theta$ gives
$\int \nabla_\theta p \, dy = 0$. Since $\nabla_\theta p = p\, \nabla_\theta \log p = p\, s$, this
reads $\int s\, p\, dy = 0$, which is the first identity. Differentiating $\int s\, p\, dy = 0$ once
more, and using $\nabla_\theta p = p\, s$ again, gives

$$
\int \left(\nabla_\theta^2 \log p\right) p\, dy + \int s\, s^\top p\, dy = 0 ,
$$

which rearranges to the second identity. $\blacksquare$

The second identity is the one that matters for optimization, because it offers two readings of the
same matrix. On the one hand, $\mathcal I(\theta)$ is the Hessian of the negative log-likelihood
averaged over the data the model would generate at $\theta$; it is what $H$ would be on average if the
model were correct. On the other hand, $\mathcal I(\theta)$ is the covariance matrix of the score, and
a covariance matrix is always positive semidefinite. The information therefore carries the curvature
information of the Hessian while being immune to the indefiniteness that afflicts the Hessian of a
particular data set.

## 3. Fisher Scoring

**Fisher scoring** is Newton's method with the Hessian replaced by the information:

$$
\mathcal I(\theta)\, \delta = -g, \qquad \theta \leftarrow \theta + \delta .
$$

That this substitution is sound can be seen from four complementary arguments.

The first is that every Fisher step is a descent direction. If the data identify every direction of
$\theta$, so that $\mathcal I(\theta) \succ 0$, then $g^\top \delta = -g^\top \mathcal I^{-1} g$ is
negative whenever $g \neq 0$. A sufficiently short step along $\delta$ therefore decreases $\ell$. The
full step may still be too long, as the first example below shows, but that is a question of step
length, which damping or a line search resolves; the direction itself is never uphill.

The second argument concerns speed. Write the Hessian as $H = \mathcal I + (H - \mathcal I)$. By
Proposition 1 the discrepancy $H - \mathcal I$ has mean zero when the data are drawn from the model,
and with informative data it is small near the optimum. Fisher scoring viewed as the fixed-point
iteration $\theta \leftarrow \theta - \mathcal I^{-1} g$ has, at $\hat\theta$, the iteration Jacobian
$\mathcal I^{-1}(\mathcal I - H)$. Its convergence is therefore linear, with a rate that becomes fast
precisely when $H$ and $\mathcal I$ agree. For an exponential family in its canonical parameterization,
such as Poisson or logistic regression with the canonical link, the Hessian does not depend on the
data at all, so $H = \mathcal I$ identically and Fisher scoring coincides with Newton's method.

The third argument shows what the substitution discards in the most common case, a model with Gaussian
errors. Suppose $y_i \sim \mathcal N(f_i(\theta), \sigma^2)$ independently, and define the whitened
residuals $r_i = (y_i - f_i(\theta))/\sigma$ with Jacobian $J = \partial r / \partial \theta$. Then

$$
\ell = \tfrac12 \sum_i r_i^2 + \text{const}, \qquad
g = J^\top r, \qquad
H = J^\top J + \sum_i r_i\, \nabla^2 r_i .
$$

**Proposition 2.** For this model, $\mathcal I(\theta) = J^\top J$, so Fisher scoring is the
Gauss–Newton method.

*Proof.* The matrix $J$ depends only on $\theta$, and $\nabla^2 r_i = -\nabla^2 f_i / \sigma$ does not
depend on $y$. Under the model, $\mathbb E[r_i] = 0$. Taking the expectation of $H$ therefore removes
the second term and leaves $J^\top J$, which by Proposition 1 is $\mathcal I$. $\blacksquare$

The discarded term $\sum_i r_i \nabla^2 r_i$ is exactly the part of the Hessian that grows with the
residuals, and it is the source of the indefiniteness observed far from an optimum. Fisher scoring
keeps the part of the curvature that the model explains and drops the part that is, on average, zero.

The fourth argument is geometric. For a small step, the Kullback–Leibler divergence between the model
at $\theta$ and at $\theta + \delta$ satisfies

$$
\mathrm{KL}\!\left(p_\theta \,\middle\|\, p_{\theta+\delta}\right)
= \tfrac12\, \delta^\top \mathcal I(\theta)\, \delta + O(\|\delta\|^3).
$$

The information is thus the metric that measures how far a step moves the *distribution*, as opposed to
how far it moves the parameter vector. Among all steps that change the distribution by a given small
amount, the Fisher step is the one that decreases $\ell$ fastest; it is the steepest-descent direction
in this metric, also called the natural gradient. A consequence is that the method does not depend on
how the model is parameterized, whereas the ordinary gradient does.

What is given up in exchange is exact agreement with Newton's method when the model is misspecified or
the residuals remain large at the optimum. In those circumstances $H$ and $\mathcal I$ differ even at
$\hat\theta$, and convergence, while still reliable, is only linear.

## 4. Example: A Nonlinear Mean

Consider three observations $y_i \sim \mathcal N(e^{\theta t_i}, 1)$ at $t = (0, 1, 2)$ with values
$y = (1, 2.5, 7)$. Writing $f_i = e^{\theta t_i}$ and $r_i = y_i - f_i$, the derivatives of the mean
are $f_i' = t_i f_i$ and $f_i'' = t_i^2 f_i$, and

$$
\ell = \tfrac12 \sum_i r_i^2, \qquad
g = -\sum_i r_i f_i', \qquad
H = \sum_i (f_i')^2 - \sum_i r_i f_i'', \qquad
\mathcal I = \sum_i (f_i')^2 .
$$

The last expression follows from Proposition 2, since $\mathbb E[r_i] = 0$. The minimizer is
$\hat\theta = 0.971048$. At the starting value $\theta = 0.5$ the residuals are
$(0,\ 0.851,\ 4.282)$, and the large third residual multiplied by $f_3''$ dominates the Hessian:
$H = -15.68$, whereas $\mathcal I = 32.27$. The two iterations proceed as follows.

| Iteration | Newton $\theta$ | Newton $\ell$ | Fisher $\theta$ | Fisher $\ell$ |
| --- | ---: | ---: | ---: | ---: |
| 0 | 0.5000 | 9.53 | 0.5000 | 9.53 |
| 1 | −1.0736 | 26.02 | 1.2647 | 15.93 |
| 2 | −1.6938 | 26.95 | 1.0423 | 0.598 |
| 3 | −2.3684 | 27.33 | 0.97593 | 0.0127 |
| 4 | −3.1224 | 27.50 | 0.971081 | 0.0103 |
| 5 | −3.9654 | 27.58 | 0.971048 | 0.0103 |

Because $H < 0$ at the start, Newton's method moves away from the optimum and never recovers. Fisher
scoring moves in the correct direction from the first step, but that step is too long: $\ell$ rises
from 9.53 to 15.93 as the iterate overshoots to 1.2647. The subsequent steps converge rapidly, and near
the optimum the two curvatures nearly agree, with $H = 201.11$ and $\mathcal I = 201.48$, so the final
iterations behave like Newton's method. The overshoot of the first step is the kind of step-length
error that the damping of Section 7 removes.

## 5. Example: A Scale Parameter

The second example concerns a scale rather than a location. Let $y_i \sim \mathcal N(\mu, s^2)$ with
$\mu = 2$ known, and estimate $s$ from the four observations $(1.8,\ 2.4,\ 1.2,\ 2.6)$. With $n = 4$
and $S = \sum_i (y_i - \mu)^2 = 1.2$, the maximum-likelihood estimate is $\hat s = \sqrt{S/n} = 0.5477$.
Up to a constant,

$$
\ell(s) = n \log s + \frac{S}{2 s^2}, \qquad
g = \frac{n}{s} - \frac{S}{s^3}, \qquad
H = -\frac{n}{s^2} + \frac{3S}{s^4} .
$$

Since $\mathbb E[S] = n s^2$ under the model, the information is
$\mathcal I = -n/s^2 + 3n s^2 / s^4 = 2n/s^2$, which is positive for every $s$. The Hessian, by
contrast, is negative whenever $S < n s^2 / 3$, that is, whenever the current guess for $s$ is too
large. Starting from $s = 3$ gives $H = -0.40$ and $\mathcal I = 0.89$.

| Iteration | Newton $s$ | Fisher $s$ |
| --- | ---: | ---: |
| 0 | 3.000 | 3.000 |
| 1 | 6.222 | 1.550 |
| 2 | 12.54 | 0.8718 |
| 3 | 25.13 | 0.6080 |
| 4 | 50.29 | 0.55071 |
| 5 | 100.6 | 0.547731 |
| 6 | 201.2 | 0.547723 |

Newton's method doubles $s$ at every step and diverges, while Fisher scoring decreases $\ell$ at every
step and converges to $\hat s$. At the estimate the two curvatures coincide exactly, $H = \mathcal I =
26.667$, because $S = n \hat s^2$ there.

This example is precisely the covariance block of skoro's Gaussian Fisher root in one dimension. A
`Normal` with covariance factor $L = s$ has whitening $W = L^{-1} = 1/s$, and its root over the
arguments $(x, \mu, L)$ is

$$
C = \begin{bmatrix} W & -W & 0 \\ 0 & 0 & \sqrt2\, W \end{bmatrix},
\qquad
C^\top C = \begin{bmatrix}
\ \ 1/s^2 & -1/s^2 & 0 \\
-1/s^2 & \ \ 1/s^2 & 0 \\
0 & 0 & 2/s^2
\end{bmatrix}.
$$

The entry for $L$ is $2/s^2$ for a single observation, and summing over the $n$ observations recovers
$\mathcal I = 2n/s^2$. The entry for $\mu$ is the familiar precision $1/s^2$ of a Gaussian mean.

## 6. Models with Many Terms

A skoro model's log-density is a sum of terms, $\log p = \sum_t \log p_t(q_t \mid \mathrm{pa}_t)$,
where term $t$ scores a quantity $q_t$ given its parents $\mathrm{pa}_t$. Each term declares, through
its bindings, the arguments $a_t$ over which its local information is expressed (the `ARGUMENT` role;
for a `Normal` these are $[x, \mu, L]$) and a Fisher root $C_t$ (the `FISHER_ROOT` role) satisfying

$$
C_t^\top C_t = \mathcal I_t(a_t),
$$

where the expectation defining $\mathcal I_t$ is over $q_t$ given its parents. Assembling the
information of the whole model over the free parameters requires two results: one that moves a term's
information from its arguments to the free parameters, and one that combines the terms.

**Proposition 3 (change of variables).** If the arguments depend on the free parameters through
$a_t = \phi_t(\theta)$, with Jacobian $J_t = \partial a_t / \partial \theta$, then the information of
term $t$ over $\theta$ is $J_t^\top \mathcal I_t J_t = (C_t J_t)^\top (C_t J_t)$.

*Proof.* By the chain rule the score with respect to $\theta$ is $J_t^\top s_{a_t}$, where $s_{a_t}$
is the score with respect to the arguments. Because $J_t$ does not depend on $q_t$, its outer product
has expectation $J_t^\top\, \mathbb E[s_{a_t} s_{a_t}^\top]\, J_t = J_t^\top \mathcal I_t J_t$.
$\blacksquare$

**Proposition 4 (additivity).** The information of the model over $\theta$ is the sum of the terms'
informations.

*Proof.* The score of the model is $s = \sum_t s_t$, so $\mathbb E[s s^\top]$ is the sum of the terms
$\mathbb E[s_t s_u^\top]$. Order the terms so that each comes after the terms that generate its
parents, and take $t$ earlier than $u$. Conditionally on all quantities generated before $q_u$, the
score $s_u$ has mean zero by Proposition 1, while $s_t$ is a function of those earlier quantities. By
the tower property of conditional expectation, $\mathbb E[s_t s_u^\top] = 0$. Only the terms with
$t = u$ survive. $\blacksquare$

Combining the two propositions, the information of the model is

$$
\mathcal I(\theta) = \sum_t J_t^\top C_t^\top C_t\, J_t = A^\top A,
\qquad
A = \begin{bmatrix} C_1 J_1 \\ \vdots \\ C_T J_T \end{bmatrix}.
$$

The matrix $A$ is a square root of the information, obtained by stacking each term's root composed with
its Jacobian. The correspondence with skoro's functions is direct.

| Quantity | skoro | Shape |
| --- | --- | --- |
| $J_t$ | `jacobian_vjp(client, points, free, &arguments)` | `[batch, Σ dim(a_t), D]` |
| $C_t J_t$ | one `InformationBlock` of `information_blocks` | `[batch, k_t, D]` |
| $A$ | `information_root(client, points, free)` | `[batch, Σ k_t, D]` |
| $A^\top A$ | `information(client, points, free)` | `[batch, D, D]` |

Two conventions deserve mention. When a stochastic quantity is itself among the free parameters, as in
maximum a posteriori estimation of a latent state, its term still takes the expectation over that
quantity; for a Gaussian prior this yields the prior precision, which is exactly that term's
contribution to the curvature. A free-standing joint factor, on the other hand, generates no quantity
over which an expectation could be taken, and its contribution is by convention the Gauss–Newton matrix
$W^\top W$ of its whitened residual, in the sense of Proposition 2.

## 7. The Algorithm

In practice Fisher scoring is combined with damping. Given a current iterate, a damping parameter
$\lambda \geq 0$, and a positive diagonal scaling $D$, one iteration computes the gradient of the
log-density, assembles the information root, solves the damped system, and takes a step on the
parameter manifold:

$$
\left(A^\top A + \lambda D\right) \delta = \nabla_\theta \log p(\theta),
\qquad
\theta \leftarrow \operatorname{retract}(\theta, \delta).
$$

The right-hand side is the gradient of $\log p$ and not of $\ell$, because skoro differentiates the
log-density; the resulting step is the same as $-\mathcal I^{-1} \nabla \ell$ when $\lambda = 0$. The
gradient is available from `log_density_ord1`, the root from `information_root`, and the step from the
retraction of the free quantities (`PointMap::retract_flat`). After each step, $\lambda$ is decreased
if $\log p$ improved and increased otherwise. A large $\lambda$ turns the step into a short gradient
step, and a small one recovers Fisher scoring, so this adaptive scheme is the Levenberg–Marquardt
method with the information in place of the Gauss–Newton matrix. It is what removes the overshoot seen
in Section 4.

The linear system is solved on the host with `skoro-dense`, since the backend has no dense
factorizations. In general the system is symmetric positive definite and is solved by a Cholesky
factorization (`spd_solve`). For models whose terms are Gaussian with fixed covariances there is a
better-conditioned route: $A$ is then the whitened residual Jacobian $J_r$ and
$\nabla \log p = -J_r^\top r$, so the damped system is the normal equations of the least-squares
problem

$$
\min_\delta \left\| \begin{bmatrix} J_r \\ \sqrt{\lambda}\, D^{1/2} \end{bmatrix} \delta
+ \begin{bmatrix} r \\ 0 \end{bmatrix} \right\|^2 ,
$$

which can be solved by a QR factorization (`lstsq`). Working with the root $A$ rather than the product
$A^\top A$ matters here, because the condition number of $A^\top A$ is the square of that of $A$.

## 8. Limitations

The information is an expected curvature, and its usefulness rests on the model being a reasonable
description of the data. When the model is misspecified, or when residuals remain large at the optimum,
$H$ and $\mathcal I$ differ at $\hat\theta$ and the final convergence is linear, as discussed in
Section 3.

If the data do not constrain some direction of $\theta$, the information is singular along that
direction and the undamped system has no unique solution. Damping restores a well-posed system, but a
persistently singular information matrix indicates that the parameters are not identifiable from the
data, which is a modeling issue that no solver can remove.

For quantities on curved manifolds, skoro's Gaussian root uses the block $[W, -W]$ for $(x, \mu)$. This
block assumes that the derivatives of $v = x \ominus \mu$ with respect to $x$ and $\mu$ are $I$ and
$-I$, which holds exactly on flat spaces. On groups such as SO(3) and SE(3) the derivatives equal the
identity only at $v = 0$. For SO(3) they are $J_r^{-1}(v) = I + \tfrac12 [v]_\times + O(\|v\|^2)$ and
$-\left(I - \tfrac12 [v]_\times\right) + O(\|v\|^2)$, and SE(3) adds further terms linear in $v$ from
its translation block.

The resulting error is smaller than this first-order discrepancy suggests. Under the model, $v$ is a
Gaussian with mean zero, so its odd moments vanish, and every correction that is linear in $v$ averages
out of the expectation that defines the information. The root is therefore exact on flat spaces and
correct to second order in the dispersion on groups: its relative error grows in proportion to
$\|\Sigma\|$. This is the same order as the approximation already present in a Gaussian on a group,
whose density omits the volume factor $\det J_r(v) = 1 - \|v\|^2/12 + \cdots$ of the retraction. When
$\Sigma$ is isotropic the corrections vanish identically, because each of them applies $[v]_\times$ to a
multiple of $v$.

A Monte Carlo experiment confirms this scaling. Take $\Sigma = \sigma^2 L_0 L_0^\top$ for a fixed and
deliberately skewed triangle $L_0$, and compare $C^\top C$ with the covariance of the model's own score
over 100,000 draws. Sharing the random draws with a flat control cancels the sampling noise, and gives
the following relative errors.

| $\sigma$ | SO(3) | SE(3) |
| --- | ---: | ---: |
| 0.1 | 0.012 | 0.016 |
| 0.2 | 0.048 | 0.064 |
| 0.4 | 0.189 | 0.255 |

Each doubling of $\sigma$ multiplies the error by very nearly four, as second-order accuracy predicts.
The error is a percent or two for rotational noise of a tenth of a radian, and it becomes material only
for widely dispersed distributions, where a Gaussian on the group is itself a poor model. An optimizer
that starts far from its solution, and therefore sees large residuals, can still use the exact
Gauss–Newton curvature at its current point, which the Jacobians of the terms' `RESIDUAL` bindings
provide. That matrix is not an expected information, which is why the Fisher root does not carry it.

Finally, the information matrix has uses beyond optimization. Riemannian manifold Hamiltonian Monte
Carlo uses $\mathcal I(\theta)$, augmented by the curvature of the prior, as a position-dependent mass
matrix. Its dynamics involve the derivatives $\partial \mathcal I / \partial \theta$, which is why skoro
keeps the Fisher root as a differentiable node in the model graph rather than assembling it outside the
graph.
