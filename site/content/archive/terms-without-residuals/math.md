---
summary: How a term that binds no residual enters a query's linearization, by a curvature of its own and its exact gradient, which of three curvatures skoro gives it, what that does to the Laplace approximation, and why the Levenberg–Marquardt method may damp Fisher scoring.
depends_on: [term-curvature, fisher-scoring, levenberg-marquardt, archive/linearization/math, archive/core/math]
---

## The problem

The [linearization entry](../linearization/math.svx) built two assemblies of a query's factored
problem, from the terms' whitened residuals and from the roots of their expected information, and
both refuse a factor that binds no residual
([what is refused](../linearization/math.svx#linearizing-a-query), its second item). The refusal
was right for what that entry set out to do: a factor with no residual would have no rows, so its
curvature would be missing altogether, and nothing in the residual role's contract
[](eq:archive/linearization/math/split) said what its gradient was.

It now stands in the way. One term that is not a normal makes a whole model impossible to
linearize, and with it the sequence that the next entries rest on: a mode, then a Laplace
approximation at it, then Hamiltonian Monte Carlo started from both. The plan's first such model is
eight schools written non-centred, $\theta_j = \mu + \tau \eta_j$, whose every term is a normal but
the half-Cauchy prior on the scale $\tau$. Real pose graphs need robust losses, a Cauchy's or a
Student's $t$ on the measurements, and none of those binds a residual of the normal's kind either.
The uniform distribution on a compact type binds none, and a term written by hand from primitives
binds none.

The entry also takes up a second, smaller gap. `LevenbergMarquardt` refuses the Fisher assembly,
on the ground that its rows are not the curvature of the objective. A model with a free scale, the
calibration case in which the size of a sensor's noise is estimated with everything else, can then
be scored only undamped, from wherever it starts.

Both have one answer, written as a [math note](../../math/term-curvature/): a term enters a
quadratic model through any positive semidefinite curvature and its exact gradient. This document
says which curvature skoro gives a term without a residual, what it costs, what the Laplace
approximation then is, and why damping Fisher scoring is sound.

## Setting

The objects are those of the linearization entry: a query with free continuous quantities $F$, the
point $\hat \point$ of their product at which it is linearized, the stacked tangent
$\R^\dimension$, $f$ the query's negative log-density and $g$ its chart gradient, and for each term
$t$ its variables $K_t = V_t \cap F$ ([](eq:archive/linearization/math/variables)), of tangent
dimension $d_t$. Every derivative is in the charts of the free quantities at $\hat \point$.

A term $t$ *binds no residual* when no node is bound under the residual role on it. Call the set of
such factors $N$. For $t \in N$ nothing splits its log-density into squares, a normalizer and a
log-volume, so the whole of it is the term's part of $f$:

$$
	f = \sum_{t \notin N} \Bigl( \tfrac12 \lvert r_t \rvert^2 - \rho_t \Bigr) + \sum_{t \in N} \phi_t + \text{the constant terms}, \qquad \phi_t = -\logterm_t , @tag(objective)
$$

with $\rho_t$ the remainder of [](eq:archive/linearization/math/split). Write $g_t$ for the chart
gradient of $\phi_t$ and $H_t$ for its chart Hessian [](eq:math/derivatives-in-charts/chart-hessian),
both over the block of $K_t$.

In eight schools the free quantities are $\mu$, the $\eta_j$ and $\tau$, the last positive; the
factors are the normals scoring $\mu$ and the $\eta_j$, the normals scoring each school's estimate
$y_j$ with mean $\mu + \tau \eta_j$ and known scale $\sigma_j$, and the half-Cauchy prior on
$\tau$, the one factor in $N$. Its variables are $\{\tau\}$ and $d_t = 1$.

## Three curvatures, read against skoro

The note gives a term without a residual three possible curvatures
([three curvatures](../../math/term-curvature/#three-curvatures)). Each keeps the gradient exact,
so each leaves the stationary points of $f$, and the mode a minimizer finds, where they were. They
differ in what skoro needs to compute them and in what they say at a mode.

**The term's own Hessian, clipped.** skoro already has what it needs. The core's Hessian pass
([operations](../core/math.svx#operations)) is forward-over-reverse with a velocity frame of
several columns and one cotangent column: seeded with the cotangent $1$ at the term's log-density
and the identity frame on the tangent of $K_t$, it returns $H_t$ in $d_t$ columns and $g_t$ with
it. Nothing is asked of the term but that its operations have order two where the pass needs them
([order](../core/math.svx#order)), which the core already checks before a pass runs and which most
primitives meet. A term from any distribution, or written by hand, is covered.

**The expected information.** A distribution would declare, as the normal does, a root of its
expected information in its own parameters, which the linearizer lifts by the parameters'
Jacobians. That is a new role for each distribution, and a closed form for each: Student's $t$ has
one, $(\nu + d)/(\nu + d + 2)$ times the normal's root [](eq:math/term-curvature/student); the
half-Cauchy as a law of a positive quantity has one too; a hand-written term has none. It needs
first derivatives only, and stays positive semidefinite by construction. It does not know which
measurements are outliers, and its Laplace approximation is not one, for the reason the Fisher
assembly gives none.

**A reweighting of a residual.** A robust distribution would bind its whitened residual and its
loss $\varphi$, and the linearizer would scale its rows by $\sqrt{\varphi'}$
[](eq:math/term-curvature/reweighted). That changes the residual role's contract, since
[](eq:archive/linearization/math/split) would then read $-\varphi(\tfrac12 \lvert r_t \rvert^2) +
n_t - \lambda_t$, and it covers only terms of that form: not the half-Cauchy prior, not the
uniform, not a hand-written term. For the terms it does cover it is the cheapest of the three and
the one that knows the outliers.

> [!DECISION] A term without a residual is given its own Hessian, clipped
> Of the three, only the Hessian covers every term that can be written, needs nothing new of a
> distribution, and has the exact curvature wherever the term is convex in the chart. It is what
> lets eight schools be linearized, and its half-Cauchy prior is convex in the logarithmic chart of
> $\tau$ ([](eq:math/term-curvature/half-cauchy)), so there it costs nothing. The other two are
> declarations a distribution may make later, when one that needs them arrives: the reweighting
> for a robust loss whose outliers dominate the cost of second derivatives, the expected
> information for a free scale of a law that is not the normal. Both fit the same factored problem
> as further roots [](eq:math/term-curvature/root).

## The rows of a term without a residual

For $t \in N$, the rows are the root [](eq:math/term-curvature/eigen-root) of the clipped curvature

$$
	M_t = V \max(\Lambda, 0)\, V\transpose, \qquad V \Lambda V\transpose = \tfrac12 \bigl( H_t + H_t\transpose \bigr) , @tag(clipped)
$$

over the block of $K_t$: $d_t$ rows $C_t$, the right-hand side $-u_t$, and the linear term $\eta_t$,
the part of $g_t$ outside the range of $M_t$. Three things are skoro's to settle about them.

**The number of rows is fixed.** Every row of a batch shares one factor structure and one plan
([batch rows share the plan](../linearization/math.svx#eliminating-the-linearization)), so a term
contributes $d_t$ rows whatever the rank of its curvature at any one batch row: a row of zeros for
each eigenvalue clipped, and the gradient along it in $\eta_t$. Which eigenvalues are clipped is a
matter of numbers, and differs between batch rows; the structure does not.

**The symmetric part is taken.** On the types that are flat in disguise the chart Hessian is the
Hessian of the pull-back everywhere, and symmetric ([quantities](../core/math.svx#quantities)). On
a curved type it differs from that Hessian by a term linear in the term's own gradient
([%full](statement:math/derivatives-in-charts/chart-hessian)), which for one term of many need not
vanish even at a mode of $f$, and need not be symmetric. The curvature is built from its symmetric
part, since only a symmetric matrix has a real eigendecomposition, and the difference is curvature
like any other: it changes the rate, not where the iteration stops. Summed over all terms at a mode,
the chart Hessians are the Hessian of the pull-back of $f$, so the accounting of the Laplace
approximation below is unchanged.

**The constant goes into $c_0$.** $\phi_t(\hat \point)$ is part of $f(\hat \point)$, and the model's
constant [](eq:math/term-curvature/model) is chosen so that $m(0) = f(\hat \point)$ with these terms
included. A term in $N$ with $K_t$ empty, reached from no free quantity, is a constant of the query
and has no rows, as before.

**The passes.** Each term in $N$ needs a Hessian pass seeded at its own log-density, since a pass
takes one cotangent column. Terms whose variables are disjoint share one: seeded with the cotangent
$1$ at each of their log-densities and with velocity columns shared across them, the pass returns
the Hessian of their sum, which is block diagonal over their disjoint variables, so each term's
block is read off its own coordinates. A colouring of the terms in $N$ by shared variables, as the
linearization's tangent pass is seeded by a colouring of the factor structure, gives as many passes
as colours, each with as many velocity columns as the largest $d_t$ of its colour. In eight schools
there is one pass of one column. The passes run at the query's precision, and the eigendecompositions,
of $d_t \times d_t$ blocks, on the host in double precision with the elimination.

**What is refused.** A factor that binds no residual is no longer refused. In its place:

1. **a factor in $N$ whose Hessian pass is not admissible**, an operation of order less than two
   where the pass needs one, naming the step, as the core refuses it;
2. a free discrete quantity, and in the residual assembly a factor whose scale is reached from a
   free quantity, as before.

The second refusal of the residual assembly is about a normal whose scale is free, read off the
scale it binds under the normal's roles. A term in $N$ binds no scale and is not refused for one:
its Hessian is its own, and whatever in it depends on a scale is curvature like any other.

**Both assemblies.** A term in $N$ contributes the same rows to the residual assembly and to the
Fisher assembly. In the Fisher assembly it is then the one term whose curvature is not an expected
information, which [](eq:math/term-curvature/agreement) allows: the model's gradient stays exact and
its matrix positive semidefinite.

## The Laplace approximation

With terms in $N$, the residual assembly's Gaussian at a mode has the precision of the squares'
rows and the terms' clipped Hessians. By
[the Gaussian at a minimum](../../math/term-curvature/#the-gaussian-at-a-minimum), it differs from
the second-order approximation by what the squares omit, as before, and by the clipped part of each
term in $N$. Where no eigenvalue is clipped at the mode, the terms in $N$ add nothing to the error,
and the approximation is as good as it was for a model of normals: in eight schools, the half-Cauchy
prior is never clipped, and its contribution to the precision of $\log \tau$ is exactly
$4y/(1 + y)^2$ at $y = \hat \tau^2 / s^2$.

Where an eigenvalue is clipped at the mode, the term has negative curvature there, which the rows
cannot carry, and the Gaussian is narrower than the second-order one in that direction. This happens
for a robust loss at a measurement far out, $\lvert r \rvert^2 > \nu$ for Student's $t$, which is
exactly the measurement a robust loss exists to discount: the second-order approximation would let
the outlier widen the posterior, and the clipped one ignores it in that direction.

> [!TODO] A Laplace approximation with clipped curvature
> When a term's curvature is clipped at the point of a Laplace approximation, should the
> approximation be refused (a `NotAnIntegral` kind of refusal, naming the term), given and
> reported with the clipped directions, or given silently? The draft leans toward giving it and
> reporting which terms were clipped, since the error has a known sign (over-confidence in the
> clipped directions) and the most common cause is an outlier under a robust loss.

> [!TODO] The Newton step and the observed information
> The same Hessian passes, run on every term rather than on those in $N$, give the observed
> information and a Newton step, and with them a Laplace approximation that keeps the residual
> curvature Gauss–Newton drops. The `EVALUATION.md` survey places that variant in this entry.
> Its matrix may be indefinite even at a mode once taken term by term, so it would need either a
> clipping of its own or an elimination of a symmetric matrix rather than of rows, which the
> elimination crate does not have. Is it in scope here, or a later entry?

## Damping Fisher scoring

`LevenbergMarquardt` needs from a model three things: its gradient at zero, for the damped step and
its promise [](eq:math/term-curvature/promised); its rows, for Marquardt's diagonal; and the
objective $f$ at a trial point, for the gain. The Fisher assembly gives all three exactly: its
gradient is $g$ ([](eq:math/fisher-scoring/scoring-step) and the argument before it), its rows are
rows, and the objective is the query's. Its matrix being the expected information and not the
Hessian changes neither the promise nor the meaning of the gain; it changes only how near one the
gain is, by [%full](statement:math/term-curvature/gain).

And there it is the better of the two assemblies. On a free scale the residual rows hold half the
curvature, $H^\ast = 2M$ in the scale's direction, and their undamped gain tends to zero, so the
schedule never releases the damping; the residual assembly refuses a free scale in any case. The
Fisher rows hold all of it for an isotropic scale, $H^\ast = M$, and the gain tends to one. Damped
Fisher scoring is therefore the Levenberg–Marquardt method on the Fisher assembly, with nothing
changed but the refusal lifted, and the end-of-run check
([what is checked at the end](../../math/levenberg-marquardt/#what-is-checked-at-the-end)) becomes
the Fisher decrement $\tfrac12 g\transpose M^{-1} g$, which is zero exactly where $g$ is.

What stays refused of the Fisher assembly is what it never gave: an evidence, draws and a
replacement, which approximate an integral of the density.

## In skoro's terms

- **What is random and what is differentiated.** Nothing changes in what a query observes or
  leaves free. The terms in $N$ are differentiated twice with respect to the free continuous
  quantities they reach, in the charts of those quantities at $\hat \point$; every other term, once,
  as before.
- **Which graphs.** Which terms are in $N$ is read off the bindings, which terms share variables
  off the probabilistic graph with the query's observed quantities removed, and whether a Hessian
  pass is admissible off the compilation graph, by the core's rule of order. None depends on a
  value, so the factor structure, the colouring and the plan are shared by every row of a batch.
- **The linearization entry, revised.** Its second refusal is withdrawn from both assemblies, and
  the minimizer's refusal of the Fisher assembly is lifted. Its rows, its right-hand sides, its
  linear term from the log-volumes and its elimination are unchanged for every term that binds a
  residual. A model in which every factor binds one is linearized exactly as before.
- **The tests' models.** The uniform on the rotations, whose log-density is constant, contributes
  $d_t$ rows of zeros and no gradient. A hand-written half-Cauchy on a positive quantity, which the
  core can build from primitives, has the curvature [](eq:math/term-curvature/half-cauchy), and
  linearizing it alone can be checked against that closed form. A hand-written Student's $t$ on a
  scalar residual is clipped beyond $\lvert r \rvert^2 = \nu$, which a test can place on either side.

## Open questions

> [!TODO] Which terms the entry brings
> The entry needs a term without a residual to linearize, and it can test with terms written by
> hand. Should it also bring the half-Cauchy and Student's $t$ as distributions, or leave each to an
> entry of its own (the half-Cauchy to the Hamiltonian Monte Carlo entry, whose example uses it)?
