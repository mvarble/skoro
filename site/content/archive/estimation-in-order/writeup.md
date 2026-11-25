---
summary: The part of the mathematics on least squares and Gaussian approximations begins with the estimate sought, the mode of a posterior, and the methods that find it, reads the linearization both as a trust-region model and as a Gaussian factor on the step, and comes to elimination last, as the way to solve the normal equations that both readings end in.
---

## Context

### The part as it stands

The [mathematics in order](../mathematics-in-order/writeup.md) set out the third part of the
mathematics, *Least squares and Gaussian approximations*, as seven notes. Read in order they are

1. [elimination in square-root form](../../math/square-root-elimination/),
2. [the improper Gaussian factor](../../math/improper-gaussian/),
3. [linearization and the Gauss–Newton step](../../math/linearization/),
4. [the Levenberg–Marquardt method](../../math/levenberg-marquardt/),
5. [the Laplace approximation](../../math/laplace-approximation/),
6. [Fisher scoring](../../math/fisher-scoring/),
7. [calibration of credible sets](../../math/calibration/).

Every note rests only on the notes before it, and the order is still not one in which the part
explains itself. Its first note opens with a function $\tfrac12 \lvert A \tangent - b \rvert^2 +
\eta\transpose \tangent$ and says that "Gauss–Newton and Levenberg–Marquardt steps, Laplace
approximations and Kalman filters all end in" it, three notes before any of them is defined. It
goes on to Householder reflections, triangular roots and junction trees, the longest note of the
part, and a reader who comes to the part from the probability before it has not yet been told
what is being estimated, why the estimate is a minimum, or why a minimum of a nonlinear function
reduces to this one. The note is first because it depends on nothing, which is the order's rule
satisfied and not its purpose.

### What the part does not say

Three ideas the part relies on have no place in it.

- **The estimate.** No note says that the thing sought is the mode of a posterior density, the
  maximum a posteriori estimate, or that a mode is a mode against a reference measure, so that it
  moves with the measure. [Models built forward](../../math/generative-models/) mentions the term in
  passing; [the Laplace approximation](../../math/laplace-approximation/#the-integral-in-a-chart)
  and [the linearization](../../math/linearization/#a-sum-of-squares-and-what-else) each meet the
  measure's log-volume on their own.
- **Newton's method and trust regions.** The part has Gauss–Newton and Levenberg–Marquardt, and
  not the method they approximate nor the frame in which the second is one instance. That a
  curvature sets an iteration's rate and not its limit is proved for Gauss–Newton alone, in
  [the linearization](../../math/linearization/#what-the-iteration-converges-to), and
  [Fisher scoring](../../math/fisher-scoring/) needs it for another curvature. The gain ratio,
  the measure of how far a model is trusted, is introduced inside
  [Levenberg–Marquardt](../../math/levenberg-marquardt/#the-gain-ratio) as though it were particular
  to it.
- **That the two readings of a linearization are one.** The linearization is a quadratic model
  to be minimized, and $e^{-m}$ is a Gaussian in the step; the first is the optimizer's reading and
  the second the state estimator's. The part uses both, in different notes and different words, and
  never says where the words correspond.

## Decision

The part is reordered to tell one story: what is estimated, how such an estimate is found in
general, why a model of normals makes the finding a least-squares problem, what the linearization
of that problem is under each reading, the linear system both readings end in, and last, how to
solve that system well. Elimination keeps everything it holds and moves from the first place to
the seventh.

### The notes, and their order

| | note | | holds | rests on |
| --- | --- | --- | --- | --- |
| 1 | `map-estimation` | new | the posterior density of the unobserved quantities against a reference measure; its mode as the minimizer of $f$, the negative log-density; the mode's dependence on the measure, which is where log-volumes enter; $f$ as a sum of terms, each reading a few variables | generative-models, chart-volume |
| 2 | `trust-regions` | new | the quadratic model of a smooth function in the charts at a point; the Newton step and its quadratic convergence at a strict minimum; any positive semidefinite curvature with the exact gradient, converging at a linear rate set by the curvature, to a limit set by the gradient alone; the trust-region step, its characterization as $(M + \lambda D)\, \tangent = -g$ with $\lambda \ge 0$ tied to the radius; the gain ratio and the radius's update | derivatives-in-charts, product-manifold |
| 3 | `nonlinear-least-squares` | new | a model of normals whose scales are fixed and whose means are computed from other quantities: its $f$ is $\sum_t \tfrac12 \lvert r_t \rvert^2 + \lambda_t$ plus a constant, so its mode is a nonlinear least-squares problem; a free scale breaks the form | normal, map-estimation |
| 4 | `linearization` | revised | the linearization of the residuals, its agreement with $f$ to first order, and what it leaves out of the Hessian; Gauss–Newton as the step of 2 with the curvature $J\transpose J$, and its convergence as an instance of 2's; the trust-region step on the Gauss–Newton model as the damped step; where the linearization is exact | nonlinear-least-squares, trust-regions, chart-volume |
| 5 | `improper-gaussian` | revised, from 2 | the factor, when it is a normal, the two examples worked by hand, its forms, how factors add, its derivatives, as now; opened by a new instance: the linearization is a factor in the increment $\delta = \point \ominus \hat \point$ at the anchor $\hat \point$, whose mean, when it has full rank, is the Gauss–Newton step | normal, linearization |
| 6 | `normal-equations` | new | the step, the damped step and the factor's mean each solve $A\transpose A\, \tangent = A\transpose b - \eta$; $A\transpose A$ as the precision and its determinant in the integral; what forming $A\transpose A$ costs, $\kappa^2$ and the example that loses everything; why a linear term costs $\kappa^2$ and a log-volume is folded into the rows | improper-gaussian |
| 7 | `square-root-elimination` | revised, from 1 | as now, from Householder reflections on, opened by the normal equations it solves without forming | normal-equations |
| 8 | `integrating-out` | new, from 5 | what elimination leaves: integrating a block out of a sum of squared rows and linear terms leaves an improper factor on its neighbours, of any rank, exactly | improper-gaussian, square-root-elimination |
| 9 | `kalman-filter` | new, from 5 | the filter as that integral repeated along a chain of states: the update, the prediction, the gain and the evidence; the filter in information form, which carries a state that is not yet determined and whose covariance does not exist | integrating-out |
| 10 | `levenberg-marquardt` | revised, from 4 | the method as an instance of 2 on the Gauss–Newton model: Marquardt's diagonal, the schedule, the stopping rule and the checks at the end | linearization, trust-regions, square-root-elimination |
| 11–13 | `laplace-approximation`, `fisher-scoring`, `calibration` | order and references | as now, citing 1 for the measure a mode is taken against and 2 for the rate a curvature sets | as now, with 1 and 2 where they are used |
| 14 | `optimization-and-estimation` | new | a table of the two vocabularies, closing the part, a row for each thing the part names twice: the objective and the negative log-density, the step and the factor's mean, the curvatures and the informations, the trust region and the prior on the step, the normal equations and the information form, the filter's gain and innovation, the covariance of an approximation | improper-gaussian, trust-regions, kalman-filter, levenberg-marquardt, laplace-approximation, fisher-scoring |

The order puts elimination after the normal equations, and two of the notes before it rest on
elimination today. Each loses that dependence as follows.

- **The linearization** names its problem as elimination's factored problem
  [](eq:math/square-root-elimination/factored), and argues for folding a log-volume into the rows
  by the condition numbers of [linear terms](../../math/square-root-elimination/#linear-terms).
  The identification is reversed: elimination, coming later, says that the linearization is an
  instance of its factored problem. The argument for folding moves to `normal-equations`: the
  part of a solution that comes from a linear term is $(A\transpose A)^{-1} \eta$, as sensitive
  as a solve of the normal equations whatever method factors $A$, and that can be said before any
  such method is. *Linear terms* keeps its own statement of the cost, and the fold of a linear
  term $A\transpose c$ into $b$, which the archive cites.
- **The improper Gaussian factor** rests on elimination for one section, *What elimination
  leaves*, and the Kalman filter in it. The section becomes the eighth note, and the filter the
  ninth. The rest of the note,
  the factor, its forms (whose orthogonal reduction needs only that an orthogonal matrix keeps
  lengths) and accumulation, needs no elimination, and the linearization takes the place of the
  integral as the instance that opens it.

### Two readings of one linearization

Notes 4 and 5 say the same thing twice, on purpose. The linearization at $\hat \point$ is

$$
	m(\tangent) = \tfrac12 \sum_t \lvert r_t + \hat c_t + J_t \tangent \rvert^2 + \eta\transpose \tangent + c_0 ,
$$

and note 4 reads it as an optimizer does: a quadratic model of $f$ near $\hat \point$, trusted only
near $\hat \point$, whose unconstrained minimizer is the Gauss–Newton step and whose minimizer
within a radius is the damped step. The Gauss–Newton step is the trust-region step with the radius
infinite; it is not itself a trust-region method, and Levenberg–Marquardt is the method that makes
it one.

Note 5 reads it as a state estimator does: $e^{-m}$ is an improper Gaussian factor in $\delta =
\point \ominus \hat \point$, with the rows $J_t$, the right-hand sides $-(r_t + \hat c_t)$ and the
linear term $\eta$. When its rows have full rank it is a normal, its mean is the Gauss–Newton step
and its precision is the Gauss–Newton matrix. A damping is then one more factor, a normal prior on
$\delta$ centred at zero with the precision $\lambda D$, and the damped step is the posterior mean
under it.

Neither reading is the more fundamental. The optimizer's explains convergence and the gain ratio;
the estimator's explains marginals, accumulation and the Laplace approximation. The part needs
both, and a reader who knows one vocabulary should find the other's word for each thing. The
part closes with `optimization-and-estimation`, a table that sets the two side by side, a note of
its own so that it can be found by its title and returned to. Its rows begin with what notes 4
and 5 name twice: the objective and the negative log-density, the
minimizer and the mode, the model's Hessian and the precision or information, its inverse and the
covariance, the gradient and the negative of the score, the step and the factor's mean, the trust
radius or the damping and a prior on the step, the gain ratio and the fit of the factor to the
density. The curvatures need the most care, since telling them apart is what Fisher scoring is
for: the Hessian of $f$ is the observed information, and the Gauss–Newton matrix is the expected
information, the two agreeing for a normal of fixed scale. Each row says which note defines each
of its two terms. Notes never name skoro, and the table names only general terms.

The rows go on through the notes after the two readings. The normal equations are the information
form of a Gaussian, their matrix its precision and their right-hand side its information vector.
The Kalman filter's update is the step of a linear model, its gain the weight the update gives the
innovation, which is a residual at the prediction, and its prediction an integral of
`integrating-out`. Damping is a prior on the step, the Laplace approximation's covariance the
inverse of a curvature at the mode, and Fisher scoring's matrix the expected information.

The table closes the part because a note rests only on the notes before it. Placed where the two
readings meet, after the fifth note, it could hold only the rows of those five, and would leave out
the information form, the filter's words and the approximations' covariances, which are where
the vocabularies are most often mixed. Placed last, it rests on every note it translates and has
a row for every pair. Until the reader reaches it, each note that brings a word from one
vocabulary says in its own text what the other calls it, so that the table gathers what the notes
already say and adds nothing new.

### The normal equations, then elimination

The `normal-equations` note gathers what both readings end in. The step of note 4, the damped step,
and the mean of note 5's factor all solve one system,

$$
	A\transpose A\, \tangent = A\transpose b - \eta ,
$$

with $A\transpose A$ the precision, whose determinant is the integral's normalizer. The note then
says what forming $A\transpose A$ costs: its condition number is $\kappa(A)^2$, and the example of
elimination's first section, in which $1 + 10^{-16}$ rounds to $1$ and the matrix is singular as
computed, moves here with that argument. So does the cost of a linear term, which no method that
works with $A$ alone escapes, and with it the reason the linearization folds a log-volume into its
rows; elimination's *Linear terms* keeps the same cost as it arises in the triangular root.
Elimination then opens with the question the note leaves, how to solve the system without forming
it, and answers it as it does now.

### What stays where it is

The archive cites this part by name, and every name it cites stays in its note.
[Elimination](../../math/square-root-elimination/) keeps its tags `factored`, `complete-square`,
`integral`, `replace`, `clique-chain` and `damping`, its statements `triangular-root`,
`eliminate-one` and `marginal`, and its sections *Linear terms*, *Eliminating one block*, *Order,
fill and the elimination tree*, *Cliques and the junction tree* and *What the eliminated problem
answers*. The pages cite [Levenberg–Marquardt](../../math/levenberg-marquardt/)'s `marquardt`,
`gain` and `schedule`, and elimination's junction tree, and these stay too. The slugs of the moved notes do
not change. What moves (the linearization's `objective` to `nonlinear-least-squares`, the
improper factor's `eliminated` to `integrating-out` and `kalman` to `kalman-filter`, elimination's `normal-equations` to
the note of that name, and the gain ratio's general form to `trust-regions`, with
`levenberg-marquardt`'s tags kept as its instance) is cited only from other notes, which are
rewritten with it. So no done entry is amended.

The notes are written at this entry's publish stage, as the
[mathematics in order](../mathematics-in-order/writeup.md#this-entry-writes-the-notes) wrote its
own, with the blurb of the part in `site/src/lib/site.ts`, which still describes elimination as
what comes first.

## Consequences

- **The part begins where the probability ends.** Its first note takes the posterior that
  [models built forward](../../math/generative-models/) defines and asks for its mode, so the step
  from the second part to the third is one sentence and not a change of subject.
- **Each method is met once, in general, and then as an instance.** Newton's method, the rate of a
  curvature and the gain ratio are stated in note 2. Gauss–Newton, Levenberg–Marquardt and Fisher
  scoring are each that note with one choice made, and their notes shrink to the choice.
- **Elimination is motivated before it is met.** A reader reaches Householder reflections knowing
  which system they solve and why the obvious way of solving it fails.
- **Five notes are new, and two are split off another.** `map-estimation`, `trust-regions`,
  `nonlinear-least-squares`, `optimization-and-estimation` and `normal-equations` are new, and
  `integrating-out` and `kalman-filter` leave `improper-gaussian`. About a third of the part's prose is rewritten or moved, and the
  linearization, now fourth, has three notes before it where it had two.
- **The entry on terms without residuals, still open, rebases onto this order.** Its open question
  on the Newton step and the observed information has a note to cite for Newton's method, and its
  note on the curvature of every term states its rate as an instance of `trust-regions` rather
  than proving it again.
- **The pages need little.** The order is the notes' own, and the pages that link into the part
  link to slugs and sections that stay; the concept page on linearization, which lists the notes
  its mathematics is in, may add the new ones.

## Alternatives

**Keep elimination first, with an introduction.** A paragraph at the head of the part, or of the
note, could say what the linear algebra is for. It would leave the order's rule and its purpose
apart: the reader is told what is coming, then made to read the linear algebra before it. And it
would not supply the three missing ideas, which are mathematics and not motivation.

**Fold the trust region into Levenberg–Marquardt.** The method is the part's one trust-region
method, so its note could carry the general frame. Then the frame would arrive after elimination,
too late for the linearization to be read as a model in a trust region, and Fisher scoring and the
curvatures of terms without residuals, which are damped by the same frame, would cite a note about
a particular method for a general fact.

**One note for the least-squares form and the linearization.** The objective of a model of normals
and its linearization are one section apart today, in one note. They are two ideas: that the mode
of such a model is a least-squares problem holds whatever method finds it, and the linearization is
one method. Kept apart, the third note is short and the fourth begins with the method.

**Leave the improper factor whole, and after elimination.** The second reading of the
linearization would then come after elimination, or be written in the linearization's note
without the factor it is an instance of, and the normal equations would be stated for a model and
not for a Gaussian. Splitting the note costs one section's move.

**Put the table at the end of `improper-gaussian`.** It is the first note in which both readings
exist, and a table is not an idea. But it is the part's glossary, read more often than any one
note and by readers who come for it alone, and it should be found by its title and not at the
foot of a note on another subject.

**Put the table where the readings meet, after the fifth note.** That was the order the part
first took, and it explains the two readings at the moment both exist. It would have rows for five
notes alone, and its reader would still look elsewhere for the information form, the filter's gain
and the covariance of an approximation. A short table there and a full one at the end would say
the same things twice.

**Keep the filter with what elimination leaves, or in elimination.** The Kalman filter is that
note's integral repeated, and could be its closing example; or the two could be a section of
elimination, beside *What the eliminated problem answers*. Elimination is about numbers and
matrices alone, and the filter is about a model. And what an integral leaves holds of any model,
the filter of a chain: two ideas, and the second is a method a reader looks for by its name.

**Do it in the entry that needed it.** The entry on terms without residuals, still open, asks
whether the Newton step belongs in it, and a note on Newton's method could have arrived there. The reordering is not that entry's subject,
it touches every note of the part, and it is argued here, where it can be read and approved alone.
