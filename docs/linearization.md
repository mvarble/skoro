# Linearization and Elimination

This note explains what `skoro-linearize`, and the elimination in `skoro-dense` beneath it, are for.
It starts from the Gauss–Newton method and the linear system it solves at every step, and spends
some time on how such a system should be solved, since that is where most of the design comes
from. It then turns to Fisher scoring, whose linear system differs in one instructive way, and
to what changes when a model has thousands of variables, each touching only a few others. With
the methods and their systems written out, the last third of the note introduces the single
abstraction they share, a factor graph of whitened rows, and then the code that implements it.
It closes with the full list of methods that abstraction supports, and what each would still need.

The companion note [fisher-scoring.md](fisher-scoring.md) is cited rather than repeated. Its
notation is kept: $\theta \in \mathbb R^D$ denotes the free parameters, $y$ the data, and
$\ell(\theta) = -\log p(y, \theta)$ the negative log-density to be minimized, with gradient
$g = \nabla \ell$ and Hessian $H = \nabla^2 \ell$. The code differentiates $\log p$ rather than
$\ell$, so its gradients are $-g$; this is noted where it matters. For parameters on a curved
manifold, $\theta \oplus \delta$ denotes the retraction and every derivative is taken in tangent
coordinates, as the site's *Manifolds and retractions* page explains. Nothing below depends on
curvature beyond that substitution, so the reader may take $\theta \oplus \delta = \theta + \delta$
throughout.

## 1. Gauss–Newton

A large class of models has a negative log-density that is a sum of squares. Suppose every term of
the model is Gaussian with a fixed covariance, $y_t \sim \mathcal N(f_t(\theta), \Sigma_t)$ with
$\Sigma_t = L_t L_t^\top$. Writing $W_t = L_t^{-1}$ for the **whitening** and
$r_t(\theta) = W_t\,(y_t - f_t(\theta))$ for the **whitened residual**,

$$
\ell(\theta) = \tfrac12 \sum_t \lVert r_t(\theta) \rVert^2 + \text{const}
= \tfrac12 \lVert r(\theta) \rVert^2 + \text{const},
$$

where $r$ stacks the $r_t$. The whitening is what makes the plain sum of squares correct: it
rescales each residual so that, under the model, $r_t$ is standard normal, and residuals measured
in different units, or with different confidence, are weighed against one another by their
covariances.

Newton's method ([fisher-scoring.md](fisher-scoring.md), §1) would minimize $\ell$ by solving
$H\delta = -g$ at every step. For a sum of squares, with $J = \partial r / \partial \theta$,

$$
g = J^\top r, \qquad H = J^\top J + \sum_i r_i \, \nabla^2 r_i .
$$

The **Gauss–Newton** method drops the second term. The cleanest way to see what that means is to
linearize the residual rather than the objective. For a small step,

$$
r(\theta \oplus \delta) = r(\theta) + J\,\delta + O(\lVert\delta\rVert^2),
$$

and substituting into $\ell$ gives the model

$$
\ell(\theta \oplus \delta) \approx m(\delta) = \tfrac12 \lVert J\delta + r \rVert^2 + \text{const}.
$$

$m$ is a convex quadratic in $\delta$. Setting its gradient $J^\top (J\delta + r)$ to zero gives the
**normal equations**

$$
J^\top J\, \delta = -J^\top r, \qquad \theta \leftarrow \theta \oplus \delta .
$$

This is the system in the shorthand $(J^\top J)\,\delta x = -J^\top \delta z$, with $\delta z$ the
residual. Its matrix is the Hessian with the term $\sum_i r_i \nabla^2 r_i$ removed. That term is
large exactly when the residuals are large, and it is what makes $H$ indefinite far from an
optimum. [fisher-scoring.md](fisher-scoring.md) (Proposition 2) shows that for these models
$J^\top J$ is precisely the Fisher information, so Gauss–Newton *is* Fisher scoring here, and every
argument made there for the reliability of that method applies. In particular, $J^\top J$ is
positive semidefinite, so the step is never uphill.

Two readings of the step will matter later. The first is as a linear system with matrix
$J^\top J$. The second, which will turn out to be the better one numerically, is as the solution of
a **linear least-squares problem**:

$$
\delta = \arg\min_\delta \lVert J\delta + r \rVert^2 .
$$

The normal equations are nothing more than the condition that this minimum is attained.

## 2. Levenberg–Marquardt

The Gauss–Newton step is the exact minimizer of a model that is only accurate near $\delta = 0$.
Far from an optimum the step can be too long, as the first example of
[fisher-scoring.md](fisher-scoring.md) (§4) shows. **Levenberg–Marquardt** shortens it by adding
a penalty on its length:

$$
\left(J^\top J + \lambda D\right) \delta = -J^\top r ,
$$

for a damping $\lambda \ge 0$ and a positive diagonal scaling $D$, often $I$ or the diagonal of
$J^\top J$. As $\lambda \to 0$ this is Gauss–Newton; as $\lambda \to \infty$ the step tends to
$-g/\lambda$ scaled by $D^{-1}$, a short step down the gradient.

**Proposition 1.** The damped step is the solution of the least-squares problem

$$
\min_\delta \left\lVert \begin{bmatrix} J \\ \sqrt\lambda\, D^{1/2} \end{bmatrix} \delta
- \begin{bmatrix} -r \\ 0 \end{bmatrix} \right\rVert^2 .
$$

*Proof.* Write $\tilde J$ and $\tilde b$ for the stacked matrix and right-hand side. The normal
equations of the stacked problem are $\tilde J^\top \tilde J\, \delta = \tilde J^\top \tilde b$.
Multiplying the blocks out gives $\tilde J^\top \tilde J = J^\top J + \lambda D$ and
$\tilde J^\top \tilde b = -J^\top r$. $\blacksquare$

Damping is therefore not a new kind of problem. It is the same least-squares problem with extra
rows appended, one block $\sqrt\lambda\, D_k^{1/2}$ for each variable. Everything said below about
solving least-squares problems applies to it unchanged.

The damping is adapted from step to step. After computing $\delta$, one compares the decrease
the model promised, $m(0) - m(\delta)$, with the decrease actually obtained,
$\ell(\theta) - \ell(\theta \oplus \delta)$. If their ratio is close to one the model is
trustworthy, the step is accepted, and $\lambda$ is reduced. If the ratio is small or negative the
step is rejected and $\lambda$ is increased. The promised decrease is a by-product of the solve
itself (§3.4), so this test costs one extra evaluation of $\ell$.

## 3. Solving a Least-Squares Problem

This section is for readers, like the author of the request that prompted it, whose numerical
linear algebra is rusty. Its conclusion is the reason the rest of the design looks the way it
does: **solve the least-squares problem, never the normal equations.**

### 3.1 Conditioning, and what forming $J^\top J$ costs

Floating-point arithmetic keeps about 16 significant decimal digits in `f64`. More precisely,
every operation is exact up to a relative error of at most $u \approx 1.1 \times 10^{-16}$, the
**unit roundoff**. A linear system amplifies such errors by its **condition number**. For a
matrix $A$ with largest and smallest singular values $\sigma_{\max}$ and $\sigma_{\min}$, it is
$\kappa(A) = \sigma_{\max} / \sigma_{\min}$, and a solution computed stably is typically accurate to
about $u\,\kappa(A)$ in relative terms. The rule of thumb is that one loses $\log_{10} \kappa$ of the
16 digits.

The singular values of $J^\top J$ are the squares of those of $J$, so

$$
\kappa(J^\top J) = \kappa(J)^2 .
$$

Forming the normal-equations matrix doubles the number of digits lost. A problem with
$\kappa(J) = 10^6$, which is not unusual when variables have very different scales, loses 6
digits through a least-squares solver and 12 through the normal equations, leaving 4.

**Example (Läuchli).** Let $\varepsilon = 10^{-8}$ and

$$
J = \begin{bmatrix} 1 & 1 \\ \varepsilon & 0 \\ 0 & \varepsilon \end{bmatrix},
\qquad
b = J \begin{bmatrix} 1 \\ 1 \end{bmatrix} = \begin{bmatrix} 2 \\ \varepsilon \\ \varepsilon \end{bmatrix}.
$$

The least-squares solution of $J\delta \approx b$ is $\delta = (1, 1)$ exactly, with zero residual,
and $\kappa(J) = \sqrt{2 + \varepsilon^2}/\varepsilon \approx 1.4 \times 10^8$, large but well within
`f64`'s range. But

$$
J^\top J = \begin{bmatrix} 1 + \varepsilon^2 & 1 \\ 1 & 1 + \varepsilon^2 \end{bmatrix},
$$

and $1 + 10^{-16}$ rounds to $1$ in `f64`, so the computed normal-equations matrix is exactly
$\left[\begin{smallmatrix} 1 & 1 \\ 1 & 1 \end{smallmatrix}\right]$, which is singular. No method for
solving the normal equations can recover the answer, because the information was destroyed when
the matrix was formed. The method of the next subsections returns $(1, 1)$ to full precision.

### 3.2 Orthogonal matrices, and the QR factorization

A square matrix $Q$ is **orthogonal** if $Q^\top Q = I$. It then preserves lengths,
$\lVert Qv \rVert = \lVert v \rVert$, and so it can be applied to both sides of a least-squares
problem without changing its solution:

$$
\lVert J\delta - b \rVert = \lVert Q^\top (J\delta - b) \rVert .
$$

Orthogonal matrices also have condition number one, so multiplying by one amplifies no errors. The
strategy is to choose $Q$ so that $Q^\top J$ has a shape in which the problem is easy.

**Theorem (QR factorization).** Every $m \times n$ matrix $J$ with $m \ge n$ can be written
$J = QR$, with $Q$ an $m \times m$ orthogonal matrix and $R$ an $m \times n$ matrix that is zero
below its diagonal. Write $R = \left[\begin{smallmatrix} R_1 \\ 0 \end{smallmatrix}\right]$ with
$R_1$ square and upper-triangular.

Split $Q^\top b = \left[\begin{smallmatrix} d \\ e \end{smallmatrix}\right]$ conformably. Then

$$
\lVert J\delta - b \rVert^2
= \left\lVert \begin{bmatrix} R_1 \delta \\ 0 \end{bmatrix} - \begin{bmatrix} d \\ e \end{bmatrix} \right\rVert^2
= \lVert R_1 \delta - d \rVert^2 + \lVert e \rVert^2 .
$$

The second term does not depend on $\delta$. The first is zero, its smallest possible value, when

$$
R_1 \delta = d .
$$

So the least-squares solution solves a **triangular** system, and the minimum value of the
objective is $\lVert e \rVert^2$, which is available for free. Moreover $R_1^\top R_1 = J^\top J$,
since $J^\top J = R^\top Q^\top Q R = R^\top R$. $R_1$ is a *square root* of the normal-equations
matrix, obtained without ever forming it, and $\kappa(R_1) = \kappa(J)$.

### 3.3 Triangular systems: back substitution

An upper-triangular system is solved from the bottom up. For $R_1 \delta = d$ with
$R_1 = (\rho_{ij})$, the last equation involves only the last unknown, the one before it only the
last two, and so on:

$$
\delta_n = \frac{d_n}{\rho_{nn}}, \qquad
\delta_i = \frac{1}{\rho_{ii}} \Big( d_i - \sum_{j > i} \rho_{ij}\, \delta_j \Big), \quad i = n-1, \dots, 1 .
$$

This is **back substitution**. It costs about $n^2$ operations, and it is backward stable. The
mirror-image procedure for a lower-triangular matrix, top down, is forward substitution, and
$R_1^\top$ is lower-triangular, so $R_1^\top y = c$ is solved that way. A zero or tiny diagonal
entry $\rho_{ii}$ means the corresponding direction is not determined by the data. The solver
reports it rather than dividing by it.

### 3.4 Computing $R$: Householder reflections

The factor $Q$ is never formed. A **Householder reflection** $P = I - 2vv^\top/\lVert v\rVert^2$
is orthogonal, and $v$ can be chosen so that $P$ maps a given column onto a multiple of the first
coordinate vector, zeroing everything below its first entry. Applying one reflection per column,
each to the part of the matrix not yet reduced, turns $J$ into $R$ in about $2mn^2$ operations.
That is roughly twice the cost of forming $J^\top J$ and factoring it, and it is the price of the
digits saved.

A convenient trick avoids storing $Q$ at all. Apply the reflections to the augmented matrix
$[\,J \mid b\,]$: they reduce $J$ to $R$ and, in the same stroke, turn $b$ into $Q^\top b$. The top
part of the last column is $d$, and the entry just below $R_1$ is $\pm\lVert e \rVert$, the residual
norm. One factorization of $[\,J \mid b\,]$ delivers everything the step needs, including the
promised decrease that the Levenberg–Marquardt test of §2 uses.

For a damped step one factors $[\,J \mid -r\,]$ with the damping rows appended (Proposition 1). No
new machinery is needed.

### 3.5 When only the normal equations are available: Cholesky

Sometimes the problem arrives as a symmetric positive-definite matrix $N$ and a right-hand side
$c$, with no rows $J$ at hand. The standard method is then the **Cholesky factorization**
$N = LL^\top$, $L$ lower-triangular, followed by the two triangular solves $Ly = c$ and
$L^\top \delta = y$. It is stable for what it is given, but it inherits $\kappa(N)$, which is
$\kappa(J)^2$ whenever $N = J^\top J$.

There is a middle case, which the next section needs. The rows $J$ *are* available, but the
right-hand side is not of the form $J^\top b$. Then QR of $J$ still gives $R_1$ with
$R_1^\top R_1 = J^\top J$ without forming the product, and

$$
J^\top J\, \delta = c \iff R_1^\top R_1\, \delta = c \iff R_1^\top y = c, \quad R_1 \delta = y ,
$$

two triangular solves. The factor was computed at $\kappa(J)$, but the solve itself is sensitive
at $\kappa(J)^2$, as any normal-equations solve is. That is the best available when no residual
comes with the rows.

## 4. Fisher Scoring and Its System

Gauss–Newton needs every term to be a sum of squares with a fixed covariance. Fisher scoring
([fisher-scoring.md](fisher-scoring.md), §3) needs only that every term offer a root of its
expected information. Its step solves

$$
\mathcal I(\theta)\, \delta = -g, \qquad \mathcal I(\theta) = A^\top A,
\qquad A = \begin{bmatrix} C_1 J_1 \\ \vdots \\ C_T J_T \end{bmatrix},
$$

with $C_t$ the term's Fisher root and $J_t$ the Jacobian of its arguments
([fisher-scoring.md](fisher-scoring.md), §6), and its damped form adds $\lambda D$ as before.

The matrix again comes as a stack of rows, $A$. The right-hand side is the difference. In
Gauss–Newton it was $-J^\top r$, the rows' own residual pulled back through the rows, so the whole
problem was a least-squares problem in $[\,J \mid -r\,]$. In Fisher scoring it is $-g$, the
gradient of the whole model. In general no vector $b$ with $A^\top b = -g$ is known, so there is no
least-squares problem to solve, only the normal equations with rows available: the middle case of
§3.5. The step is computed as

$$
R^\top R = A^\top A \ \ (\text{QR of } A), \qquad R^\top y = -g, \qquad R\,\delta = y .
$$

Two special cases are worth knowing.

- **Gaussian terms with fixed covariances.** Then $C_t J_t$ is the whitened residual's Jacobian,
  above rows of zeros, and $-g = -J^\top r$. The two methods coincide, and the least-squares route
  of §§1–3 is preferable because it works at $\kappa(J)$.
- **Gaussian terms whose covariance is itself estimated** (heteroscedastic noise). Now
  $\ell = \tfrac12 \lVert r \rVert^2 - \log\lvert\det W(\theta)\rvert + \text{const}$. The
  log-determinant depends on $\theta$, so minimizing $\tfrac12\lVert r\rVert^2$ alone optimizes the
  wrong objective, and Gauss–Newton is simply incorrect. Fisher scoring remains correct, because
  the Gaussian's root carries the information of the covariance factor as well
  ([fisher-scoring.md](fisher-scoring.md), §5). This is the main reason both methods exist.

## 5. Sparsity: The Same Systems at Scale

Everything so far treated $J$ as one dense matrix. Real models are different. A pose graph, a
time series, or a hierarchical model has thousands of variables, and each term touches only a
few of them. A measurement of one pose touches that pose; an odometry term touches two
consecutive poses; a prior touches one. The Jacobian of a four-pose chain, with a prior on the
first pose, odometry between neighbours, and one measurement per pose, has this block pattern:

```text
                 x1   x2   x3   x4
prior            ■
odometry 1→2     ■    ■
odometry 2→3          ■    ■
odometry 3→4               ■    ■
measure x1       ■
measure x2            ■
measure x3                 ■
measure x4                      ■
```

Each ■ is a small dense block (6 × 6 for a pose), and everything else is zero. A model with $n$
poses has $O(n)$ nonzero blocks in a matrix with $O(n^2)$ blocks. Dense QR would cost $O(n^3)$ and
spend nearly all of it multiplying zeros. This section shows how to exploit the pattern, one
variable at a time.

### 5.1 Eliminating one variable

Pick a variable $x_j$. Only the terms that touch it involve $\delta_j$. Collect them, and call the
other variables they touch the **separator** $S$ of $x_j$. Stacking those terms' rows gives a small
dense matrix $[\,A_j \mid A_S \mid b\,]$, with $x_j$'s columns first. One QR factorization (§3.4)
brings it to

$$
Q^\top [\,A_j \mid A_S \mid b\,] = \begin{bmatrix} R_j & T & d_j \\ 0 & A'_S & b' \\ 0 & 0 & e \end{bmatrix},
$$

and since $Q$ preserves norms, those terms' contribution to the objective splits exactly into three
pieces:

$$
\lVert A_j\delta_j + A_S\delta_S - b \rVert^2
= \underbrace{\lVert R_j\delta_j + T\delta_S - d_j \rVert^2}_{\text{conditional}}
+ \underbrace{\lVert A'_S\delta_S - b' \rVert^2}_{\text{new term on } S}
+ e^2 .
$$

The first piece is a **conditional**: for any value of $\delta_S$ it is zero at
$\delta_j = R_j^{-1}(d_j - T\delta_S)$, so minimizing over $\delta_j$ removes it. The second is a
new term on the separator alone, and it replaces every term that touched $x_j$. The third is error
that no step can remove. The problem now has one variable fewer.

**Proposition 2.** The new term is a square root of the Schur complement:
$A_S'^{\top} A_S' = \Lambda_{SS} - \Lambda_{Sj}\Lambda_{jj}^{-1}\Lambda_{jS}$, where $\Lambda$ is the
normal-equations matrix of the collected terms.

*Proof.* Because $Q$ is orthogonal, $\Lambda$ equals the product of the reduced matrix's columns
with themselves: $\Lambda_{jj} = R_j^\top R_j$, $\Lambda_{jS} = R_j^\top T$, and
$\Lambda_{SS} = T^\top T + A_S'^\top A_S'$. Substituting,
$\Lambda_{SS} - \Lambda_{Sj}\Lambda_{jj}^{-1}\Lambda_{jS}
= T^\top T + A_S'^\top A'_S - T^\top R_j (R_j^\top R_j)^{-1} R_j^\top T = A_S'^\top A'_S$.
$\blacksquare$

The Schur complement is what Gaussian elimination on $\Lambda$ would have produced. Elimination by
QR reaches the same thing through square roots, never forming $\Lambda$ and never inverting
anything.

**Example.** Two scalar variables: a prior $x_1 = 1$ and an odometry measurement $x_2 - x_1 = 1$,
both with unit variance. The rows are $[\,1\ \ 0 \mid 1\,]$ and $[\,{-1}\ \ 1 \mid 1\,]$. Eliminating
$x_1$ with one QR gives

$$
\begin{bmatrix} \sqrt2 & -1/\sqrt2 & 0 \\ 0 & 1/\sqrt2 & \sqrt2 \end{bmatrix}.
$$

The first row is the conditional $\sqrt2\,x_1 - x_2/\sqrt2 = 0$, that is, $x_1 = x_2/2$. The second
is the new term on $x_2$, $x_2/\sqrt2 = \sqrt2$, which gives $x_2 = 2$ and then $x_1 = 1$. Its
information $(1/\sqrt2)^2 = \tfrac12$ is the Schur complement of
$\Lambda = \left[\begin{smallmatrix} 2 & -1 \\ -1 & 1 \end{smallmatrix}\right]$, namely
$1 - 1 \cdot \tfrac12 \cdot 1 = \tfrac12$. So the marginal variance of $x_2$ is $2$: one unit from
the prior, one from the odometry.

### 5.2 Order, and fill

Eliminate every variable in turn, and the conditionals stack into one upper-triangular $R$ with
$R^\top R = J^\top J$, just as dense QR would give (§5.4). The cost, however, depends entirely on
the **order**. Eliminating $x_j$ puts a new term on its whole separator, so later eliminations
face larger separators. The new nonzeros this creates are called **fill**.

The extreme case is a star: one hub variable touched by every term, each term also touching one
leaf. Eliminating the hub first creates a term over all $n$ leaves at once, a dense $n \times n$
block, and everything after it is dense work. Eliminating the leaves first creates, for each, a
new term on the hub alone, with no fill, and the hub is eliminated last in a tiny problem. Same
matrix, same answer, but $O(n^3)$ against $O(n)$.

Finding the order with least fill is NP-hard, but good heuristics are cheap. **Approximate
minimum degree** repeatedly picks a variable whose separator is currently smallest. **COLAMD** is
its standard form for least-squares problems, and it works on the pattern of $J$ alone, without
forming $J^\top J$. It is applied to the *block* pattern, one entry per term and variable rather
than per scalar, since every scalar column of a variable has the same pattern. Sometimes the order
must put chosen variables last, and §6 will need that.

### 5.3 Trees: doing the bookkeeping once

The separators depend only on which variables each term touches, not on any numbers. So the whole
elimination can be planned before it is performed. Give each variable $j$ a **parent**, the
earliest variable of its separator in the elimination order: the variable its new term will be
eliminated with next. This makes an **elimination tree**. Its leaves are eliminated first; each
variable's work waits only on its children's.

Chains of variables in which each has a single child, and a separator equal to the child's minus
itself, can be eliminated together as one dense block with no extra fill. Grouping them gives
**cliques** (the tree is then a **junction tree**), and eliminating each clique with a single QR is
**multifrontal** elimination. It does the same arithmetic in fewer, larger dense operations, which
is faster.

This planning, the **symbolic** factorization, is reusable. Successive iterations of an optimizer
change the numbers in $J$ but not its pattern, so the order and the tree are computed once and the
**numeric** elimination is repeated each step.

### 5.4 What elimination leaves: the Bayes tree

The conditionals, one per clique, arranged as the cliques were, form a **Bayes tree**. Each says

$$
R_c\,\delta_{F_c} + S_c\,\delta_{S_c} = d_c ,
$$

for the clique's own variables $F_c$ given its separator $S_c$. Stacked in elimination order, their
rows form the upper-triangular $R$ of §3, and every solve of §3 becomes a traversal of the tree:

| Computes | Traversal | Use |
| --- | --- | --- |
| $R^{-1} d$ | root to leaves: each clique solves for its own variables given its separator's | the least-squares step (§1) |
| $R^{-\top} c$ | leaves to root | the first half of a normal-equations solve (§3.5) |
| $R^{-1} R^{-\top} c = (J^\top J)^{-1} c$ | both | the Fisher-scoring step (§4) |
| $R^{-1} z$ | root to leaves | a draw from $\mathcal N(0, (J^\top J)^{-1})$ when $z \sim \mathcal N(0, I)$ (§6.3) |
| $\sum_i \log R_{ii}$ | none | $\tfrac12 \log\det J^\top J$ (§6.4) |

The first row is exact least squares at $\kappa(J)$, done block-sparsely. The accumulated
$\sum e^2$ from every elimination is the minimum of the objective, the Levenberg–Marquardt test's
promised decrease.

## 6. Beyond Steps: Marginals

The same elimination answers a second family of questions, about uncertainty rather than location.

### 6.1 The Laplace approximation

Near its mode $\hat\theta$, a posterior is approximately Gaussian:

$$
p(\theta \mid y) \approx \mathcal N\!\left(\hat\theta,\ \Lambda^{-1}\right),
$$

with $\Lambda$ the Hessian of $\ell$ at $\hat\theta$, or its Gauss–Newton or Fisher replacement.
For a linear-Gaussian model it is exact. Its **marginal** on a subset $K$ of the variables is again
Gaussian, with information given by the Schur complement

$$
\Lambda_K = \Lambda_{KK} - \Lambda_{KE}\Lambda_{EE}^{-1}\Lambda_{EK},
$$

where $E$ is everything else. This is what a standard error, a marginal covariance, or a
confidence region needs.

### 6.2 Marginals by partial elimination

By Proposition 2, applied repeatedly, eliminating exactly the variables $E$ leaves terms on $K$
whose normal-equations matrix is $\Lambda_K$. The marginal is therefore a *partial* elimination,
ordered so that $K$ comes last, and it is cheap: only the terms touching eliminated variables are
ever read. One more QR squares the leftover terms up into a single triangular root
$\Lambda_K = W^\top W$, with a right-hand side $d$. Then

- the marginal information is $W^\top W$, and the covariance is $W^{-1}W^{-\top}$;
- the marginal mean, relative to the linearization point, is $W^{-1} d$;
- $\log\lvert\det W\rvert = \tfrac12 \log\det \Lambda_K$.

When $W$ is to become the whitening of a Gaussian, it must be *lower*-triangular, which QR does not
directly give. Reversing the order of $K$'s scalar coordinates before the last QR, and reversing
both axes of the result, turns the upper factor into a lower one with the same product.

### 6.3 Sampling

If $z \sim \mathcal N(0, I)$ then $R^{-1} z \sim \mathcal N(0, (R^\top R)^{-1})$. Back substitution
with $z$ in place of $d$ therefore draws from the Laplace approximation's spread, which retracted
onto $\hat\theta$ gives an approximate posterior sample. Such draws make a natural proposal for
importance sampling against the exact posterior, and $R$ is a natural constant preconditioner
(mass matrix) for Hamiltonian Monte Carlo.

### 6.4 Evidence

The Laplace approximation to the model evidence is

$$
\log p(y) \approx \log p(y, \hat\theta) + \tfrac{D}{2}\log 2\pi - \tfrac12 \log\det\Lambda ,
$$

and $\tfrac12\log\det\Lambda = \sum_i \log R_{ii}$ comes from the diagonal of the factor, with no
determinant ever computed.

### 6.5 Marginalizing a variable *out*

A sliding-window estimator keeps only recent variables and must retire old ones without discarding
what they said. This is a subtly different question from §6.2. It asks not for the marginal of the
whole model on the remaining variables, but for the one term that can **replace** the terms of the
retired variable. Split the objective into the terms touching the retired variables $E$ and the
rest, $\Phi = \Phi_{\ni E} + \Phi_{\not\ni E}$, and let $B$ be the other variables the first group
touches, the **Markov blanket** of $E$. Since the rest does not involve $\delta_E$,

$$
\min_{\delta_E} \Phi
= \underbrace{\min_{\delta_E} \Phi_{\ni E}(\delta_E, \delta_B)}_{\varphi_B(\delta_B)}
+ \Phi_{\not\ni E} .
$$

So replacing the terms $\Phi_{\ni E}$ by the single term $\varphi_B$, computed from those terms
alone, leaves the problem on everything else unchanged. Computing instead the marginal of the whole
model on $B$ would fold $\Phi_{\not\ni E}$ in as well, and that would count it twice once attached,
because the model still holds those terms. For a linear-Gaussian model the replacement is exact.
For a chain, retiring the oldest state folds in its measurement and pushes the result through the
dynamics: the Kalman filter's update-and-predict step, in square-root information form
(§9, item 8).

The replacement term is a Gaussian over $B$ in the tangent coordinates of the point
$\hat\theta_B$ where it was computed: $\tfrac12\lVert W(\theta_B \ominus \hat\theta_B) - d\rVert^2$.
Keeping the linearization point and the offset $d$, rather than moving the anchor to the marginal's
mean, makes its value and Jacobian at $\hat\theta_B$ equal to the eliminated rows exactly, on any
manifold.

## 7. The Abstraction: A Factor Graph of Whitened Rows

Every method in §§1–6 manipulates the same object:

$$
\Phi(\delta) = \sum_f \tfrac12 \big\lVert A_f\,\delta_{K_f} - b_f \big\rVert^2 ,
$$

a sum of **factors** $f$. Each is a small dense block of rows $A_f$ over the few variables $K_f$ it
touches, with a right-hand side $b_f$. Its curvature is $\Lambda = A^\top A$, with $A$ the factors'
rows stacked and scattered into their variables' columns. What changes from method to method is only
what the rows are, what the right-hand side is, and which solve is asked of the eliminated result:

| Method | Rows $A_f$ | Right-hand side $b_f$ | Extra factors | Asked of the elimination |
| --- | --- | --- | --- | --- |
| Gauss–Newton (§1) | $J_t$, residual Jacobian | $-r_t$ | none | $R^{-1}d$ |
| Levenberg–Marquardt (§2) | $J_t$ | $-r_t$ | $\sqrt\lambda\,D_k^{1/2}$ per variable, $b = 0$ | $R^{-1}d$; the error for the gain ratio |
| Fisher scoring (§4) | $C_t J_t$, root composed with Jacobian | $0$ | damping, as above | $R^{-1}R^{-\top}(-g)$ |
| Laplace marginal (§6.2) | either | either | none | partial elimination, then $W$, $d$ |
| Laplace sampling (§6.3) | either | irrelevant | none | $R^{-1}z$ |
| Evidence (§6.4) | either | irrelevant | none | $\sum \log R_{ii}$ |
| Marginalizing out (§6.5) | either | either | none | partial elimination of the retired variables' factors |

The elimination machinery of §5 never asks what the rows mean. That is the central design point:
**one representation, one elimination, many methods.** The meaning lives entirely in how the factor
graph was built, and in which solve is called.

Three properties of the rows carry through every row of the table. The rows are **whitened**, so a
plain sum of squares weighs every term correctly. They are **block-sparse**, each factor touching
only the variables it reaches. And they are **square roots**: nothing ever forms $A^\top A$, so every
method that has a right-hand side for its rows works at $\kappa(A)$.

## 8. The Code

The implementation is in three layers, which follow the three stages of the argument: from a
model to rows, from rows to eliminated factors, and the questions asked of the result.

### 8.1 From a model to rows (`skoro-model`)

The rows come from the model's derivatives, one term at a time. A term declares what its rows are
through **bindings**: a residual under the `RESIDUAL` role, a Fisher root under `FISHER_ROOT`, and
the quantities its root is expressed over under `ARGUMENT`.

| Concept | Code |
| --- | --- |
| The variables a term touches, $K_f$ | `ModelReaderOps::leaf_support`: the free leaves a node depends on |
| $J_t$ over only $K_t$, for every term, in one pass | `jacobian::jacobian_blocks`, which shares derivative passes between terms that touch disjoint variables |
| $(K_t, J_t, r_t)$ per term: Gauss–Newton rows | `least_squares::residual_blocks` |
| $(K_t, C_t J_t)$ per term: Fisher rows | `information::information_blocks` |
| "every term offers the rows I need" | `coverage::require_role` |

These run on the tensor backend (CPU or GPU) and return tensors.

### 8.2 Rows, elimination, and solves (`skoro-dense`)

This layer knows nothing of models. Its variables are **keys** of any type, and it works in host
`f64`.

| Concept | Code |
| --- | --- |
| A factor $(K_f, A_f, b_f)$ | `factor_graph::JacobianFactor` |
| $\Phi$: declared keys and their dimensions, and factors | `factor_graph::GaussianFactorGraph` (`declare`, `add`) |
| Damping rows (§2) | `GaussianFactorGraph::damped`, `damped_diagonal` |
| $\Phi(\delta)$ and $\nabla\Phi$ | `GaussianFactorGraph::error`, `gradient` |
| A vector split by key ($\delta$, $g$, $z$) | `values::VectorValues` |
| The order (§5.2) | `ordering::Ordering`: `Colamd`, `ConstrainedLast(keys)`, `Natural`, `Custom` |
| The symbolic plan (§5.3) | `elimination::JunctionTree` (`new`, `partial`, `sequential`). It is reused while `GaussianFactorGraph::structure` is unchanged |
| Numeric elimination (§5.1) | `bayes::eliminate(tree, graph)`, returning an `Elimination { tree, remaining, error }` |
| The Bayes tree (§5.4) | `bayes::BayesTree` and its `GaussianConditional`s |
| $R^{-1}d$, $R^{-1}z$, $R^{-\top}c$, $R^{-1}R^{-\top}c$, $\sum\log R_{ii}$ | `back_substitute`, `solve_upper`, `solve_upper_transpose`, `solve_normal`, `log_det_r` |
| The residual error $\sum e^2$ | `Elimination::error` |
| Marginal on $K$ (§6.2) | `marginal::marginal(graph, keys)`, `marginal_with` to reuse a tree |
| Marginalizing out (§6.5) | `marginal::marginalize_out(graph, keys)` |
| $W$, $d$, $W^\top W$, $W^{-1}W^{-\top}$, $W^{-1}d$, $\log\det W$ | `marginal::Marginal`: `whitening`, `rhs`, `information`, `covariance`, `offset`, `log_det` |

A direction that the rows do not determine is reported as `SkoroError::RankDeficient`, naming its
place in the elimination order; §3.3's tiny diagonal is where it is detected.

### 8.3 The bridge (`skoro-linearize`)

This layer turns a model at a point into a `GaussianFactorGraph` whose keys are the model's node
ids, and connects the answers back to the model.

| Concept | Code |
| --- | --- |
| Which rows (§7's second and third columns) | `Curvature::Residual` (Gauss–Newton rows, $b = -r$) or `Curvature::Fisher` (root rows, $b = 0$, and the gradient $-g$ alongside) |
| The model at a point, as $\Phi$ | `linearize(model, points, free, curvature)`, returning one `Linearized { graph, gradient }` per batch row |
| A marginal on chosen quantities | `marginalize(model, points, free, keep, curvature)` |
| The factor replacing retired quantities | `marginalize_out(model, points, free, eliminate, curvature)` |
| That factor, attached back to the model | `joint_normal_from(model, &marginal, at)`: a `JointNormal` with `W`, the offset `d`, anchored at the linearization point |
| A step back onto the model | `to_flat_tensor`, then `PointMap::retract_flat` |

`linearize` also enforces the preconditions of §4. It refuses a model missing the rows it needs, and
it refuses the residual curvature for a term whose normalizer depends on a free quantity (the
heteroscedastic case), with `SkoroError::NormalizerNotConstant`.

### 8.4 The three methods, as code

A Levenberg–Marquardt step:

```rust
let free = model.tangent_layout(&ids)?;
let lin = linearize(&model, at.as_ref(), &free, Curvature::Residual)?.remove(0);
let damped = lin.graph.damped(lambda)?;
let tree = JunctionTree::new(&damped, &Ordering::Colamd)?;    // once per structure
let elim = eliminate(&tree, &damped)?;
let step = elim.tree.back_substitute()?;                      // R⁻¹ d
// promised decrease: Φ(0) − elim.error; compare with the real one, adapt λ
at = at.retract_flat(&free, to_flat_tensor(&step, &free, &device)?)?;
```

A Fisher-scoring step differs in the rows, the right-hand side, and the solve. Its factors can touch
different variables than the residual ones, so it gets a tree of its own:

```rust
let lin = linearize(&model, at.as_ref(), &free, Curvature::Fisher)?.remove(0);
let damped = lin.graph.damped(lambda)?;
let tree = JunctionTree::new(&damped, &Ordering::Colamd)?;    // once per structure
let step = eliminate(&tree, &damped)?
    .tree
    .solve_normal(lin.gradient.as_ref().unwrap())?;           // (AᵀA + λI)⁻¹ ∇log p
```

The gradient there is $\nabla\log p = -g$, so the step already points downhill in $\ell$.

A sliding-window step, retiring `old` whose blanket is `next`:

```rust
let m = marginalize_out(&model, at.as_ref(), &free, &[old.id()], Curvature::Residual)?.remove(0);
model.demote_to_parameter(&next)?;                     // removes the term that read `old`
add_joint_factor(&model, &joint_normal_from(&model, &m, &at)?)?;
model.remove_closure(&[old.id(), /* its measurements */])?;
```

### 8.5 How the pieces move

```text
model at a point ──(skoro-model: jacobian_blocks)──▶ per-term blocks (tensors)
        │                                                   │
        └──────────(skoro-linearize: linearize)─────────────┘
                              │
                              ▼
                GaussianFactorGraph<Id>  (host f64)
                              │  Ordering ─▶ JunctionTree  (symbolic, reused)
                              ▼
                  eliminate ─▶ Elimination { BayesTree, remaining, error }
                              │
     ┌───────────────┬────────┴───────┬──────────────┬─────────────────┐
     ▼               ▼                ▼              ▼                 ▼
 back_substitute  solve_normal    solve_upper    log_det_r     marginal / marginalize_out
  (GN / LM)       (Fisher)        (sampling)     (evidence)            │
     │               │                                                 ▼
     └──── to_flat_tensor ─▶ retract_flat                      joint_normal_from ─▶ model
```

## 9. What Can Be Built on It

The following methods use this machinery directly. For each, the entry says what it asks of the
code and what, if anything, is missing today.

1. **Nonlinear least squares by Gauss–Newton or Levenberg–Marquardt**: curve fitting, pose-graph
   SLAM, bundle adjustment, calibration. It is `Curvature::Residual`, `damped`, and
   `back_substitute` in a loop. Missing: a packaged loop with a damping schedule and a stopping
   rule. The primitives are all present.
2. **Maximum a posteriori estimation.** The same as item 1: a prior is just more rows, since a
   Gaussian prior is a term like any other.
3. **Fisher scoring, and iteratively reweighted least squares for generalized linear models.**
   It is `Curvature::Fisher` and `solve_normal`. It already covers Gaussians with estimated
   covariances. Missing: Fisher roots for non-Gaussian families (Poisson, Bernoulli). The machinery
   takes any root; only `Normal` defines one today.
4. **Trust-region methods, e.g. Powell's dogleg.** These need the Gauss–Newton step
   (`back_substitute`), the gradient ($-A^\top b$, which is `gradient` at zero), and the curvature
   along it ($\lVert A g\rVert^2$, a sum over factors). All are computable from the factor graph and
   the Bayes tree. Missing: the method itself.
5. **Laplace posteriors: standard errors, marginal covariances, confidence regions.** Solve to the
   mode, then `marginalize` onto the quantities of interest and read `covariance`. With the residual
   curvature this is the Gauss–Newton covariance, and with the Fisher curvature the inverse expected
   information.
6. **Laplace sampling, and Laplace proposals for importance sampling.** `solve_upper` of standard
   normal draws, retracted onto the mode, then `log_weight` against the exact model. Missing: a
   sampler that draws `z` reproducibly (see the RNG gap in `skoro.md` §5).
7. **Laplace evidence and model comparison.** $\log p(y, \hat\theta)$ from `log_density` plus
   `log_det_r`.
8. **Kalman filtering and smoothing, as a special case.** For a linear-Gaussian chain, eliminating
   the states in time order *is* the square-root information filter. The marginal on the newest
   state is the filtered density, each elimination is a predict-and-update, and back substitution
   from the newest state backwards is the Rauch–Tung–Striebel smoother. Nothing is missing. It is
   the elimination with the natural order, and the window test is its linear-Gaussian check.
9. **Fixed-lag smoothing and sliding-window estimation.** `marginalize_out`, `joint_normal_from`,
   `demote_to_parameter`, and `remove_closure` per tick (§6.5). For nonlinear models the retired
   factor stays frozen at its linearization point, which is the standard first-order
   approximation. Missing: a policy for when to relinearize.
10. **Iterated extended Kalman filtering and iterated smoothing.** Relinearize the window at each
    new estimate and re-solve: items 1 and 9 combined.
11. **Constant-metric Hamiltonian Monte Carlo.** Use the Laplace precision $M = R^\top R$ as the
    mass matrix. The velocity $M^{-1}p$ is `solve_normal(p)`. Drawing a momentum $p \sim \mathcal
    N(0, M)$ needs the product $R^\top z$, which the conditionals hold but no method exposes.
    Equivalently, sampling in the whitened coordinates $u = R\,\delta$ needs only `solve_upper` to
    map back. Missing: the sampler, the product, and RNG control.
12. **Many problems of one shape.** A batch of starting points or data sets is linearized into one
    graph per batch row, all of one structure, so one `JunctionTree` serves all of them.
13. **Identifiability diagnostics.** A `RankDeficient` error names the first undetermined direction
    in the elimination order, and a tiny `log_det` of a marginal flags a weakly determined subset.

Some methods this machinery does **not** support yet, and why:

- **Newton's method with the exact Hessian.** The Hessian is not a sum of squares, so it has no
  rows. It needs a second kind of factor, $\tfrac12\delta^\top G\delta - g^\top\delta$, eliminated by
  Cholesky rather than QR, and a per-term Hessian to fill it. The dense whole-model Hessian exists
  (`hessian_hvp`), but the factor form does not.
- **Incremental smoothing (iSAM2).** This means updating the Bayes tree in place as new factors
  arrive, re-eliminating only the affected cliques. Today the symbolic plan is reused, but every
  solve re-eliminates everything.
- **Riemannian-manifold HMC.** The metric must vary with $\theta$ and be differentiated, which host
  elimination cannot provide.
- **Reparameterization-gradient variational inference.** This needs derivatives through samplers.
  It is unrelated to this machinery, and noted only because natural-gradient variational methods
  would otherwise be a use of the Fisher solve.

## 10. Limitations

Everything here is a **linearization**: exact for linear-Gaussian models, and otherwise accurate
near the point where it was taken. A step is only as good as the model $m(\delta)$, which is why
damping exists. A marginal or a retired factor is only as good as its linearization point, which
is why a sliding window on a nonlinear model drifts unless it relinearizes.

The host solve is in `f64`, but the rows are only as precise as the backend that produced them.
Under an `f32` float policy the Jacobian carries about seven digits, and a genuinely undetermined
direction can then fail the rank test and come back as a very large step rather than an error.

The Fisher route's step is sensitive at $\kappa(A)^2$ (§3.5), even though its factor is computed
at $\kappa(A)$. When every term is Gaussian with a fixed covariance, the residual route gives the same
curvature and should be preferred.

Finally, the elimination's cost is governed by fill (§5.2). COLAMD is a heuristic and the only
ordering available. Problems with large two- or three-dimensional structure, such as dense visual
maps, are where nested-dissection orderings like METIS pay off, and where this implementation will
be slowest.
