---
title: Removable singularities in floating point
part: geometry
order: 8
summary: The coefficients of the rotations' and the rigid motions' maps are smooth at the identity, but their closed forms divide by a power of the angle there; each is computed by a series below a cutoff and its closed form above, and the cutoff is set by how the error of a sine and a cosine, relative or absolute, is amplified by that division.
depends_on: [automatic-differentiation, rotations, rigid-motions]
katex_macros:
    '\atantwo': '\operatorname{atan2}'
---

The exponential of a rotation, its logarithm and their Jacobians are smooth maps, and their
formulas are not: each has a coefficient such as $\sin(\theta/2)/\theta$ whose closed form is
$0/0$ at the identity, the point computations visit most. The singularity is _removable_: the
coefficient extends smoothly through $\theta = 0$, and its power series says how. This note lists
the coefficients, says how each is computed so that its value and its derivatives are right on
both sides of the identity, and works out what decides where the series should hand over to the
closed form. The answer depends on how accurately the machine computes a sine.

## The coefficients

The coefficients of [the rotations](../rotations/) and of [the rigid motions](../rigid-motions/)
are each a function of $\theta^2$, with $\theta$ the angle of a rotation vector, or for the
logarithm of $n^2$ and $w$, with $n = \lvert \operatorname{vec}(q) \rvert$.

| coefficient | closed form                                                   | series                                                                                                                | used by                                                |
| ----------- | ------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------ |
| $a$         | $\cos \tfrac\theta2$                                          | $\sum_k \frac{(-1)^k \theta^{2k}}{4^k (2k)!} = 1 - \frac{\theta^2}{8} + \frac{\theta^4}{384} - \cdots$                | the exponential [](eq:math/rotations/exp)              |
| $b$         | $\sin(\theta/2)/\theta$                                       | $\sum_k \frac{(-1)^k \theta^{2k}}{2^{2k+1} (2k+1)!} = \frac12 - \frac{\theta^2}{48} + \frac{\theta^4}{3840} - \cdots$ | the exponential                                        |
| $A$         | $(1 - \cos \theta)/\theta^2$                                  | $\sum_k \frac{(-1)^k \theta^{2k}}{(2k+2)!} = \frac12 - \frac{\theta^2}{24} + \frac{\theta^4}{720} - \cdots$           | the Jacobians [](eq:math/rotations/jacobians)          |
| $B$         | $(\theta - \sin \theta)/\theta^3$                             | $\sum_k \frac{(-1)^k \theta^{2k}}{(2k+3)!} = \frac16 - \frac{\theta^2}{120} + \frac{\theta^4}{5040} - \cdots$         | the Jacobians                                          |
| $C$         | $\bigl(1 - \frac\theta2 \cot \frac\theta2\bigr)/\theta^2$     | $\frac1{12} + \frac{\theta^2}{720} + \frac{\theta^4}{30240} + \cdots$                                                 | their inverses [](eq:math/rotations/jacobian-inverses) |
| $L$         | $2 \atantwo(n, w)/n$                                          | $\frac2w \sum_k \frac{(-1)^k x^k}{2k+1} = \frac2w \bigl(1 - \frac x3 + \frac{x^2}5 - \cdots\bigr)$, $x = n^2/w^2$     | the logarithm [](eq:math/rotations/log)                |
| $D$         | $(\theta^2 + 2 \cos \theta - 2)/(2 \theta^4)$                 | $\sum_k \frac{(-1)^k \theta^{2k}}{(2k+4)!} = \frac1{24} - \frac{\theta^2}{720} + \cdots$                              | the motions' Jacobians [](eq:math/rigid-motions/q)     |
| $E$         | $(2\theta - 3 \sin \theta + \theta \cos \theta)/(2 \theta^5)$ | $\sum_k \frac{(-1)^k (k+1)\, \theta^{2k}}{(2k+5)!} = \frac1{120} - \frac{\theta^2}{2520} + \cdots$                    | the motions' Jacobians                                 |

The series of $C$ is that of $\tfrac\theta2 \cot \tfrac\theta2 = 1 - \tfrac{\theta^2}{12} -
\tfrac{\theta^4}{720} - \tfrac{\theta^6}{30240} - \cdots$, and that of $L$ is $\atantwo(n, w) / n =
\arctan(n/w)/n$ for $w > 0$, expanded in $x$. Other spaces have coefficients of the same kind: the
volume of the exponential chart of a sphere, for one, is a power of $\sin r / r$, and its
logarithm $\log(\sin r / r) = -\tfrac{r^2}{6} - \tfrac{r^4}{180} - \tfrac{r^6}{2835} - \cdots$ is
computed the same way.

## A series below a cutoff

Each coefficient is computed by its series, truncated, where its argument is below a cutoff, and by
its closed form above it. Three things have to be right for the result to be the smooth function it
stands for, in its derivatives as well as its value.

**The series is a function of the same arguments.** A derivative is taken of the formula that was
evaluated, so the series must depend on its arguments as the function does. The coefficients in
$\theta$ are functions of $\theta^2$ and their series are polynomials in it. The logarithm's
coefficient depends on $n^2$ and on $w$, and its series keeps both. On the sphere $w = (1 -
n^2)^{1/2}$, and a series in $n^2$ alone would have the right values, but it would have no
derivative in $w$, and second derivatives along the sphere, which move $w$ at second order, would
be wrong. The logarithm's series also holds near the identity only, for $n^2 < c\, w^2$ and $w >
0$; near $q = -1$ the closed form is used, and what it has there is the logarithm's own cut, which
no series removes.

**The series keeps enough terms.** It keeps at least the terms through $\theta^2$, so that
derivatives of second order at the identity are exact, and beyond that as many as make its
truncation error at the cutoff smaller than the rounding of the closed form there.

**Neither formula is evaluated where it is undefined.** When both formulas are computed and one is
then chosen, as a computation over many values at once does, the one not chosen must still be
finite, since [a reverse pass](../automatic-differentiation/) multiplies its derivative by zero,
and zero times an undefined number is undefined. So the closed form is given a harmless argument
where the series is chosen, and the series, where it can be undefined, as the logarithm's $1/w$ is
at $w \le 0$, a harmless one where the closed form is.

## Relative and absolute error

A computed value carries an error, the difference between what the machine returns and the exact
value at the same, already rounded, input. Write $u$ for the unit roundoff of the arithmetic,
$2^{-24} \approx 6.0 \cdot 10^{-8}$ in single precision and $2^{-53}$ in double. A function has
_relative error_ $\eta$ when its computed value is $f(x)(1 + \delta)$ with $\lvert \delta \rvert \le
\eta$, and _absolute error_ $\varepsilon$ when it is $f(x) + \delta$ with $\lvert \delta \rvert \le
\varepsilon$. The two differ only where $f(x)$ is small, which is exactly where a removable
singularity is computed, since $\sin \theta \approx \theta$ near $0$.

A mathematical library's sine is accurate relative to its value, to within a few $u$. That is not
a law of nature. An implementation built for throughput, as a graphics processor's often is, may
promise only an absolute error $\varepsilon_s$ for its sine and $\varepsilon_c$ for its cosine, each
many times $u$. Near $\cos \theta \approx 1$ the two kinds of error hardly differ. Near $\sin
\theta \approx 0$ they differ by the factor $1/\theta$.

## How the error reaches a closed form

Every closed form of the table but the logarithm's is

$$
	f(\theta) = \frac{P\bigl(\theta, \sin \theta, \cos \theta\bigr)}{\theta^k}, \qquad P(\theta, \sin
	\theta, \cos \theta) = O(\theta^k), @tag(closed-form)
$$

with $P$ a polynomial whose own partial derivatives in its second and third arguments do not vanish
at $0$: $B$ has $P = \theta - \sin \theta$ and $k = 3$, and $E$ has $P = 2\theta - 3 \sin \theta +
\theta \cos \theta$ and $k = 5$. The numerator cancels to $O(\theta^k)$. The error of the sine and
the cosine does not. With the computed $\tilde s = \sin \theta + \delta_s$ and $\tilde c = \cos
\theta + \delta_c$, to first order in the errors,

$$
	\tilde f - f = \frac{\partial_s P\, \delta_s + \partial_c P\, \delta_c}{\theta^k},
$$

so that under absolute errors $\lvert \tilde f - f \rvert \le K\, \varepsilon\, \theta^{-k}$, with $K$
bounded near $0$ and $\varepsilon = \max(\varepsilon_s, \varepsilon_c)$, while under relative ones,
$\lvert \delta_s \rvert \le u\, \theta$ and $\lvert \delta_c \rvert \le u$, the sine's part is
smaller by the factor $\theta$.

Derivatives make it worse. Automatic differentiation evaluates each step's derivative formula at
the computed values, so it differentiates the formula as written, never the rounding, and the
$m$-th derivative of [](eq:closed-form) in $\theta^2$ is another expression of the same form: a
polynomial in $\theta$, $\tilde s$ and $\tilde c$ over $\theta^{k + 2m}$, since each derivative in
$\theta^2$ is $\tfrac{1}{2 \theta} \tfrac{\dd}{\dd \theta}$ and lowers the order of the numerator's
cancellation by two. Its error is therefore at most

$$
	K_m\, \varepsilon\, \theta^{-(k + 2m)} \quad \text{under absolute error}, \qquad K_m\, u\,
	\theta^{1 - (k + 2m)} \quad \text{for the sine's part under relative error}. @tag(derivative-error)
$$

Against a sine of relative accuracy, a sine of absolute accuracy loses the factor $\varepsilon_s /
(u\, \theta)$ in every closed form whose numerator holds a sine with a coefficient that does not
cancel, as $B$'s and $E$'s do, and nothing in those that hold only a cosine, as $A$'s and $D$'s do.
With $\varepsilon_s = 2^{-21}$ in single precision that factor is $8 / \theta$: about $25$ at
$\theta^2 = 0.1$, in a second derivative of $E$ that is already divided by $\theta^9$.

Three kinds of map are spared. One without a cancelling numerator, such as a rotation's matrix,
its action on a vector, or the exponential's $\cos(\theta / 2)$, carries $\varepsilon$ into its
value once. The logarithm's $L$ divides by $n$ the angle of a point, and an $\atantwo$ of relative
accuracy loses nothing. And the argument itself, $\theta$ computed from $\theta^2$ with a relative
error of a few roundings, costs no more than a relatively accurate sine does: where the numerator
and the power $\theta^k$ are formed from the same computed $\tilde\theta$, the closed form is
evaluated consistently at a nearby argument, and its error is $f'(\theta)(\tilde\theta - \theta)$,
bounded near $0$.

## Where to put the cutoff

The closed form is used above a cutoff $\theta_c$ and the series below it. Two conditions decide
$\theta_c$, and they pull in opposite directions.

**The closed form above the cutoff.** Take each order of derivative $m$ that is computed, and each
map that uses the coefficient $f$, and let $w_m(\theta)$ be the factor by which the map multiplies
$f$'s $m$-th derivative. Then

$$
	K_m\, \varepsilon\, \theta_c^{-(k + 2m)}\, w_m(\theta_c) \le \tau, @tag(closed-condition)
$$

where $\tau$ is the error the coefficient may contribute and $\varepsilon$ the absolute error of
the sine and the cosine on the least accurate machine the computation must run on. The bound
decreases in $\theta_c$, so a larger cutoff always meets it, and by [](eq:derivative-error) a
machine with a relatively accurate sine meets it wherever one with the absolute error $\varepsilon$
does. How far a cutoff must rise for a sine of absolute error $\varepsilon$ to do no worse than a
relatively accurate one did at $\theta_c$ is, by the same scaling, the factor $(\varepsilon / (u\,
\theta_c))^{1 / (k + 2m)}$, which is largest where $k + 2m$ is smallest. It is a guide only, since
it drops $K_m$.

**The series below it.** The series truncated after $N$ terms, with remainder $R_N$, must meet the
rounding of the most precise arithmetic in use at the cutoff, in its value and its derivatives:

$$
	\Bigl\lvert \tfrac{\dd^m}{\dd (\theta^2)^m} R_N(\theta_c^2) \Bigr\rvert \le u\, \lvert f(\theta_c)
	\rvert . @tag(truncation-condition)
$$

The series of $b$, $A$, $B$, $D$ and $E$ are entire, with terms falling like $\theta^{2n} / (2n +
k)!$, so every cutoff can be met with enough terms: at $\theta^2 = 1$ in double precision, about
ten. $C$'s series converges for $\theta < 2\pi$, with terms falling by $(\theta / 2\pi)^2$, and that
of $\log(\sin r / r)$ for $r < \pi$, with terms falling by $(r / \pi)^2$. Either radius bounds the
cutoff away from it, and the number of terms grows as the cutoff nears it. A series costs a product
and a sum per term.

**One cutoff serves every machine.** The first condition is hardest in the least precise
arithmetic with the least accurate sine, and the second in the most precise arithmetic. A cutoff
that meets the first for the one and the second for the other meets both everywhere else, so a
formula need not know which machine it runs on. For the entire series such a cutoff always exists.
Raising it costs terms, and nothing else.
