---
summary: How the absolute error of a GPU's sine and cosine reaches the closed forms of the removable singularities and their derivatives, measured on a wgpu adapter; the error the cutoffs are designed for and how a device is admitted to it; and the conditions a cutoff and its series must meet so that single precision on the GPU agrees with double precision as it does on the CPU.
depends_on: [archive/rotations/math, archive/measures/math]
---

## The problem

The rotations' exponential, logarithm and Jacobians, and the exponential test sphere's
log-volume, are smooth at the identity, but their closed forms divide by a power of the angle there.
Each is computed, as [Removable singularities](../rotations/math.svx#removable-singularities) has
it, by a truncated series below a cutoff and by its closed form above it. The cutoffs were chosen so
that the closed form just above the cutoff, its value and its first and second derivatives, agrees
in single precision with double precision, and that choice was measured on the CPU. On a wgpu
adapter three checks of that agreement fail: the second-order pass of the pose's left Jacobian in
the rotations' tests, and the second-order map of the exponential sphere's log-volume at $t = 0$ in
the core's. Every other check passes there.

This writeup finds why, by measuring the adapter's elementary functions, shows how their error
reaches a closed form and its derivatives, states the error the design assumes of a device and how
a device is admitted to it, and states the conditions a cutoff and its series must meet for single
precision on the GPU to agree with double precision as closely as it does on the CPU. It decides no cutoff: as the rotations' mathematics left them, the numbers are settled by the
implementation, against measurements, under the conditions found here.

## Setting

A computed value carries an error, the difference between what single precision returns and the
exact value at the same, already rounded, input. Write $u = 2^{-24} \approx 6.0 \cdot 10^{-8}$ for
single precision's unit roundoff. A function has **relative error** $\eta$ when its computed value
is $f(x)(1 + \delta)$ with $\lvert \delta \rvert \le \eta$, and **absolute error** $\varepsilon$
when it is $f(x) + \delta$ with $\lvert \delta \rvert \le \varepsilon$. The two differ only where
$f(x)$ is small, which is exactly where a removable singularity is computed: $\sin \theta \approx
\theta$ near $0$.

The coefficients are those of the rotations' table, each a function of $\theta^2$: $b = \sin(\theta /
2) / \theta$; $A = (1 - \cos \theta) / \theta^2$; $B = (\theta - \sin \theta) / \theta^3$; $C$,
with $\cot(\theta/2)$; and the motions' $D$ and $E$, whose closed forms divide by $\theta^4$ and
$\theta^5$. The sphere's is $\Lambda(r) = \log(\sin r / r)$ with $r^2 = \lVert \tangent \rVert^2$,
from [the measures' exponential sphere](../measures/math.svx). Their derivatives are taken by the
core's passes: each primitive's derivative is a formula ($\sin' = \cos$, $\log' = 1/x$) evaluated at
the computed values, so a pass differentiates the formula as written, never the rounding.

## The adapter's elementary functions

Measured on an NVIDIA GeForce RTX 4070 Ti through wgpu, against double precision at the same
single-precision inputs, beside burn's Flex device on the CPU in single precision:

| function | inputs | CPU, single | GPU, single |
| --- | --- | --- | --- |
| $\sin x$ | $10^{-7} \le x < 10^{-4}$ | relative $1.7 \cdot 10^{-9}$ | absolute $1.9 \cdot 10^{-7}$, relative up to $1$ |
| $\sin x$ | $10^{-4} \le x < 10^{-2}$ | relative $5.7 \cdot 10^{-8}$ | absolute $1.8 \cdot 10^{-7}$, relative $1.5 \cdot 10^{-3}$ |
| $\sin x$ | $10^{-2} \le x < \pi$ | relative $5.5 \cdot 10^{-8}$ | absolute $3.6 \cdot 10^{-7}$ |
| $\sin x$ | $10 \le x < 100$ | relative $5.0 \cdot 10^{-8}$ | absolute $7.8 \cdot 10^{-6}$ |
| $\cos x$ | $10^{-7} \le x < \pi$ | absolute $3.1 \cdot 10^{-8}$ | absolute $3.8 \cdot 10^{-7}$ |
| $\exp x$, $\log x$ | $\lvert x \rvert < 1.7$, $10^{-7} < x < 3.2$ | relative $6.7 \cdot 10^{-8}$ | absolute $1.5 \cdot 10^{-7}$ near $x = 1$, relative below $2 \cdot 10^{-7}$ elsewhere |
| $\operatorname{atan2}(x, 1)$ | $10^{-7} \le x < \pi$ | relative $5.6 \cdot 10^{-8}$ | relative $1.1 \cdot 10^{-7}$ |

The CPU's sine is accurate relative to its value, as a library function is. The GPU's is accurate
only to an absolute $\varepsilon_s \approx 4 \cdot 10^{-7} \approx 2^{-21}$ on $[-\pi, \pi]$,
whatever the argument, and worse beyond, where its argument is reduced. Its cosine is likewise
absolute, at $\varepsilon_c \approx 4 \cdot 10^{-7}$, which near $\cos x \approx 1$ is a relative
error of a few roundings and hardly differs from the CPU's. Its exponential, logarithm and
$\operatorname{atan2}$ are relative, within a few roundings. The WGSL specification promises less
still for the sine and cosine, an absolute error of $2^{-11}$ on $[-\pi, \pi]$, and nothing outside
it: a bound under which single precision agrees with double precision to $10^{-4}$ nowhere, not even
in a rotation's matrix.

## How the error reaches a closed form

Every closed form here is

$$
	f(\theta) = \frac{P\bigl(\theta, \sin \theta, \cos \theta\bigr)}{\theta^k}, \qquad P(\theta, \sin
	\theta, \cos \theta) = O(\theta^k), @tag(closed-form)
$$

with $P$ a polynomial whose own partial derivatives in its second and third arguments do not vanish
at $0$: $B$ has $P = \theta - \sin \theta$ and $k = 3$, $E$ has $P = 2\theta - 3 \sin \theta + \theta
\cos \theta$ and $k = 5$. The numerator cancels to $O(\theta^k)$; the error of $\sin$ and $\cos$
does not. With the computed $\tilde s = \sin \theta + \delta_s$ and $\tilde c = \cos \theta +
\delta_c$, to first order in the errors,

$$
	\tilde f - f = \frac{\partial_s P\, \delta_s + \partial_c P\, \delta_c}{\theta^k},
$$

so that under the GPU's absolute errors $\lvert \tilde f - f \rvert \le K\, \varepsilon\, \theta^{-k}$,
with $K$ bounded near $0$ and $\varepsilon = \max(\varepsilon_s, \varepsilon_c)$, while under the
CPU's relative ones, $\lvert \delta_s \rvert \le u\, \theta$ and $\lvert \delta_c \rvert \le u$, the
sine's part is smaller by the factor $\theta$.

A pass computes the $m$-th derivative of [](eq:closed-form) in $\theta^2$ as another expression of
the same form, a polynomial in $\theta$, $\tilde s$ and $\tilde c$ over $\theta^{k + 2m}$, since each
derivative in $\theta^2$ is $\tfrac{1}{2 \theta} \tfrac{\dd}{\dd \theta}$ and lowers the order of the
numerator's cancellation by two. Its error is therefore

$$
	\Bigl\lvert \tfrac{\dd^m}{\dd (\theta^2)^m} (\tilde f - f) \Bigr\rvert \le K_m\, \varepsilon\,
	\theta^{-(k + 2m)} \quad \text{on the GPU}, \qquad K_m\, u\, \theta^{1 - (k + 2m)} \quad
	\text{for the sine's part on the CPU}. @tag(derivative-error)
$$

The GPU loses, against the CPU, the factor $\varepsilon_s / (u\, \theta)$, about $7 / \theta$, in every
closed form whose numerator holds a sine, and nothing in those that hold only a cosine. The
measurements bear both out. At $\theta^2 = 0.1$, the errors of the closed forms' second derivatives
in $\theta^2$, against double precision, are:

| coefficient | $k$ | holds | CPU | GPU | $\varepsilon\, \theta^{-(k + 4)}$ |
| --- | --- | --- | --- | --- | --- |
| $b$ | $1$ | $\sin$ | $1.6 \cdot 10^{-6}$ | $1.0 \cdot 10^{-6}$ | $1.3 \cdot 10^{-4}$ |
| $A$ | $2$ | $\cos$ | $4.9 \cdot 10^{-5}$ | $7.7 \cdot 10^{-5}$ | $4 \cdot 10^{-4}$ |
| $B$ | $3$ | $\sin$ | $1.3 \cdot 10^{-4}$ | $1.2 \cdot 10^{-3}$ | $1.3 \cdot 10^{-3}$ |
| $C$ | $2$ | $\sin$, $\cos$ of $\theta / 2$ | $1.9 \cdot 10^{-4}$ | $1.5 \cdot 10^{-4}$ | $4 \cdot 10^{-4}$ |
| $D$ | $4$ | $\cos$ | $4.4 \cdot 10^{-3}$ | $5.0 \cdot 10^{-3}$ | $4 \cdot 10^{-3}$ |
| $E$ | $5$ | $\sin$, $\cos$ | $6.4 \cdot 10^{-3}$ | $4.9 \cdot 10^{-2}$ | $1.3 \cdot 10^{-2}$ |

The last column is the scale of [](eq:derivative-error) without its constant. $B$ and $E$, whose
numerators carry a sine with a coefficient that does not cancel, lose a factor of eight to ten on
the GPU, as [](eq:derivative-error) predicts at $\theta \approx 0.32$; $A$ and $D$,
of cosines alone, do not. These are the coefficients of the pose's Jacobian, whose second-order pass
is the one that fails. A coefficient reaches a map multiplied by powers of $\theta$, which is why
errors of $10^{-2}$ in $E$'s second derivative are tolerable on the CPU, but the factor of ten is
not, and it carries the pass past the agreement of $10^{-4}$.

The same factor explains the sphere. Its $\Lambda(r) = \log(\sin r / r)$ divides a sine by $r$ and
its series is used below $r^2 = 10^{-8}$; the check at $t = 0$ compares the second-order map, taken
from the series, with central differences of the cotangent map at single precision's step of
$10^{-2}$, where $r^2 = 10^{-4}$ and the closed form is used, and on the GPU the closed form's
derivatives there are wrong by $K_m \varepsilon\, r^{-(1 + 2m)}$, far beyond the check's tolerance.
On the CPU the same derivatives are smaller by the factor $u\, r / \varepsilon_s$, about $10^{-3}$. At $r^2 = 0.1$ its second derivative's
error on the GPU is $1.7 \cdot 10^{-5}$ against a value of $10^{-2}$.

Maps without a cancelling numerator are not affected: a rotation's matrix, its action, the
exponential's $\cos(\theta / 2)$, the elementary rotations of Euler angles, each carries the absolute
$\varepsilon$ into its value once, below $10^{-6}$. The logarithm's $L = 2 \operatorname{atan2}(n, w)
/ n$ divides by $n$ the angle of a point, which the GPU computes to its relative accuracy, so it
loses nothing either.

## The error designed for

The cutoffs cannot be designed for the specification's bound. Under an absolute error of
$2^{-11}$ no map of a rotation agrees with double precision to $10^{-4}$, closed form or not, and
not even a rotation's matrix. Nor can they be designed for the one adapter measured, since other
adapters, and other drivers of this one, may differ. They are designed instead for a stated bound on
the absolute error of the sine and the cosine,

$$
	\varepsilon_\star = 2^{-20} \approx 9.5 \cdot 10^{-7} \quad \text{on } [-\pi, \pi], \qquad
	\varepsilon_\star^{\text{far}} = 2^{-16} \approx 1.5 \cdot 10^{-5} \quad \text{beyond}, @tag(design-error)
$$

about $2.5$ times the error measured on $[-\pi, \pi]$ and twice the error measured between $10$ and
$100$. The two bounds differ because only the first is amplified. A closed form evaluates its sine
and cosine beyond $\pi$ only where $\theta > \pi$, where the factor $\theta^{-(k + 2m)}$ of
[](eq:derivative-error) is below $\pi^{-(k + 2m)}$ and amplifies nothing. There, as in a map
without a cancelling numerator, the error reaches a value about once, and $\varepsilon_\star^{\text{far}}$
leaves room under the agreement of $10^{-4}$.

The sine's and cosine's parts of the errors in [](eq:derivative-error) are linear in $\varepsilon$,
and the rest does not depend on it. At $\varepsilon_\star$, then, the GPU's errors are at most
$\varepsilon_\star / \hat\varepsilon \approx 2.5$ times those measured, with $\hat\varepsilon$ the
measured error. That factor, applied to the measurements, is what the conditions below are checked
against. The constant $K_m$ is left out of the bound, and the measurements stand in for it.

The exponential, the logarithm and $\operatorname{atan2}$ need no bound of their own, because the
cancelling numerators do not amplify their errors as they amplify the sine's. A closed form's
$\theta = e^{\frac12 \log \theta^2}$ is computed with a relative error $\eta$, of a few roundings,
and so is a nearby argument $\tilde\theta = \theta(1 + \eta)$. Where the numerator and the power
$\theta^k$ are both formed from $\tilde\theta$, the closed form is evaluated consistently there, and
its error is $f'(\theta)(\tilde\theta - \theta)$, bounded near $0$. Where a power is formed from the
exact $\theta^2$ instead, the numerator's $\eta\, \theta$ is left uncancelled. Divided by $\theta^k$,
it gives an error of order $\eta\, \theta^{1 - k}$. That is the order of the CPU's own sine error,
$u\, \theta^{1-k}$, larger only by the few roundings $\eta / u$, and never by the factor
$\varepsilon / (u\, \theta)$. Either way, the design asks of them only the relative accuracy
measured.

**Admission.** The checks that compare single precision with double precision rest on
[](eq:design-error), so a device must show that it meets it. With $X$ a finite set of
single-precision inputs, and the exact values taken in double precision, whose own error of
$2^{-53}$ is negligible, the device is admitted when

$$
	\hat\varepsilon = \max_{x \in X \cap [-\pi, \pi]} \max\bigl(\lvert \widetilde{\sin}\, x - \sin x
	\rvert, \lvert \widetilde{\cos}\, x - \cos x \rvert\bigr) \le \varepsilon_\star, @tag(admission)
$$

and the same maximum over $X$ beyond $\pi$ is at most $\varepsilon_\star^{\text{far}}$. A device
that is not admitted fails there. The failure names $\hat\varepsilon$ and the bound it exceeds,
rather than a coefficient's second derivative several steps later. Admission is necessary, not
sufficient: $\hat\varepsilon$ is a maximum over a sample and can miss the worst input. The room
between the bound and the error measured is what absorbs the inputs the sample misses, on an
adapter like the one measured. $X$ therefore covers the arguments the closed forms and the maps
evaluate, near $0$ as densely as near $\pi$, as the measurements above did.

## Conditions on a cutoff

The coefficient's closed form is used above its cutoff $\theta_c$ and its series below it. Two
conditions decide $\theta_c$, and the first is new.

**The closed form above the cutoff.** Take each coefficient $f$, each order $m \le 2$, and each map
that uses $f$, and let $w_m(\theta)$ be the factor by which the map multiplies $f$'s $m$-th
derivative. Then

$$
	K_m\, \varepsilon_\star\, \theta_c^{-(k + 2m)}\, w_m(\theta_c) \le \tau, @tag(closed-condition)
$$

where $\tau$ is the share of the agreement of single with double precision that the coefficient
may spend. The bound decreases in $\theta_c$, so a larger cutoff always meets it. By
[](eq:derivative-error), it is also met on the CPU wherever it is met on the GPU.

The scaling of [](eq:derivative-error) suggests how far a cutoff holding a sine must rise for its
error on the GPU at $\varepsilon_\star$ to be no larger than its error on the CPU at the present
$\theta_c^2 = 0.1$. The factor is $(\varepsilon_\star / (u\, \theta_c))^{1 / (k + 2m)}$. At $m = 2$
it is about $1.75$ for $B$, giving $\theta_c^2 \approx 0.31$, and $1.55$ for $E$, giving
$\theta_c^2 \approx 0.24$. The factor is largest where $k + 2m$ is smallest. It is only a guide: it
drops $K_m$ and the CPU's errors from sources other than the sine, and for $b$, where it predicts
$\theta_c^2 \approx 0.48$, the measurements show the GPU's error already below the CPU's. The
measurements decide. At $\theta^2 = 1$ the GPU's errors in the second derivatives in $\theta^2$
were $3.7 \cdot 10^{-7}$ for $B$, $3.8 \cdot 10^{-7}$ for $D$ and $1.5 \cdot 10^{-6}$ for $E$.
Scaled to $\varepsilon_\star$, they are at most $4 \cdot 10^{-6}$, two to three orders below what
the CPU reaches at the present cutoff.

**The series below it.** The series truncated after $N$ terms, with remainder $R_N$, must meet
double precision's rounding at the cutoff, in its value and its first two derivatives:

$$
	\Bigl\lvert \tfrac{\dd^m}{\dd (\theta^2)^m} R_N(\theta_c^2) \Bigr\rvert \le 2^{-53}\, \lvert f(\theta_c)
	\rvert, \qquad m \le 2. @tag(truncation-condition)
$$

The series of $b$, $A$, $B$, $D$ and $E$ are entire, with terms falling like $\theta^{2n} / (2n +
k)!$, so every cutoff can be met with enough terms: at $\theta^2 = 1$, about ten. $C$'s series is
that of $\tfrac\theta2 \cot \tfrac\theta2$. It converges for $\theta < 2\pi$, with terms falling by
$(\theta / 2\pi)^2$. $\Lambda$'s series, $\log(\sin r / r) = -\tfrac{r^2}{6} - \tfrac{r^4}{180} -
\tfrac{r^6}{2835} - \cdots$, converges for $r < \pi$, with terms falling by $(r / \pi)^2$. Either
radius bounds the cutoff away from it, and the number of terms grows as the cutoff nears it. A
series costs a product and a sum per term, on every device.

**Every coefficient has a cutoff meeting both.** For $b$, $A$, $B$, $D$ and $E$, $\theta_c^2 = 1$
meets [](eq:closed-condition) at $\varepsilon_\star$ by the measurements just scaled, and
[](eq:truncation-condition) with about ten terms. $C$, of sines and cosines of $\theta / 2$ with
$k = 2$, was measured no worse on the GPU than on the CPU at $\theta^2 = 0.1$. Its series at
$\theta_c^2 = 1$ falls by $(2\pi)^{-2} \approx 0.025$ a term, so it reaches double precision's
rounding in about a dozen. The sphere needs its cutoff moved furthest. Its present one, $r^2 =
10^{-8}$, leaves the closed form in use at the check's $r^2 = 10^{-4}$, where its second
derivative's error is of the order of $\varepsilon_\star\, r^{-5} \approx 10^{4}$. At $r^2 = 0.1$ the
error measured, $1.7 \cdot 10^{-5}$, is at most $4.3 \cdot 10^{-5}$ at $\varepsilon_\star$. That is well
inside the check's $5 \cdot 10^{-3}$, which is relative to the larger of the value and one and so,
for values this small, absolute. There the series falls by $0.1 / \pi^2 \approx 0.01$ a term
and reaches double precision's rounding in about eight. Below such a cutoff the check's central
differences, at $r^2 = 10^{-4}$, are taken from the series, as the second-order map is.

One cutoff still serves every target, since a composite sees neither its precision nor its
device. The conditions are met for the worst of them: [](eq:design-error) in single precision for
[](eq:closed-condition), and double precision for [](eq:truncation-condition). Every other target,
the CPU's in either precision and an admitted GPU's in double, meets them a fortiori. The archive's
cutoff for single precision, $\theta^2 \approx 1.37$, with its series through $\theta^8$, was of
this size.

## In skoro's terms

Nothing random or estimated changes, and no map changes its value in exact arithmetic: the series
and the closed form agree, and only the point where one replaces the other moves. What changes is
the computation of a composite's value, and of its derivatives of both orders, on a target whose
elementary functions have absolute error. The core's passes inherit that error by evaluating each
primitive's derivative formula at the computed values. The checks that compare single precision
with double precision, on both sides of each cutoff, hold on the GPU once two things are true: the
cutoffs meet [](eq:closed-condition) at $\varepsilon_\star$, and the device is admitted by
[](eq:admission). The bounds of [](eq:design-error) are a property of the target that the design
assumes, stated once. They are not a tolerance of any one check.
