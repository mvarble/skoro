---
type: statement
kind: proposition
title: Calibration under the model
---

Let $(\randomPoint, \bfz)$ be drawn from a joint law whose conditional law of $\randomPoint$ given
$\bfz = z$ is the kernel $\kernel(\cdot \mid z)$, and for each $z$ let $C(z)$ be a set with
$\kernel(C(z) \mid z) = \alpha$, chosen so that $\{(\point, z) : \point \in C(z)\}$ is measurable.
Then $\Pr(\randomPoint \in C(\bfz)) = \alpha$. If $(\randomPoint_r, \bfz_r)$, $r = 1, \dots, N$,
are independent draws from the joint law, the indicators $\mathbb 1[\randomPoint_r \in
C(\bfz_r)]$ are independent Bernoulli variables of mean $\alpha$.
