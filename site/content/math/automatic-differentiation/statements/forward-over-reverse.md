---
type: statement
kind: proposition
title: Forward-over-reverse computes Hessian products
---

Let $f$ be a scalar target, seed its cotangent with $\alpha_f = 1$ and $\beta_f = 0$, and seed the
leaves with velocities $\dot X$. Then the reverse recurrences deliver, at every leaf $x_i$,

$$
	\alpha_{x_i} = \nabla_{x_i} f(x), \qquad \beta_{x_i} = \sum_l \nabla_{x_i}\nabla_{x_l} f(x)\,\dot X_l .
$$
