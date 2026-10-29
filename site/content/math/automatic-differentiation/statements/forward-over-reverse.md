---
type: statement
kind: proposition
title: Forward-over-reverse computes Hessian products
---

Let $f$ be a scalar target, seed its cotangent with $\alpha_f = 1$ and $\beta_f = 0$, and seed the
roots with velocities $\tangent$. Then the reverse recurrences deliver, at every root $\point_i$,

$$
	\alpha_{\point_i} = \nabla_{\point_i} f(\point), \qquad \beta_{\point_i} = \sum_l \nabla_{\point_i}\nabla_{\point_l} f(\point)\,\tangent_{\point_l} .
$$
