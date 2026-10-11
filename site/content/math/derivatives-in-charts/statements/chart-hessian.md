---
type: statement
kind: proposition
title: The chart Hessian and the Hessian of the pull-back
---

Let $x_t = x \oplus t v$ and $\varphi_t = R_x^{-1} \circ R_{x_t}$, defined near $0$ for small $t$.
Then

$$
	H(x)\,v = \nabla^2 (f \circ R_x)(0)\, v + \Bigl(\frac{\mathrm d}{\mathrm dt}\Big|_{t=0} \mathrm D\varphi_t(0)\Bigr)\transpose g(x),
$$

so $H(x)\,v = \nabla^2 (f \circ R_x)(0)\, v$ at every stationary point, where $g(x) = 0$.
