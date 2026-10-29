---
type: statement
kind: proposition
title: The chart Hessian and the Hessian of the pull-back
---

Let $\point_t = \point \oplus t \tangent$ and $\varphi_t = R_\point^{-1} \circ R_{\point_t}$, defined
near $0$ for small $t$. Then

$$
	H(\point)\,\tangent = \nabla^2 (f \circ R_\point)(0)\, \tangent + \Bigl(\frac{\dd}{\dd t}\Big|_{t=0} \D\varphi_t(0)\Bigr)\transpose g(\point),
$$

so $H(\point)\,\tangent = \nabla^2 (f \circ R_\point)(0)\, \tangent$ at every stationary point, where
$g(\point) = 0$.
