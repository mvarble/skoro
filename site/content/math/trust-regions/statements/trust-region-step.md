---
type: statement
kind: proposition
title: The trust-region step
---

Let $M$ be a symmetric positive semidefinite $\dimension \times \dimension$ matrix, $g \in \R^\dimension$, $D$ a positive diagonal matrix, $\lvert \tangent \rvert_D^2 = \tangent\transpose D \tangent$ and $\Delta > 0$, and let

$$
	m(\tangent) = g\transpose \tangent + \tfrac12 \tangent\transpose M \tangent .
$$

1. If $\lambda \ge 0$ and $\tangent^\ast$ satisfy
    $$
    	(M + \lambda D)\, \tangent^\ast = -g , \qquad \lvert \tangent^\ast \rvert_D \le \Delta , \qquad \lambda \bigl( \Delta - \lvert \tangent^\ast \rvert_D \bigr) = 0 ,
    $$
    then $\tangent^\ast$ minimizes $m$ over the region $\lvert \tangent \rvert_D \le \Delta$.
2. If $M$ is positive definite, such a $\lambda$ and $\tangent^\ast$ exist for every $\Delta > 0$, and are unique.
3. For every $\lambda > 0$, the solution $\tangent(\lambda)$ of $(M + \lambda D)\, \tangent = -g$ minimizes $m$ over the region of radius $\lvert \tangent(\lambda) \rvert_D$, and that radius does not increase with $\lambda$.
