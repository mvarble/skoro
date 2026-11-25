---
type: statement
kind: proposition
title: The curvature sets the rate, the gradient the limit
---

Let $\manifold$, $\oplus$, $f$ and $g$ be as in [%full](statement:newton), and let $M(q)$ be a matrix that depends continuously on $q$ and is invertible near the iterates of

$$
	q_{k+1} = q_k \oplus \delta_k, \qquad M(q_k)\, \delta_k = -g(q_k) .
$$

Then $\delta_k = 0$ if and only if $q_k$ is a stationary point of $f$.
If $q^\ast$ is a stationary point, $H^\ast$ the chart Hessian there and $M^\ast = M(q^\ast)$, and the spectral radius $\rho$ of $I - (M^\ast)^{-1} H^\ast$ is less than $1$, the iteration converges to $q^\ast$ from every start near enough to it, and $\limsup_k \lvert q_{k+1} \ominus q^\ast \rvert / \lvert q_k \ominus q^\ast \rvert \le \rho$ in any norm for which the spectral radius is attained.
