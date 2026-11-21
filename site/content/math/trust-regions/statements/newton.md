---
type: statement
kind: proposition
title: Newton's method converges quadratically
---

Let $\manifold$ be a manifold with a retraction $\oplus$ that is smooth near the points below,
$f : \manifold \to \R$ smooth, and $g(q)$ and $H(q)$ its chart gradient and chart Hessian at $q$.
Let the iteration be

$$
	q_{k+1} = q_k \oplus \delta_k, \qquad H(q_k)\, \delta_k = -g(q_k) .
$$

If $q^\ast$ is a stationary point of $f$ at which $H(q^\ast)$ is invertible, the iteration converges
to $q^\ast$ from every start near enough to it, and there is a constant $C$ such that
$\lvert q_{k+1} \ominus q^\ast \rvert \le C \lvert q_k \ominus q^\ast \rvert^2$.
