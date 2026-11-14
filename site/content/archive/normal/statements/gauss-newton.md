---
type: statement
kind: proposition
title: Gauss–Newton with a scalar term
---

Let $\manifold$ be a manifold with a retraction, $f(q) = \tfrac12 \lvert r(q) \rvert^2 + s(q)$ with
$r : \manifold \to \R^m$ and $s : \manifold \to \R$ smooth, and $J(q) = \D r(q)$ and
$g(q) = J\transpose r + \D s\transpose$ the Jacobian of $r$ and the gradient of $f$ in the chart at
$q$. Let the iteration be

$$
	q_{k+1} = q_k \oplus \delta_k, \qquad J\transpose J\, \delta_k = -g(q_k) ,
$$

with $J\transpose J$ invertible near the iterates. Then $\delta_k = 0$ if and only if $q_k$ is a
stationary point of $f$. If $q^*$ is a stationary point, $H^*$ the chart Hessian of $f$ there and
$E = H^* - J\transpose J$ at $q^*$, and the spectral radius $\rho$ of $(J\transpose J)^{-1} E$ is
less than $1$, then the iteration converges to $q^*$ from every start near enough to it, and
$\limsup_k \lvert q_{k+1} \ominus q^* \rvert / \lvert q_k \ominus q^* \rvert \le \rho$ in any norm
for which the spectral radius is attained.
