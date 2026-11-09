---
type: statement
kind: definition
title: The Fisher linearization of a query
---

Let a query be as in [%full](statement:linearization), except that a term $t$ that scores a
quantity may have its scale reached from free quantities, through coordinates $\theta_t$, and then
binds the information root [](eq:archive/normal/math/root) with scale block $C_{\theta, t}$. For
such a term, let $J^\mu_t$ be the Jacobian of its residual $r_t = W_t\, s_t$ with the scale held
fixed, $J^\mu_t = W_t \D s_t$, and let $K_t = C_{\theta, t} \D \theta_t$ be its scale block
composed with the Jacobian of the scale's coordinates, both in the charts of the free quantities at
$\hat \point$. Let its *scale residual* $u_t$ be the vector, indexed as the rows of
$C_{\theta, t}$, with

$$
	(u_t)_{(a, a)} = \frac{(r_t)_a^2 - 1}{\sqrt 2} , \qquad (u_t)_{(a, b)} = (r_t)_a (r_t)_b \quad (a > b) .
$$

Every other term contributes as in [%full](statement:linearization), and its scale must be fixed.
The query's *Fisher linearization* at $\hat \point$ is

$$
	m_F(\tangent) = c_F + \sum_t \tfrac12 \bigl\lVert J^\mu_t \tangent + r_t + \hat c_t \bigr\rVert^2
	+ \sum_{t \text{ free scale}} \tfrac12 \bigl\lVert K_t \tangent + u_t \bigr\rVert^2 + \eta\transpose \tangent ,
$$

with $J^\mu_t = J_t$ for a term whose scale is fixed, $\hat c_t$ and $\eta$ as in
[%full](statement:linearization), $W_t$ taken at $\hat \point$ where the scale is free, and $c_F$
chosen so that $m_F(0) = f(\hat \point)$.
