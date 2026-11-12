---
type: statement
kind: definition
title: The Fisher linearization
---

Let $f$ be the negative log-density of a model of normals at a point $\point$ of a product of
manifolds, as in [%full](statement:math/linearization/linearization), except that the scale of a
term $t$ may depend on $\point$ through coordinates $\theta_t$, with the scale block $C_{\theta, t}$
of the root [](eq:math/normal/root). For such a term, write its residual as $r_t = W_t\, s_t$, let
$J^\mu_t = W_t \D s_t$ be the Jacobian of the residual with the scale held fixed, and $J^\theta_t =
C_{\theta, t} \D \theta_t$ its scale block composed with the Jacobian of the scale's coordinates,
both in the charts at $\hat \point$. Let its *scale residual* $u_t$ be the vector, indexed as the
rows of $C_{\theta, t}$, with

$$
	(u_t)_{(a, a)} = \frac{(r_t)_a^2 - 1}{\sqrt 2} , \qquad (u_t)_{(a, b)} = (r_t)_a (r_t)_b \quad (a > b) .
$$

The *Fisher linearization* of $f$ at $\hat \point$ is

$$
	m_F(\tangent) = c_F + \sum_t \tfrac12 \bigl\lVert J^\mu_t \tangent + r_t + \hat c_t \bigr\rVert^2
	+ \sum_{t \text{ free scale}} \tfrac12 \bigl\lVert J^\theta_t \tangent + u_t \bigr\rVert^2 + \eta\transpose \tangent ,
$$

with $J^\mu_t = J_t$ for a term whose scale is fixed, $\hat c_t$ and $\eta$ as in the linearization,
$W_t$ taken at $\hat \point$ where the scale is free, and $c_F$ chosen so that $m_F(0) = f(\hat
\point)$.
