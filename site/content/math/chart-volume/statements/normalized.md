---
type: statement
kind: proposition
title: The measure a retraction normalizes
---

Let $\manifold \subset \R^n$ be a submanifold of dimension $\dimension$ with a retraction whose
tangent basis $B(\point) = \partial_\tangent R_\point(0)$ is smooth, and $\sigma$ the
$\dimension$-dimensional surface measure induced by $\R^n$. The volume of the chart at $\mu$ with
respect to $\sigma$ is

$$
	v^\sigma_\mu(\tangent) = \sqrt{\det\bigl(\partial R_\mu(\tangent)\transpose\, \partial R_\mu(\tangent)\bigr)},
$$

with $\partial R_\mu(\tangent) \in \R^{n \times \dimension}$ the Jacobian of $R_\mu$ in storage.
There is exactly one measure $\nu$ on $\manifold$ with a continuous density with respect to
$\sigma$ such that $v_\mu(0) = 1$ at every $\mu$, namely

$$
	\dd \nu(\point) = \det\bigl(B(\point)\transpose B(\point)\bigr)^{-1/2} \, \dd \sigma(\point),
$$

and its chart volumes are $v_\mu = v^\sigma_\mu / v^\sigma_\mu(0)$.
