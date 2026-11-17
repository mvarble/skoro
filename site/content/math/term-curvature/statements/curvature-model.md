---
type: statement
kind: definition
title: A quadratic model from curvatures
---

Let $f = \sum_t \phi_t$ on a product of manifolds with retractions, each $\phi_t$ smooth and
reading only its variables $K_t$, and let $g_t$ be the chart gradient of $\phi_t$ at $\hat \point$,
nonzero only in the coordinates of $K_t$. A _curvature_ of the term $t$ at $\hat \point$ is a
symmetric positive semidefinite matrix $M_t$, nonzero only in the block of $K_t$. A _root_ of it
is a triple $(C_t, u_t, \eta_t)$ of a matrix, a vector and a linear term such that

$$
	C_t\transpose C_t = M_t \qquad \text{and} \qquad C_t\transpose u_t + \eta_t = g_t . @tag(root)
$$

The _quadratic model_ of $f$ at $\hat \point$ with these curvatures is the function of
$\tangent \in \R^\dimension$

$$
	m(\tangent) = c + \sum_t \Bigl( \tfrac12 \bigl\lvert C_t \tangent + u_t \bigr\rvert^2 + \eta_t\transpose \tangent \Bigr) , @tag(model)
$$

with $c$ chosen so that $m(0) = f(\hat \point)$.
