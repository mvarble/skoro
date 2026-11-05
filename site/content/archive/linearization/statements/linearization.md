---
type: statement
kind: definition
title: The linearization of a query
---

Let a query have no free discrete quantity, let its free continuous quantities, in an order, have
the stacked tangent $\R^\dimension$, and let every term $t$ whose log-density is reached from a
free quantity bind a whitened residual $r_t$, so that $\logterm_t = -\tfrac12 \lVert r_t \rVert^2 +
\rho_t$. With $f = -\log \density$ the query's negative log-density, its *linearization* at a point
$\hat \point$ is the function of $\tangent \in \R^\dimension$

$$
	m(\tangent) = c_0 + \sum_t \tfrac12 \bigl\lVert J_t \tangent + r_t \bigr\rVert^2 + \eta\transpose \tangent ,
$$

where $r_t$ is the residual at $\hat \point$, $J_t = \D r_t(\hat \point)$ its Jacobian in the
charts of the free quantities, $g$ the chart gradient of $f$ at $\hat \point$,

$$
	\eta = g - \sum_t J_t\transpose r_t = -\sum_t \nabla \rho_t(\hat \point),
	\qquad
	c_0 = f(\hat \point) - \tfrac12 \sum_t \lVert r_t \rVert^2 .
$$

Its *chart linearization* is the linearization of $f(\hat \point \oplus \tangent) -
\lambda_{\hat \point}(\tangent)$, with $\lambda_{\hat \point}$ the log-volume of the chart at
$\hat \point$ of the free quantities: the same rows, with $\eta - \nabla \lambda_{\hat \point}(0)$
and $c_0 - \lambda_{\hat \point}(0)$ in place of $\eta$ and $c_0$.
