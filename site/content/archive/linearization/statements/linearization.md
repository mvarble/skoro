---
type: statement
kind: definition
title: The linearization of a query
---

Let a query have no free discrete quantity, let its free continuous quantities, in an order, have
the stacked tangent $\R^\dimension$, and let every term $t$ whose log-density is reached from a
free quantity bind a whitened residual $r_t$, of any number of components, none included, with

$$
	\logterm_t = -\tfrac12 \lvert r_t \rvert^2 + n_t - \lambda_t ,
$$

$n_t$ constant in the query and $\lambda_t$ the node the term binds under the log-volume role, or
zero where it binds none. Call the log-volume *folded* when the residual is $r_t = W_t\, s_t$ for a
constant invertible $W_t$ and $\lambda_t$ is a function of $s_t$ alone, and set
$\hat c_t = W_t^{-\top} \nabla \lambda_t(s_t)$, the gradient of the log-volume with respect to the
residual; otherwise set $\hat c_t = 0$. With $f = -\log \density$ the query's negative log-density,
its *linearization* at a point $\hat \point$ is the function of $\tangent \in \R^\dimension$

$$
	m(\tangent) = c_0 + \sum_t \tfrac12 \bigl\lvert J_t \tangent + r_t + \hat c_t \bigr\rvert^2 + \eta\transpose \tangent ,
$$

where

$$
	\eta = \sum_{t \text{ not folded}} \nabla \lambda_t(\hat \point)
	\qquad \text{and} \qquad
	c_0 = f(\hat \point) - \tfrac12 \sum_t \lvert r_t + \hat c_t \rvert^2 ,
$$

where $r_t$ is the residual at $\hat \point$ and $J_t = \D r_t(\hat \point)$ and $\nabla
\lambda_t$ are the Jacobian and the gradient in the charts of the free quantities. Its *chart
linearization* is the linearization of $f(\hat \point \oplus \tangent) - \lambda_{\hat
\point}(\tangent)$, with $\lambda_{\hat \point}$ the log-volume of the chart at $\hat \point$ of
the free quantities: the same rows, with $\eta - \nabla \lambda_{\hat \point}(0)$ and
$c_0 - \lambda_{\hat \point}(0)$ in place of $\eta$ and $c_0$.
