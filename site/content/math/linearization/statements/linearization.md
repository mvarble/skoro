---
type: statement
kind: definition
title: The linearization of a sum of squares
---

Let $f$ be as in [](eq:objective), and call the log-volume $\lambda_t$ _folded_ when the residual
is $r_t = W_t\, s_t$ for a constant invertible $W_t$ and $\lambda_t$ is a function of $s_t$ alone.
For a folded term set $\hat c_t = W_t^{-\top} \nabla \lambda_t(s_t)$, the gradient of the log-volume
with respect to the residual, and otherwise set $\hat c_t = 0$. The _linearization_ of $f$ at a
point $\hat \point$ is the function of $\tangent \in \R^\dimension$

$$
	m(\tangent) = c_0 + \sum_t \tfrac12 \bigl\lVert J_t \tangent + r_t + \hat c_t \bigr\rVert^2 + \eta\transpose \tangent ,
$$

where

$$
	\eta = \sum_{t \text{ not folded}} \nabla \lambda_t(\hat \point)
	\qquad \text{and} \qquad
	c_0 = f(\hat \point) - \tfrac12 \sum_t \lVert r_t + \hat c_t \rVert^2 ,
$$

with $r_t$ the residual at $\hat \point$, and $J_t = \D r_t(\hat \point)$ and $\nabla \lambda_t$
the Jacobian and the gradient in the charts at $\hat \point$.
