---
type: statement
kind: proposition
title: The normal's expected information
---

Let $\randomPoint \sim \mathcal N(\mu, \Sigma)$ on a product $\manifold$ of Euclidean and positive
factors, with $\Sigma^{-1} = \Lambda = W\transpose W$, and let the scale depend smoothly on
coordinates $\theta \in \R^k$. The expected negative chart Hessian of
$\logterm_{\mathcal N}(\point;\, \mu, W)$ over $(\point, \mu, \theta)$ is

$$
	\mathcal I = \begin{pmatrix}
		\Lambda & -\Lambda & 0 \\
		-\Lambda & \Lambda & 0 \\
		0 & 0 & F
	\end{pmatrix},
	\qquad
	F_{ij} = \tfrac12 \operatorname{tr}\bigl(\Sigma\, \partial_i \Lambda\, \Sigma\, \partial_j \Lambda\bigr),
$$

and $F$ is the Fisher information of $\theta$. Every block is exact, at every $\mu$ and $\Sigma$.
