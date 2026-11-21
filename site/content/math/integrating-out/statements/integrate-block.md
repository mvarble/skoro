---
type: statement
kind: proposition
title: Integrating a block out leaves an improper factor
---

Let $\logterm$ be an improper Gaussian factor over the blocks $(\point_k, \point_S)$, with the stacked tangent $\delta = (\tangent_k, \tangent_S)$ at its anchors, so that $-\logterm + c = \phi(\tangent_k, \tangent_S)$ is the function of [%full](statement:math/square-root-elimination/eliminate-one), and let the columns $A_k$ of its rows for block $k$ have rank $d_k$.
With the notation of that lemma,

$$
	\log \int_{\R^{d_k}} e^{\logterm}\, \dd \tangent_k
	= -\tfrac12 \lvert A' \tangent_S - b' \rvert^2 - (\eta_S - T\transpose y)\transpose \tangent_S
	+ c - \gamma + \tfrac{d_k}{2} \log 2\pi - \log \lvert \det R \rvert ,
$$

an improper Gaussian factor over $\point_S$ at the same anchors, whatever the rank of $A'$.
If $A_k$ has rank less than $d_k$, the integral is infinite for every $\tangent_S$.
