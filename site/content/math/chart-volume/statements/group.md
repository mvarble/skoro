---
type: statement
kind: proposition
title: The volume of a chart on a Lie group
---

Let $G$ be a Lie group of dimension $\dimension$ with a Haar measure $\nu$, invariant under left
and right translations, and let $\varphi : \R^\dimension \to G$ be smooth, with $\varphi(0) = e$,
a diffeomorphism of an open $U \ni 0$ onto an open subset of $G$. For the retraction
$R_\mu(\tangent) = \mu\, \varphi(\tangent)$, and equally for $R_\mu(\tangent) = \varphi(\tangent)\,
\mu$, the chart at every $\mu$ has the domain $U$ and the volume $v_\mu = v_e$, which does not
depend on $\mu$. If $\varphi = \operatorname{Exp}$ and $\nu$ is normalized so that $v_e(0) = 1$, then

$$
	v_e(\tangent) = \det J_r(\tangent) = \prod_{\alpha} \frac{1 - e^{-\alpha}}{\alpha},
	\qquad J_r(\tangent) = \sum_{k \ge 0} \frac{(-\operatorname{ad}_\tangent)^k}{(k + 1)!},
$$

the product taken over the eigenvalues $\alpha$ of $\operatorname{ad}_\tangent$, with multiplicity,
and each factor read as $1$ where $\alpha = 0$.
