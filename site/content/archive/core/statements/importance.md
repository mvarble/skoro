---
type: statement
kind: proposition
title: Importance weights
---

Let a query leave the quantities $\point_F$ free and fix $\point_O$, and suppose that every free
quantity is either scored by exactly one term, whose sampler has density $\proposal_t$, or is a
parameter with a proposal of density $\proposal_\theta$; that every one of these densities is
positive wherever the query's density $\density(\cdot, \point_O)$ is; and that each pair of
densities is with respect to the same base measure. If $\point_F$ is drawn ancestrally, then for
every $h$ with $\int \lvert h \rvert\, \density(\point_F, \point_O)\, \dd\point_F < \infty$,

$$
	\E_\proposal\bigl[\weight(\point_F)\, h(\point_F)\bigr] = \int h(\point_F)\, \density(\point_F, \point_O)\, \dd \point_F .
$$

When the model has no parameters, the right side is $\density(\point_O)\, \E[h(\point_F) \mid \point_O]$.
In general it is $Z\, \E_\pi[h]$ whenever
$Z = \int \density(\point_F, \point_O)\, \dd\point_F$ is finite and positive, with $\pi$ the
query's target normalized, which a score of a parameter may prevent.
