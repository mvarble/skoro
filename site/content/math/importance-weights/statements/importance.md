---
type: statement
kind: proposition
title: Importance weights
---

Let the quantities $\point_F$ of a model built forward be free and $\point_O$ observed, with
$\dd\point_F$ the product of the free quantities' reference measures and
$\density(\point_F, \point_O)$ the product of every kernel's density, every likelihood factor and
every score. Suppose that every free quantity has a proposal, of density $\proposal_i$ with respect
to the quantity's reference measure, and that the product $\proposal$ of the proposals is positive
wherever $\density(\cdot, \point_O)$ is. If $\point_F$ is drawn ancestrally from the proposals,
then for every $h$ with $\int \lvert h \rvert\, \density(\point_F, \point_O)\, \dd\point_F < \infty$,

$$
	\E_\proposal\bigl[\weight(\point_F)\, h(\point_F)\bigr] = \int h(\point_F)\, \density(\point_F, \point_O)\, \dd \point_F .
$$

When the model has neither parameters nor likelihood factors, the right side is
$\E[h(\randomPoint_F) \mid \randomPoint_O = \point_O]$ times the marginal density of
$\randomPoint_O$ at $\point_O$. In general it is $Z\, \E_\pi[h]$ whenever
$Z = \int \density(\point_F, \point_O)\, \dd\point_F$ is finite and positive, with $\pi$ the
measure of density $\density(\cdot, \point_O) / Z$.
