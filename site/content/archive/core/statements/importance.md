---
type: statement
kind: proposition
title: Importance weights
---

Let a query leave the quantities $\point_F$ free and fix $\point_O$, and suppose that every free
quantity is scored by exactly one term, whose sampler has density $\proposal_t$ with
$\proposal_t > 0$ wherever $\density_t > 0$, each pair with respect to the same base measure. If
$\point_F$ is drawn ancestrally from the samplers, then for every integrable $h$,

$$
	\E_\proposal\bigl[\weight(\point_F)\, h(\point_F)\bigr] = \int h(\point_F)\, \density(\point_F, \point_O)\, \dd \point_F = \density(\point_O)\; \E\bigl[h(\point_F) \mid \point_O\bigr] .
$$
