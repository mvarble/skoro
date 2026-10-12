---
type: statement
kind: proposition
title: Importance weights
---

Let a query leave the quantities $x_F$ free and fix $x_O$, and suppose that every free quantity is
scored by exactly one term, whose sampler has density $q_t$ with $q_t > 0$ wherever $\density_t > 0$, each
pair with respect to the same base measure. If $x_F$ is drawn ancestrally from the samplers, then
for every integrable $h$,

$$
	\E_q\bigl[w(x_F)\, h(x_F)\bigr] = \int h(x_F)\, \density(x_F, x_O)\, \mathrm d x_F = \density(x_O)\; \E\bigl[h(x_F) \mid x_O\bigr] .
$$
