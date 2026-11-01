---
type: statement
kind: proposition
title: A density pushed through a chart
---

With the chart of [%full](statement:chart-volume), let $\varepsilon$ be a random vector of
$\R^\dimension$ with density $f$, and $\varepsilon \in U_\mu$ almost surely. Then
$\randomPoint = \mu \oplus \varepsilon$ has, with respect to $\nu$, the density

$$
	\density(\point) = \frac{f(\point \ominus \mu)}{v_\mu(\point \ominus \mu)}
	= \exp\bigl(\log f(\point \ominus \mu) - \lambda_\mu(\point \ominus \mu)\bigr)
$$

for $\point \in V_\mu$, and $\density(\point) = 0$ elsewhere.
