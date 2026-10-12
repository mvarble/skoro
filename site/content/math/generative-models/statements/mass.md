---
type: statement
kind: proposition
title: A mass function is its measure on points
---

Let $\manifold = \{\point_0, \point_1, \dots\}$ be countable, with every subset measurable, and let
$\nu = \sum_k \delta_{\point_k}$ be counting measure. Every probability measure $\mu$ on
$\manifold$ is absolutely continuous with respect to $\nu$, its density is unique, and it is

$$
	\density(\point_k) = \mu(\{\point_k\}), \qquad \text{so that} \qquad
	\mu(\dd\point) = \sum_k \density(\point_k)\, \delta_{\point_k}(\dd\point) .
$$
