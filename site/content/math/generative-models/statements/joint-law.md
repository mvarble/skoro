---
type: statement
kind: proposition
title: The joint law of a model built forward
---

The measure $\mu(\dd\point) = \density(\point)\, \nu(\dd\point)$ is a probability measure on
$\manifold$. For each $i$, its marginal on the first $i$ quantities has density
$\prod_{j \le i} \density_j(\point_j \mid \point_{<j})$ with respect to
$\nu_1 \otimes \cdots \otimes \nu_i$, and

$$
	\kernel_i(A \mid \point_{<i}) = \int_A \density_i(\point_i \mid \point_{<i})\, \nu_i(\dd\point_i)
$$

is a version of $\rmP_{\randomPoint_i \mid \randomPoint_{<i}}$, the regular conditional distribution of the
$i$-th quantity given the ones before it.
