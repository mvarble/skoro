---
type: statement
kind: proposition
title: A factor of full rank is a normal
---

As a function of $\delta \in \R^D$, $e^{\logterm}$ is integrable if and only if $A$ has rank $D$.
When it has, let $W$ be the lower-triangular matrix with a positive diagonal and
$W\transpose W = A\transpose A$, and

$$
	m = (A\transpose A)^{-1} (A\transpose b - \eta) , \qquad
	c_{\mathcal N} = c + \tfrac12 \lvert W m \rvert^2 - \tfrac12 \lvert b \rvert^2 - \sum_i \log W_{ii} + \tfrac{D}{2} \log 2\pi .
$$

Then $\logterm(\point_B) = \logterm_{\mathcal N}\bigl(\delta(\point_B);\, m, W\bigr) + c_{\mathcal N}$:
the factor is the normal's `Stacked` replacement [](eq:archive/normal/math/joint) with the mean
$m$, the whitening $W$ and the constant $c_{\mathcal N}$.
