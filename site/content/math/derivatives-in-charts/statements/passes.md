---
type: statement
kind: proposition
title: The passes compute chart derivatives
---

Let a program's steps be maps between manifolds, with partial derivatives $\partial_j \map$ taken in
tangent charts, and define each step's second-order map at its inputs $s$, with velocities
$\tangent_s$, by

$$
	\hvp_j(\alpha) = \frac{\dd}{\dd t}\Big|_{t=0} \partial_j \map(s \oplus t\tangent_s)\transpose \alpha .
$$

Then for a scalar target $f$, the reverse pass delivers the chart gradient $g(\point)$ at the
leaves, and forward-over-reverse with velocities $\tangent$ delivers the chart Hessian product
$H(\point)\,\tangent$.
