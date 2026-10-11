---
type: statement
kind: proposition
title: The passes compute chart derivatives
---

Let a program's steps be maps between manifolds, with partial derivatives $\partial_j F$ taken in
tangent charts, and define each step's second-order map at its inputs $s$, with velocities
$\dot s$, by

$$
	\operatorname{hvp}_j(\alpha) = \frac{\mathrm d}{\mathrm dt}\Big|_{t=0} \partial_j F(s \oplus t\dot s)\transpose \alpha .
$$

Then for a scalar target $f$, the reverse pass delivers the chart gradient $g(x)$ at the leaves, and
forward-over-reverse with velocities $v$ delivers the chart Hessian product $H(x)\,v$.
