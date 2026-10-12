---
type: statement
kind: corollary
title: Every expectation is an integral against the joint density
---

Let a model built forward be carried by quantities $\bfx = (\bfx_1, \dots, \bfx_n)$ on a probability space
$(\Omega, \mathscr F, \mathrm P)$, so that $\bfx$ has law $\mu$. Every random variable $\bfy$ the model
can express, that is every $\sigma(\bfx)$-measurable one, is $g \circ \bfx$ for a measurable $g$, and when
it is integrable,

$$
	\E[\bfy] = \int_{\manifold} g(\point)\, \density(\point)\, \nu(\dd\point)
	= \int_{\manifold_1} \!\!\cdots \int_{\manifold_n} g(\point) \prod_{i=1}^n \density_i(\point_i \mid \point_{<i})\, \nu_n(\dd\point_n) \cdots \nu_1(\dd\point_1) .
$$
