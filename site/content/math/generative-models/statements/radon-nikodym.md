---
type: statement
kind: theorem
title: Radon–Nikodym
---

Let $\nu$ be a $\sigma$-finite measure on a measurable space $(\manifold, \mathscr X)$, and $\mu$ a
finite measure on it with $\mu \ll \nu$, that is, $\nu(A) = 0$ implies $\mu(A) = 0$. Then there is
a measurable $\density : \manifold \to [0, \infty)$ with

$$
	\mu(A) = \int_A \density \,\dd\nu \qquad \text{for every } A \in \mathscr X,
$$

and any two such functions agree $\nu$-almost everywhere. One writes $\density = \dd\mu / \dd\nu$
and $\mu(\dd\point) = \density(\point)\, \nu(\dd\point)$.
