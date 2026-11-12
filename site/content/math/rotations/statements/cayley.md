---
type: statement
kind: proposition
title: The Cayley chart of the rotations
---

For $q \in S^3$ and $\tangent \in \R^3$, let

$$
	q \oplus \tangent = q \otimes \gamma(\tangent), \qquad
	\gamma(\tangent) = \frac{(1,\ \tangent / 2)}{\sqrt{1 + \lVert \tangent \rVert^2 / 4}} ,
	\qquad
	p \ominus q = \frac{2 \operatorname{vec}(\bar q \otimes p)}{w(\bar q \otimes p)}
	\quad \bigl(w(\bar q \otimes p) \ne 0\bigr) .
$$

Then $(-q) \oplus \tangent = -(q \oplus \tangent)$, and $p \ominus q$ is unchanged when either $p$
or $q$ is negated. The map $\tangent \mapsto \pi(q \oplus \tangent)$ is a diffeomorphism of $\R^3$
onto the rotations by less than $\pi$ from $\pi(q)$, with inverse $R \mapsto p \ominus q$ for either
$p$ over $R$; $\gamma(\tangent)$ is the rotation by $\theta = 2 \arctan(\lVert \tangent \rVert / 2)$
about $\tangent$; $\pi(q \oplus \tangent) = \pi(q)\, (I + [\tangent]_\times) + O(\lVert \tangent
\rVert^2)$; and $q \oplus \tangent = (1 \oplus \pi(q)\, \tangent) \otimes q$.
