---
type: statement
kind: proposition
title: The stereographic chart of the unit quaternions
---

For $q \in S^3$ and $\tangent \in \R^3$, with $s = \lVert \tangent \rVert^2 / 16$, let

$$
	q \oplus \tangent = q \otimes \delta(\tangent), \qquad
	\delta(\tangent) = \frac{\bigl(1 - s,\ \tangent / 2\bigr)}{1 + s} .
$$

Then $\tangent \mapsto q \oplus \tangent$ is a diffeomorphism of $\R^3$ onto $S^3 \setminus \{-q\}$,
whose inverse is $p \mapsto 4\, \operatorname{vec}(\bar q \otimes p) / (1 + w(\bar q \otimes p))$;
$\delta(\tangent)$ is the rotation by the angle $\theta = 4 \arctan(\lVert \tangent \rVert / 4)$
about $\tangent$; and $\pi(q \oplus \tangent) = \pi(q)\,\bigl(I + [\tangent]_\times\bigr) +
O(\lVert \tangent \rVert^2)$, so that $\tangent$ is, to first order, the rotation vector of the
perturbation in the frame of $q$.
