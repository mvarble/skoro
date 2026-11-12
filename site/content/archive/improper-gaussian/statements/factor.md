---
type: statement
kind: definition
title: The improper Gaussian factor
---

Let $B_F = (\point_1, \dots, \point_k)$ be continuous quantities, with anchors $p_i$ constants of
their types and $\delta(\point_B) \in \R^D$ the stacked tangent
[](eq:archive/normal/math/stacked). Given a matrix $A \in \R^{r \times D}$, for any number $r \ge 0$
of rows, a vector $b \in \R^r$, a vector $\eta \in \R^D$ and a constant $c$, the *improper Gaussian
factor* is the likelihood factor

$$
	\logterm(\point_B) = -\tfrac12 \bigl\lvert A\, \delta(\point_B) - b \bigr\rvert^2 - \eta\transpose \delta(\point_B) + c .
$$

Its *whitened residual* is $A \delta - b$, of $r$ components, and its *linear part* is
$\eta\transpose \delta$.
