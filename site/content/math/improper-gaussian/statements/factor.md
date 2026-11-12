---
type: statement
kind: definition
title: The improper Gaussian factor
---

Let $\point_B = (\point_1, \dots, \point_k)$ be points of manifolds with retractions, with anchors
$p_i$ and the stacked tangent $\delta(\point_B) \in \R^D$ of [](eq:stacked). Given a matrix $A \in
\R^{r \times D}$, for any number $r \ge 0$ of rows, a vector $b \in \R^r$, a vector $\eta \in \R^D$
and a constant $c$, the _improper Gaussian factor_ is the function

$$
	\logterm(\point_B) = -\tfrac12 \bigl\lVert A\, \delta(\point_B) - b \bigr\rVert^2 - \eta\transpose \delta(\point_B) + c .
$$

Its _whitened residual_ is $A \delta - b$, of $r$ components, and its _linear part_ is
$\eta\transpose \delta$.
