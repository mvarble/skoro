---
type: statement
kind: proposition
title: A retraction that descends to the quotient
---

Let a finite group $G$ act freely on $\manifold$ by diffeomorphisms, with quotient
$N = \manifold / G$ and quotient map $\varpi$, and let $R$ be a retraction on $\manifold$ with

$$
	R_{g \point}(\tangent) = g\, R_\point(\tangent)
$$

for every $g \in G$, $\point \in \manifold$ and $\tangent$.

1. $[\point] \oplus \tangent \defeq \varpi\bigl(R_\point(\tangent)\bigr)$ is the same whichever
   representative $\point$ is taken, and is a retraction on $N$.
2. Let $Q_\point$ be a map into $\R^\dimension$, defined on a subset of $\manifold$ that $G$ carries
   onto itself, with $Q_\point(R_\point(\tangent)) = \tangent$ for every $\tangent$ in a set $U$, and
   $Q_{g \point}(\otherPoint) = Q_\point(\otherPoint) = Q_\point(g \otherPoint)$. Then
   $\tangent \mapsto [\point] \oplus \tangent$ is injective on $U$, with the inverse
   $[\otherPoint] \ominus [\point] = Q_\point(\otherPoint)$, the same whichever representatives are
   taken.
3. Let $f$ be a smooth map of $\manifold$ into a manifold with a retraction that is invariant,
   $f(g \point) = f(\point)$, so that $f = \bar f \circ \varpi$. The derivative of $\bar f$ at
   $[\point]$ in the charts of $N$, and for a scalar $f$ its gradient and its chart Hessian, are
   those of $f$ at $\point$ in the charts of $\manifold$, the same at every representative.
