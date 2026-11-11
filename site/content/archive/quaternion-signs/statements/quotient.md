---
type: statement
kind: proposition
title: A type stored up to a group
---

Let $\manifold \subset \R^n$, $R$, $Q$ and $P$ be as in [%full](statement:archive/core/math/embedded),
and let $G$ be a finite group of linear maps of $\R^n$ that carry $\manifold$ onto itself and act
on it freely, such that for every $g \in G$, $\point, \otherPoint \in \manifold$ and $\tangent$,

$$
	R(g \point, \tangent) = g\, R(\point, \tangent), \qquad
	Q(g \point, \otherPoint) = Q(\point, \otherPoint) = Q(\point, g \otherPoint), \qquad
	P(g w) = g\, P(w) ,
$$

wherever either side is defined. Then $N = \manifold / G$ is a manifold of dimension $\dimension$,
the quotient map $\varpi : \manifold \to N$ is a covering, and $[\point] \oplus \tangent =
\varpi(R(\point, \tangent))$ and $[\otherPoint] \ominus [\point] = Q(\point, \otherPoint)$ are a
retraction on $N$ and its inverse, the same whichever representatives are taken. For a function
$f$ on $\manifold$ invariant under $G$, $f = \bar f \circ \varpi$, the chart derivatives of $f$ at
$\point$ through $R$, of every order, are those of $\bar f$ at $[\point]$ through the retraction
on $N$, and are the same at every representative $g \point$.
