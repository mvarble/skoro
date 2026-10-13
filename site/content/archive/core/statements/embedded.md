---
type: statement
kind: proposition
title: The boundary of an embedded type
---

Let $\manifold \subset \R^n$ be a submanifold of dimension $\dimension$, $U, W \subseteq \R^n$
open sets, and

- $B : U \to \R^{n \times \dimension}$ smooth, with orthonormal columns spanning the tangent space
  $T_\point \manifold$ at every $\point \in \manifold \cap U$;
- $P : W \to \manifold \cap U$ smooth, with $\manifold \cap U \subseteq W$ and $P(\point) = \point$
  for every $\point \in \manifold \cap U$.

Then $\point \oplus \tangent = P(\point + B(\point)\, \tangent)$, where it is defined, is a
retraction on $\manifold \cap U$, whose tangent coordinates at $\point$ are the coordinates in the
basis $B(\point)$. In its charts, the inclusion $\iota : \manifold \cap U \to \R^n$ and the
projection $\pi : W \to \manifold$, $\pi(w) = P(w)$, have the derivative maps

$$
	\D \iota(\point) = B(\point), \qquad
	\hvp_\iota(\alpha)\, \tangent = \bigl(\D B(\point)[B(\point)\, \tangent]\bigr)\transpose \alpha, @tag(inclusion)
$$

$$
	\D \pi(w) = B(\otherPoint)\transpose\, \D P(w), \qquad
	\hvp_\pi(\alpha)\, \dot w = \bigl(\D^2 P(w)[\dot w]\bigr)\transpose B(\otherPoint)\, \alpha
	+ \D P(w)\transpose\, \D B(\otherPoint)\bigl[\D P(w)\, \dot w\bigr]\, \alpha, @tag(projection)
$$

where $\otherPoint = P(w)$, $\D B(\point)[u]$ is the derivative of $B$ along $u \in \R^n$, and
$\D^2 P(w)[\dot w]$ that of $\D P$ along $\dot w$.
