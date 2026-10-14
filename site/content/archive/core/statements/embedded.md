---
type: statement
kind: proposition
title: The boundary of an embedded type
---

Let $\manifold \subset \R^n$ be a submanifold of dimension $\dimension$, and
$R : U \to \R^n$ and $Q : V \to \R^\dimension$ smooth maps on open sets
$U \subseteq \R^n \times \R^\dimension$ and $V \subseteq \R^n \times \R^n$, such that for every
$\point \in \manifold$:

- $(\point, 0) \in U$, $R(\point, 0) = \point$, $R(\point, \tangent) \in \manifold$ whenever
  $(\point, \tangent) \in U$, and $B(\point) = \partial_2 R(\point, 0) \in \R^{n \times \dimension}$
  has rank $\dimension$;
- $Q(\point, R(\point, \tangent)) = \tangent$ for every $\tangent$ in a neighbourhood of $0$;

and, optionally, $P : W \to \manifold$ a smooth map on an open set $W \subseteq \R^n$ containing
$\manifold$, with $P(\point) = \point$ for every $\point \in \manifold$.

Then $\point \oplus \tangent = R(\point, \tangent)$ is a retraction on $\manifold$, where it is
defined, with $T_\point \manifold$ identified with $\R^\dimension$ by the basis $B(\point)$, and
$\otherPoint \ominus \point = Q(\point, \otherPoint)$ for $\otherPoint$ near $\point$. Write
$L(\otherPoint) = \partial_2 Q(\otherPoint, \otherPoint)$, the derivative of $Q$ in its second
argument on the diagonal. In these charts the inclusion $\iota : \manifold \to \R^n$ has the
derivative maps

$$
	\D \iota(\point) = B(\point), \qquad
	\hvp_\iota(\alpha)\, u = \bigl(\D B(\point)[\dot \point]\bigr)\transpose \alpha, @tag(inclusion)
$$

and the retraction, as a map $\manifold \times \R^\dimension \to \manifold$ with
$\otherPoint = \point \oplus \tangent$, has the partial derivatives

$$
	\partial_\point (\point \oplus \tangent) = L(\otherPoint)\, \partial_1 R(\point, \tangent)\, B(\point), \qquad
	\partial_\tangent (\point \oplus \tangent) = L(\otherPoint)\, \partial_2 R(\point, \tangent) @tag(retraction-maps)
$$

and the second-order maps

$$
	\begin{aligned}
		\hvp_\point(\alpha)\,(u, v) &= B(\point)\transpose \dot g_1 + \bigl(\D B(\point)[\dot \point]\bigr)\transpose g_1, \\
		\hvp_\tangent(\alpha)\,(u, v) &= \dot g_2,
	\end{aligned}
	\qquad
	\dot g_j = \partial_j R\transpose\, \bigl(\D L(\otherPoint)[\dot \otherPoint]\bigr)\transpose \alpha
	+ \bigl(\D \partial_j R\, [\dot \point, v]\bigr)\transpose L(\otherPoint)\transpose \alpha, @tag(retraction-second)
$$

where $u$ and $v$ are the velocities of $\point$ and $\tangent$, $\dot \point = B(\point)\, u$ and
$\dot \otherPoint = \partial_1 R\, \dot \point + \partial_2 R\, v$ are the velocities of $\point$ and
$\otherPoint$ in $\R^n$, $g_j = \partial_j R\transpose L(\otherPoint)\transpose \alpha$, every
$\partial_j R$ is taken at $(\point, \tangent)$, and $\D \partial_j R\,[\dot \point, v]$ is its
derivative along $(\dot \point, v)$. When $P$ is given, the projection $\pi : W \to \manifold$,
$\pi(w) = P(w)$, has, with $\otherPoint = P(w)$ and $\dot \otherPoint = \D P(w)\, \dot w$ for a
velocity $\dot w$ of $w$, the derivative maps

$$
	\D \pi(w) = L(\otherPoint)\, \D P(w), \qquad
	\hvp_\pi(\alpha)\, \dot w = \bigl(\D^2 P(w)[\dot w]\bigr)\transpose L(\otherPoint)\transpose \alpha
	+ \D P(w)\transpose \bigl(\D L(\otherPoint)[\dot \otherPoint]\bigr)\transpose \alpha, @tag(projection)
$$

where $\D^2 P(w)[\dot w]$ is the derivative of $\D P$ along $\dot w$.

Each of these derivative maps holds with $B$ and $L$ replaced by any smooth maps $\hat B$ and
$\hat L$, on open sets of $\R^n$ containing $\manifold$, that agree with them on $\manifold$.
