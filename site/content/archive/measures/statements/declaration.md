---
type: statement
kind: definition
title: A declared chart volume
---

Let an embedded type $\manifold \subset \R^n$ of dimension $\dimension$ have the retraction $R$ and
the inverse $Q$ of [%full](statement:archive/core/math/embedded), and let $\nu$ be the measure its
retraction normalizes. A _declared chart volume_ for it is a domain and a log-volume, as follows.

The _domain_ assigns to every $\mu \in \manifold$ an open set $U_\mu \subseteq \R^\dimension$
containing $0$ on which $R(\mu, \cdot)$ is a diffeomorphism onto an open set $V_\mu$ of
$\manifold$ whose complement has $\nu$-measure zero, and such that $Q(\mu, R(\mu, \tangent)) =
\tangent$ for every $\tangent \in U_\mu$. It is declared either as _global_, $U_\mu =
\R^\dimension$ at every $\mu$, or by a discretization
$\Omega : \R^n \times \R^\dimension \to \mathrm{Boolean}$ that is true at $(\mu, \tangent)$ exactly
when $\tangent \in U_\mu$.

The _log-volume_ is an operation $\Lambda : \R^n \times \R^\dimension \to \R$ on Euclidean values
such that

$$
	\Lambda(\mu, \tangent) = \lambda_\mu(\tangent) \qquad (\mu \in \manifold,\ \tangent \in U_\mu),
$$

the log-volume of the chart at $\mu$ with respect to $\nu$. It has derivative maps to the order the
type claims, and is smooth as computed on every $U_\mu$, at $\tangent = 0$ in particular.
