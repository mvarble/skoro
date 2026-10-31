---
type: statement
kind: definition
title: The volume of a chart
---

Let $\manifold$ be a manifold of dimension $\dimension$ with a retraction and a reference measure
$\nu$, let $\mu \in \manifold$, and let $U_\mu \subseteq \R^\dimension$ be an open set containing
$0$ on which $R_\mu$ is a diffeomorphism onto an open set $V_\mu = R_\mu(U_\mu)$ of $\manifold$. The
_volume_ of the chart at $\mu$ is the function $v_\mu : U_\mu \to \positive$ such that

$$
	\nu\bigl(R_\mu(A)\bigr) = \int_A v_\mu(\tangent) \, \dd \tangent
$$

for every Borel set $A \subseteq U_\mu$, and its _log-volume_ is $\lambda_\mu = \log v_\mu$. The
chart is _global_ if $U_\mu = \R^\dimension$, and _full_ if $\nu(\manifold \setminus V_\mu) = 0$.
