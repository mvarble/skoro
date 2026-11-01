---
type: statement
kind: proposition
title: The area formula for a declared volume
---

Let a continuous type be stored in $\R^n$, with reference measure $\dd \nu = h \, \dd \sigma$,
where $\sigma$ is the measure its storage induces and $h$ is continuous and positive: $h = 1$ on
the core's Euclidean and positive types, $h(\point) = \det(B(\point)\transpose B(\point))^{-1/2}$
on an embedded type, and the product of the factors' $h$ on a product. For every $\mu$ and every
$\tangent \in U_\mu$,

$$
	\lambda_\mu(\tangent) = \tfrac12 \log \det\bigl(\partial_2 R(\mu, \tangent)\transpose\, \partial_2 R(\mu, \tangent)\bigr) + \log h\bigl(R(\mu, \tangent)\bigr),
$$

with $\partial_2 R(\mu, \tangent) \in \R^{n \times \dimension}$ the derivative of the retraction in
storage with respect to its tangent argument. If the chart is full, then

$$
	\int_{U_\mu} e^{\lambda_\mu(\tangent)} \, \dd \tangent = \nu(\manifold)
$$

at every $\mu$, finite when $\manifold$ is compact.
