---
type: statement
kind: proposition
title: The normal equations
---

Let $A \in \R^{m \times \dimension}$ have rank $\dimension$, $b \in \R^m$, $\eta \in \R^\dimension$, and

$$
	\phi(\tangent) = \tfrac12 \lvert A \tangent - b \rvert^2 + \eta\transpose \tangent .
$$

Then $\phi$ has exactly one minimizer $\tangent^\ast$, the solution of the _normal equations_

$$
	A\transpose A\, \tangent^\ast = A\transpose b - \eta ,
$$

and for every $\tangent$

$$
	\phi(\tangent) = \phi(\tangent^\ast) + \tfrac12 (\tangent - \tangent^\ast)\transpose A\transpose A\, (\tangent - \tangent^\ast) ,
	\qquad
	\int_{\R^\dimension} e^{-\phi(\tangent)} \, \dd \tangent = (2\pi)^{\dimension / 2} \det (A\transpose A)^{-1/2}\, e^{-\phi(\tangent^\ast)} .
$$
