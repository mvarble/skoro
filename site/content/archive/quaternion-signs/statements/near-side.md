---
type: statement
kind: proposition
title: The sign-free normal by its near side
---

In the setting of [%full](statement:sign-free), let $s = \operatorname{sign} \langle q_R, \mu_R
\rangle$ for the quaternions $q_R$ and $\mu_R$ of $q$ and $\mu$, defined where they are not
orthogonal, and let $\tangent = (\tangent_R, \tangent_T) = (s q) \ominus \mu$, its rotation's
coordinate first and the rest after. Then $\lVert \tangent_R \rVert \le 4$,

$$
	\log \density^{\pm}_\mu(q) = \log \density_\mu(s q) + \log\bigl(1 + \rho(\tangent)\bigr) - \log 2 ,
$$

and, with $c = 16 / \lVert \tangent_R \rVert^2$ and $\tilde\tangent = (-c\, \tangent_R, \tangent_T)$,

$$
	\rho(\tangent) = \frac{\density_\mu(-s q)}{\density_\mu(s q)}
	= c^3 \exp\Bigl( -\tfrac12 \bigl( \lVert W \tilde\tangent \rVert^2 - \lVert W \tangent \rVert^2 \bigr) \Bigr) .
$$

The function $\log \density^{\pm}_\mu$ is smooth on the whole space. If $\Sigma$ has no block
coupling the rotation to the rest, and the precision of its rotation's block is at least
$\tfrac{3}{16} I$, then $\rho \le 1$, so that

$$
	\log \density_\mu(s q) - \log 2 \le \log \density^{\pm}_\mu(q) \le \log \density_\mu(s q) ,
$$

and $\rho(\tangent) \to 0$ faster than any power of $\lVert \tangent_R \rVert$ as
$\tangent_R \to 0$.
