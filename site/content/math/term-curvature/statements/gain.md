---
type: statement
kind: proposition
title: The gain of a step near a minimum
---

Let $q^\ast$ be a stationary point of $f$, $H^\ast$ the chart Hessian of $f$ there, and let the
quadratic model at $\hat \point$ have a positive definite matrix $M$, continuous in $\hat \point$.
For $\lambda \ge 0$ and a positive diagonal $D$ let
$\tangent$ solve $(M + \lambda D)\, \tangent = -g$, and let $\Delta = \tfrac12 \tangent\transpose M
\tangent + \lambda\, \tangent\transpose D \tangent$ be the decrease of the model it promises. Then,
as $\hat \point \to q^\ast$ with $\lambda$ bounded,

$$
	\varrho_\lambda = \frac{f(\hat \point) - f(\hat \point \oplus \tangent)}{\Delta} = 1 + \frac{\tangent\transpose (M - H^\ast)\, \tangent}{2 \Delta} + O\bigl(\lvert \hat \point \ominus q^\ast \rvert\bigr) , @tag(gain-limit)
$$

and for the undamped step, $\lambda = 0$, $\varrho_0 = 2 - \tangent\transpose H^\ast \tangent /
\tangent\transpose M \tangent + O(\lvert \hat \point \ominus q^\ast \rvert)$.
