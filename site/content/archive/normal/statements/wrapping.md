---
type: statement
kind: proposition
title: The mass an exponential chart wraps
---

Let $\manifold$ be $SO(3)$, $SE(3)$ or $SE_2(3)$ with its exponential chart, whose domain is
$U = \{\tangent : \lVert \tangent_\phi \rVert < \pi\}$, $\tangent_\phi$ the rotation's coordinates,
let $\varepsilon$ be normal with mean zero and covariance $\Sigma$, and let $\sigma^2$ be the largest
eigenvalue of the rotation's block $\Sigma_{\phi\phi}$. Then

$$
	p = \Pr(\varepsilon \notin U) \le \operatorname{erfc}\frac{\pi}{\sigma \sqrt 2}
	+ \sqrt{\frac 2 \pi}\, \frac{\pi}{\sigma}\, e^{-\pi^2 / 2 \sigma^2},
$$

with equality when $\Sigma_{\phi\phi} = \sigma^2 I$.
