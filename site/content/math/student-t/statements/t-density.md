---
type: statement
kind: proposition
title: The density of Student's t
---

Let $\manifold$, $\mu$ and $W$ be as for the normal, with the chart at $\mu$ global, and let
$\varepsilon = L \zeta$ with $L = W^{-1}$ and $\zeta$ a Student t vector of $\R^\dimension$ with
$n > 0$ degrees of freedom and unit scale. Then $\mu \oplus \varepsilon$ has, with respect to
$\nu$, the log-density

$$
	\logterm_{t}(\point;\, \mu, W, n) = -\frac{n + \dimension}{2} \log\Bigl(1 + \frac{\lvert r \rvert^2}{n}\Bigr)
	+ \sum_{i=1}^{\dimension} \log W_{ii} + k_{n, \dimension} - \lambda_\mu(\point \ominus \mu) , @tag(log-density)
$$

with $r = W (\point \ominus \mu)$ and the constant

$$
	k_{n, \dimension} = \log \Gamma\Bigl(\frac{n + \dimension}{2}\Bigr) - \log \Gamma\Bigl(\frac{n}{2}\Bigr) - \frac{\dimension}{2} \log (n \pi) . @tag(constant)
$$
