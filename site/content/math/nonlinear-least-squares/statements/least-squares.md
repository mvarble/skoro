---
type: statement
kind: proposition
title: The mode of a model of normals is a least-squares problem
---

Let every term of a model built forward be a normal
$\mathcal N\bigl(\mu_t(\point), \Sigma_t\bigr)$ on a space with a global chart, scoring one
quantity $\point_{s_t}$, whether free or observed, with a mean $\mu_t$ computed smoothly from other
quantities and a scale $\Sigma_t$ that is fixed, with the whitening $W_t$. Then the objective of
[%full](statement:math/map-estimation/mode) is

$$
	f(\point_F) = \sum_t \Bigl( \tfrac12 \lvert r_t(\point_F) \rvert^2 + \lambda_t(\point_F) \Bigr) + \text{a constant},
	\qquad r_t = W_t \bigl( \point_{s_t} \ominus \mu_t(\point) \bigr) ,
$$

with $\lambda_t$ the log-volume of the chart at $\mu_t$, read at $\point_{s_t} \ominus \mu_t$, and the
observed quantities held at their values. Its minimizers are those of a sum of squared residuals,
corrected by the log-volumes.
