---
type: statement
kind: proposition
title: A likelihood factor is a change of measure
---

Let $L : \manifold \to [0, \infty)$ be measurable with $0 < Z = \int L \,\dd\mu < \infty$. Then
$\mu_L(\dd\point) = L(\point)\, \mu(\dd\point) / Z$ is a probability measure with density
$L \density / Z$ with respect to $\nu$. If $L(\point) = h(\otherPoint_0 \mid \point)$ for a
conditional density $h$ of a further quantity $Y$ appended to the model, then $\mu_L$ is the
conditional law of $X$ given $Y = \otherPoint_0$, as conditioning by densities gives it.
