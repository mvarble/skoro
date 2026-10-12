---
type: statement
kind: definition
title: A model built forward
---

A _model built forward_ is a finite sequence of _quantities_ $i = 1, \dots, n$, each with

1. a space $(\manifold_i, \mathscr X_i)$ and a $\sigma$-finite _reference measure_ $\nu_i$ on it:
   Lebesgue measure on $\R^{\dimension_i}$ or on an open subset of it, or counting measure on a
   countable set; and
2. a _conditional density_: a measurable function
   $\density_i : \manifold_i \times \manifold_1 \times \cdots \times \manifold_{i-1} \to [0, \infty)$
   such that $\density_i(\cdot \mid \point_{<i})$ is a probability density with respect to $\nu_i$
   for every $\point_{<i} = (\point_1, \dots, \point_{i-1})$.

Its _joint density_ is
$\density(\point) = \prod_{i=1}^n \density_i(\point_i \mid \point_{<i})$, with respect to the
product measure $\nu = \nu_1 \otimes \cdots \otimes \nu_n$ on
$\manifold = \manifold_1 \times \cdots \times \manifold_n$.
