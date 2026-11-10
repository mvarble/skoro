---
type: statement
kind: corollary
title: Marginals by elimination
---

Let $\phi$ be a sum of factors over blocks $E \cup K$ whose rows have rank $\dimension$, the number
of their columns, so that $e^{-\phi}$ is proportional to a normal density $p$ on $\R^\dimension$.
Eliminate the blocks of $E$ one at a time by part 2 of [%full](statement:eliminate-one), in any
order, and let $\phi_K$ be the sum of the factors left on $K$. Then $e^{-\phi_K}$ is proportional
to the marginal density of $p$ on $K$: its precision is the matrix of $\phi_K$, the Schur
complement of $\Lambda_{EE}$ in $\Lambda$, and its mean is the minimizer of $\phi_K$.
