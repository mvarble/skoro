---
type: statement
kind: lemma
title: Forms of one factor
---

Each of the following changes $(A, b, \eta, c)$ and leaves the value of the factor unchanged at
every $\point_B$:

1. for an orthogonal $Q \in \R^{r \times r}$, $(Q\transpose A,\, Q\transpose b,\, \eta,\, c)$;
2. if $[A \;\, b] = \begin{bmatrix} A' & b' \\ 0 & e \end{bmatrix}$, with zero rows in $A$ below
   $A'$, $(A',\, b',\, \eta,\, c - \tfrac12 \lvert e \rvert^2)$;
3. for any $u \in \R^r$, $(A,\, b - u,\, \eta - A\transpose u,\, c - u\transpose b + \tfrac12 \lvert u \rvert^2)$.

Moving the anchor $p_i$ of each point $\point_i$ of a space that is flat in disguise to
$p_i' = p_i \oplus w_i$, and keeping the anchors of the others, with $w \in \R^D$ stacked from the
$w_i$ and zero in the blocks of the anchors kept, also leaves it unchanged with
$(A,\, b - A w,\, \eta,\, c - \eta\transpose w)$.
