---
type: statement
kind: lemma
title: Reflections that leave a leading block alone
---

Let $w = (0; w') \in \R^p$, with $0 \in \R^i$ and $w' \in \R^{p - i}$ nonzero. Then

$$
	H_w = \begin{bmatrix} I & 0 \\ 0 & H_{w'} \end{bmatrix},
	\qquad
	H_w \begin{bmatrix} v \\ u \end{bmatrix} = \begin{bmatrix} v \\ H_{w'} u \end{bmatrix}
$$

for every $v \in \R^i$ and $u \in \R^{p - i}$. In particular, $H_w$ leaves the first $i$ entries
of every vector unchanged, and fixes every vector whose other entries are zero:

$$
	H_w \begin{bmatrix} v \\ 0 \end{bmatrix} = \begin{bmatrix} v \\ 0 \end{bmatrix} .
$$
