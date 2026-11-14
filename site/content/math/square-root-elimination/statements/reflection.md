---
type: statement
kind: lemma
title: Householder reflections
---

For a nonzero $w \in \R^p$, the _Householder reflection_

$$
	H_w = I - \frac{2\, w w\transpose}{\lVert w \rVert^2}
$$

is symmetric and orthogonal. For a nonzero $x \in \R^p$, let
$\alpha = -\operatorname{sgn}(x_1) \lVert x \rVert$, with $\operatorname{sgn}(0) = 1$, and let $w$
equal $x$ except in its first entry, $w_1 = x_1 - \alpha$. Then $w \neq 0$, and

$$
	H_w x = \begin{bmatrix} \alpha \\ 0 \end{bmatrix} ,
$$

the vector with first entry $\alpha$ and every other entry zero.
