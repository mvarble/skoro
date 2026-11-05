---
type: statement
kind: lemma
title: Eliminating one block
---

Let

$$
	\phi(\tangent_k, \tangent_S) = \tfrac12 \lVert A_k \tangent_k + A_S \tangent_S - b \rVert^2
	+ \eta_k\transpose \tangent_k + \eta_S\transpose \tangent_S ,
$$

with $A_k$ of $m$ rows and of rank $d_k$, and let an orthogonal $Q$ bring the rows to

$$
	Q\transpose \begin{bmatrix} A_k & A_S & b \end{bmatrix}
	= \begin{bmatrix} R & T & d \\ 0 & A' & b' \end{bmatrix},
$$

with $R$ a $d_k \times d_k$ upper-triangular matrix with a nonzero diagonal. With
$y = R^{-\top} \eta_k$,

$$
	\phi(\tangent_k, \tangent_S)
	= \tfrac12 \bigl\lVert R \tangent_k + T \tangent_S - (d - y) \bigr\rVert^2
	+ \phi'(\tangent_S) + \gamma,
	\qquad
	\phi'(\tangent_S) = \tfrac12 \lVert A' \tangent_S - b' \rVert^2 + (\eta_S - T\transpose y)\transpose \tangent_S,
$$

and $\gamma = y\transpose d - \tfrac12 \lVert y \rVert^2$. Consequently:

1. for every $\tangent_S$, $\min_{\tangent_k} \phi = \phi'(\tangent_S) + \gamma$, attained only at
   $\tangent_k = R^{-1}(d - y - T \tangent_S)$;
2. for every $\tangent_S$,
   $\int_{\R^{d_k}} e^{-\phi(\tangent_k, \tangent_S)}\, \dd \tangent_k = (2\pi)^{d_k / 2}\, \lvert \det R \rvert^{-1}\, e^{-\phi'(\tangent_S) - \gamma}$;
3. with $\Lambda = [A_k \;\, A_S]\transpose [A_k \;\, A_S]$ in blocks, the matrix of $\phi'$ is the
   Schur complement, $A'^\top A' = \Lambda_{SS} - \Lambda_{Sk} \Lambda_{kk}^{-1} \Lambda_{kS}$, and
   its gradient at zero is $g_S - \Lambda_{Sk} \Lambda_{kk}^{-1} g_k$, with $g$ the gradient of
   $\phi$ at zero.
