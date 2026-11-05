---
type: statement
kind: theorem
title: The triangular root
---

Let $A$ be an $m \times n$ matrix of rank $n$ and $b \in \R^m$. There is an orthogonal $m \times m$
matrix $Q$ with

$$
	Q\transpose \begin{bmatrix} A & b \end{bmatrix} = \begin{bmatrix} R & d \\ 0 & e \end{bmatrix},
$$

$R$ an $n \times n$ upper-triangular matrix with a nonzero diagonal, $d \in \R^n$ and
$e \in \R^{m - n}$. For every $\tangent \in \R^n$,
$\lVert A \tangent - b \rVert^2 = \lVert R \tangent - d \rVert^2 + \lVert e \rVert^2$, and
$R\transpose R = A\transpose A$.
