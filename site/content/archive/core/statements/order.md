---
type: statement
kind: definition
title: The order a pass requires
---

Fix a pass over a program: the leaves $L$ it differentiates with respect to, the leaves
$S \subseteq L$ seeded with velocities (for a Hessian pass), and the targets $T$. A step $k$ with map $F_k$
is *on the pass* when its output is continuous and reaches a target in $T$, and one of its
continuous inputs depends on a leaf in $L$. The order the pass requires of step $k$ is

$$
	r_k = \begin{cases}
		0 & \text{if step } k \text{ is not on the pass,} \\
		2 & \text{if it is, the pass is a Hessian pass, and an input of step } k \text{ depends on } S, \\
		1 & \text{otherwise.}
	\end{cases}
$$

The pass is admissible when $o(F_k) \ge r_k$ for every step $k$.
