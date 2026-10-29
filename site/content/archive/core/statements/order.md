---
type: statement
kind: definition
title: The order a pass requires
---

Fix a pass over a program: the roots $L$ it differentiates with respect to, the roots
$S \subseteq L$ seeded with velocities (for a Hessian pass), and the targets $T$. A step $k$ with map
$\map_k$ is *on the pass* when, in the program's computation graph, its output is continuous and
reaches a target in $T$, and one of its continuous inputs is reached from a root in $L$. The order
the pass requires of step $k$ is

$$
	\req(k) = \begin{cases}
		0 & \text{if step } k \text{ is not on the pass,} \\
		2 & \text{if it is, the pass is a Hessian pass, and an input of step } k \text{ is reached from } S, \\
		1 & \text{otherwise.}
	\end{cases}
$$

The pass is admissible when $\order(\map_k) \ge \req(k)$ for every step $k$.
