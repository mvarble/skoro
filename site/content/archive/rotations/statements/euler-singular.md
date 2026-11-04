---
type: statement
kind: proposition
title: No continuous Euler angles
---

Fix a sequence of axes $(i, j, k)$, and call a rotation *at the lock* when its $\beta$ is $0$ or
$\pi$ for a proper sequence, $\pm\tfrac\pi2$ for a Tait–Bryan one.

1. Away from the lock, $\alpha$ and $\gamma$ are smooth maps of the rotation into the circle
   $\R / 2\pi\Z$, but neither is represented by a continuous real-valued map on any set of
   rotations containing the loop $t \mapsto \pi\bigl(q_i(t) \otimes q_j(\beta) \otimes
   q_k(\gamma)\bigr)$, $t \in [0, 2\pi]$, for $\gamma$ the loop in its own angle, with $\beta$ off
   the lock.
2. At a rotation at the lock, neither $\alpha$ nor $\gamma$ has a limit, even in the circle. For a
   proper sequence, $\alpha + \gamma$ has one at $\beta = 0$ and $\alpha - \gamma$ at $\beta = \pi$;
   for a Tait–Bryan sequence, $\alpha - \epsilon \gamma$ has one at $\beta = -\tfrac\pi2$ and
   $\alpha + \epsilon \gamma$ at $\beta = \tfrac\pi2$, with $\epsilon$ as in
   the proposition on the Euler angles of a unit quaternion.
