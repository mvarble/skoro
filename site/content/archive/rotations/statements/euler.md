---
type: statement
kind: proposition
title: The Euler angles of a unit quaternion
---

Let $(i, j, k)$ be axes with $i \ne j \ne k$, let $l$ be the axis other than $i$ and $j$, and let
$\epsilon = \pm 1$ with $e_i \times e_j = \epsilon\, e_l$. Write $q_e(\theta) = (\cos \tfrac\theta2,
\sin \tfrac\theta2\, e)$.

1. *Proper angles*, $k = i$. If $q = q_i(\alpha) \otimes q_j(\beta) \otimes q_i(\gamma)$ with
   $\beta \in [0, \pi]$, then, with $\sigma = (\alpha + \gamma)/2$ and $\delta = (\alpha - \gamma)/2$,

   $$
   	(w,\ u_i) = \cos \tfrac\beta2\, (\cos \sigma,\ \sin \sigma), \qquad
   	(u_j,\ \epsilon\, u_l) = \sin \tfrac\beta2\, (\cos \delta,\ \sin \delta) .
   $$

   So for $\beta \in (0, \pi)$ the angles are recovered, modulo $2\pi$, from any positive multiple
   of $q$ or of $-q$ as $\beta = 2 \operatorname{atan2}\bigl(\lvert (u_j, u_l) \rvert,\ \lvert (w,
   u_i) \rvert\bigr)$, $\alpha = \sigma + \delta$ and $\gamma = \sigma - \delta$, where $\sigma =
   \operatorname{atan2}(u_i, w)$ and $\delta = \operatorname{atan2}(\epsilon\, u_l, u_j)$.

2. *Tait–Bryan angles*, $k = l$. If $q = q_i(\alpha) \otimes q_j(\beta) \otimes q_k(\gamma)$ with
   $\beta \in [-\tfrac\pi2, \tfrac\pi2]$, then $q \otimes q_j(\tfrac\pi2)$ has the proper angles
   $(\alpha,\ \beta + \tfrac\pi2,\ -\epsilon \gamma)$ about $(i, j, i)$, and is a positive multiple of
   $(w - u_j,\ u_i - \epsilon u_k,\ u_j + w,\ u_k + \epsilon u_i)$, listed as the coordinates
   $(w, u_i, u_j, u_k)$.
