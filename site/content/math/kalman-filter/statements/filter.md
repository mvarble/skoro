---
type: statement
kind: proposition
title: The Kalman filter
---

Let states $\randomPoint_0, \randomPoint_1, \dots$ in $\R^n$ and measurements $\bfz_0, \bfz_1,
\dots$ follow the linear-Gaussian chain

$$
	\randomPoint_0 \sim \mathcal N(\mu_0, P_0), \qquad
	\bfz_k \mid \randomPoint_k \sim \mathcal N(H_k \randomPoint_k, R_k), \qquad
	\randomPoint_{k+1} \mid \randomPoint_k \sim \mathcal N(F_k \randomPoint_k, Q_k),
$$

with $P_0$, $R_k$ and $Q_k$ positive definite, and let the measurements be observed at $\hat z_0,
\hat z_1, \dots$. Set $\mu_0^- = \mu_0$ and $P_0^- = P_0$, and for each $k$

$$
	\begin{gathered}
		\nu_k = \hat z_k - H_k \mu_k^- , \qquad S_k = H_k P_k^- H_k\transpose + R_k , \qquad K_k = P_k^- H_k\transpose S_k^{-1} , \\
		\mu_k^+ = \mu_k^- + K_k \nu_k , \qquad P_k^+ = (I - K_k H_k) P_k^- , \\
		\mu_{k+1}^- = F_k \mu_k^+ , \qquad P_{k+1}^- = F_k P_k^+ F_k\transpose + Q_k .
	\end{gathered}
$$

Then integrating $\point_0, \dots, \point_{k-1}$ out of the joint density, in that order, leaves on
$\point_k$ the factor

$$
	\log \mathcal N(\point_k;\, \mu_k^-, P_k^-) + \sum_{j < k} \log \mathcal N(\nu_j;\, 0, S_j) ,
$$

and with $\bfz_k$ also observed, the posterior of $\randomPoint_k$ given $\bfz_0, \dots, \bfz_k$ is
$\mathcal N(\mu_k^+, P_k^+)$.
