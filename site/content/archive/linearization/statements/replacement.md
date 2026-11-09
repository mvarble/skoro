---
type: statement
kind: proposition
title: The linearized replacement
---

Let a retirement retire the quantities $E$, with $E_O$ given values and $E_F$ continuous, of types
with a volume and of total dimension $d_E$, and let $B_F$ be continuous quantities of its blanket
$B$, of total dimension $D$.
Let $m_T$ be the linearization at $\hat \point = (\hat \point_E, \hat \point_B)$ of the query of the
terms $\mathcal T_E$ alone, free in $E_F$ and $B_F$ and observing every other quantity, taken with
the chart's log-volume of $E_F$ alone, and let the columns of its rows for $E_F$ have rank $d_E$.
Then, for every $\tangent_B \in \R^D$,

$$
	\log \int_{\R^{d_E}} e^{-m_T(\tangent_E, \tangent_B)} \, \dd \tangent_E
	= -\tfrac12 \lVert A \tangent_B - b \rVert^2 - \eta\transpose \tangent_B + c ,
$$

where $A$ has full row rank $r \le D$, $\eta$ lies in the null space of $A$, and $A$, $b$,
$\eta$ and $c$ are those of [](eq:improper). Replacing the integrand of the replacement
$\logterm_B$ by $e^{-m_T}$ gives $\logterm_B(\hat \point_B \oplus \tangent_B)$ approximately as
this right-hand side.

If the rows have rank $d_E + D$, the number of their columns, then $r = D$ and $\eta = 0$. The
right-hand side is then $\logterm_{\mathcal N}(\tangent_B;\, m, W) + \log Z$, with
$Z = \int e^{-m_T}$, and with $m$ the mean and $W$ the lower-triangular whitening, with a positive
diagonal, of the marginal on $\tangent_B$ of the normal density proportional to $e^{-m_T}$.
