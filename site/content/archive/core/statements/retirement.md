---
type: statement
kind: proposition
title: Retirement leaves the rest of the model unchanged
---

Let $E$ be the quantities to retire, of which $E_O$ are fixed at given values and $E_F$ are
integrated out, let $\mathcal T_E$ be the terms that involve a quantity of $E$, and let $B$ be the
other quantities those terms involve. Replace the terms $\mathcal T_E$ by the single term

$$
	\ell_B(x_B) = \log \int \exp \Bigl(\sum_{t \in \mathcal T_E} \ell_t\bigl(x_{E_F}, x_{E_O}, x_B\bigr)\Bigr)\, \mathrm d x_{E_F} .
$$

Then, as a function of every quantity outside $E$, the new model's density equals the old model's
density with $E_F$ integrated out and $E_O$ fixed at its values.
