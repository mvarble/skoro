---
type: statement
kind: proposition
title: Retirement leaves the rest of the model unchanged
---

Let $E$ be the quantities to retire, of which $E_O$ are fixed at given values and $E_F$ are
integrated out, let $\mathcal T_E$ be the terms joined to a quantity of $E$ in the probabilistic
graph, and let $B$ be the other quantities joined to those terms. Replace the terms $\mathcal T_E$
by the single term

$$
	\logterm_B(\point_B) = \log \int \exp \Bigl(\sum_{t \in \mathcal T_E} \logterm_t\bigl(\point_{E_F}, \point_{E_O}, \point_B\bigr)\Bigr)\, \dd \point_{E_F} .
$$

Then, as a function of every quantity outside $E$, the new model's density equals the old model's
density with $E_F$ integrated out and $E_O$ fixed at its values.
