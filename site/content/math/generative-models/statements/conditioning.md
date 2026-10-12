---
type: statement
kind: proposition
title: Conditioning is division of densities
---

Split the quantities of a model built forward into observed ones $O$ and free ones $F$, and let

$$
	\density_O(\point_O) = \int \density(\point_F, \point_O)\, \nu_F(\dd\point_F) .
$$

Then $\density_O$ is a density of the law of $\bfx_O$, finite $\nu_O$-almost everywhere, and the
kernel

$$
	\kernel(\point_O, \dd\point_F) = \frac{\density(\point_F, \point_O)}{\density_O(\point_O)}\, \nu_F(\dd\point_F)
	\qquad \text{where } 0 < \density_O(\point_O) < \infty,
$$

defined arbitrarily elsewhere, is a regular conditional distribution of $\bfx_F$ given $\bfx_O$.
