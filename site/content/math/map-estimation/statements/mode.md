---
type: statement
kind: definition
title: The maximum a posteriori estimate
---

Let a model built forward have free quantities $F$ and observed ones $O$, observed at $\point_O$, with the joint density $\density$ against the product $\nu = \nu_F \otimes \nu_O$ of reference measures, and let $0 < \density_O(\point_O) < \infty$.
The _objective_ is the negative logarithm of the joint density at the observations,

$$
	f(\point_F) = -\log \density(\point_F, \point_O) ,
$$

and a _maximum a posteriori estimate_, or _mode_ of the posterior, is a point $\hat \point_F$ at which $f$ is least: $f(\hat \point_F) \le f(\point_F)$ for every $\point_F$.
A _local_ mode is a point at which $f$ is least among the points near it.
