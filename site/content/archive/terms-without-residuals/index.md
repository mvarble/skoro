---
title: "A place in a linearization for a term that binds no residual, and damped Fisher scoring"
pages: []
crates: []
revises: [linearization]
---

This entry sets out to let a model whose terms are not all normals be linearized, and a model with
a free scale reach a mode by a damped method. `skoro-linearize` refuses a model with any term that
binds no residual (`NoResidual`), and the roles it reads are the normal's: one term that is not a
normal, such as the half-Cauchy prior on the scale that eight schools usually carries, makes the
whole model unlinearizable, and with it the sequence of a mode, a Laplace approximation and
Hamiltonian Monte Carlo started from them; robust losses, which real pose-graph data needs, have no
route either. The entry gives such a term a place in a linearization, and weighs three routes to it:
the term's gradient and Hessian products through the core's second-order passes, which would bring
the Newton step and the observed information with them; a distribution that declares its expected
information in its parents' coordinates, lifted by their Jacobians as the normal's information root
is; and the whitened residual reweighted by a robust loss at its current value, the route
factor-graph libraries take. It also damps Fisher scoring. `LevenbergMarquardt` refuses the Fisher
assembly (`NotAnIntegral`), since its rows are not the objective's curvature, yet its gain ratio
needs only the objective and a model with the objective's gradient, which the Fisher rows have;
without it a model with a free scale, the calibration case the residual assembly refuses
(`ScaleNotFixed`), can be scored only undamped. The work comes before Hamiltonian Monte Carlo, whose
example needs a mode and a Laplace approximation to start from.
