`skoro-linearize` reads a model as least squares, through the core's public interface, the roles
[the normal](../../../concepts/the-normal/#the-roles) binds, and the problems of
[`skoro-elimination`](../skoro-elimination/). A `Linearizer` is built once for a snapshot, a density
query, a target and an `Assembly`, the residual linearization of Gauss–Newton or the Fisher
linearization of a free scale. It refuses by structure, naming the term or the quantity, what
cannot be linearized; reads the factor structure off the terms' variables; and compiles one program
of every node the factors bind, whose tangent pass is seeded by a colouring of the factor structure,
so that its frames are as wide as the colours and not as the model. At each batch of points it
gives a `Linearized`, one problem per row, with the curved types' log-volumes folded into the
right-hand sides and every other log-volume in the linear term, never the query's gradient less
what the rows account for.

A `Linearized` eliminated along a schedule is a `Solved`: its steps, as `Points` a layout retracts,
its marginals over named quantities with the curved ones among them named, and, of the residual
linearization in the chart, the Laplace evidence and draws, which the Fisher linearization refuses.
A `Replacing` linearizes a retirement's retired terms alone and gives a `LinearizedReplacement`, the
[improper factor](../../../concepts/the-normal/#an-improper-factor) of their integral over the
retired quantities, which it builds with its roles bound in the retirement's own scope: the Kalman
filter where the terms are linear and Gaussian.

`LevenbergMarquardt` iterates the steps to the mode of every row of a batch, each with its own
damping by Marquardt's diagonal and Nielsen's schedule, its own acceptance and its own stop on the
decrease a step promises, sharing only the passes; `decrement`, the undamped promise, checks where
it stopped, and `Linearizer::objective` evaluates the density at a trial point. The
[concepts page](../../../concepts/linearization/) shows each piece.
