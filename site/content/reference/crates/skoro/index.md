`skoro` is the crate a user depends on. It re-exports everything `skoro-core` makes public, and
`skoro-tensor` as `skoro::tensor`, with `Device`, `Precision` and `Target` at its root, so a model,
a query and a target are all written against one crate.

It also holds the tests that may use the public interface alone, as a later crate would: the
acceptance test of the mathematics, a sphere declared as an embedded type with its chart's volume,
and an exponential sphere whose chart is not global, a density pushed through that chart,
generation and retirement, with the nodes a retirement frees and the pins that keep them, and a
normal distribution private to the core's tests. The tests of
[`skoro-normal`](../skoro-normal/) on those spheres live here too, with the crate as a development
dependency: its density, sampler and information on a curved type, Gauss–Newton with the
log-volume, its refusals, the core's retirement replacements written with its stacked
tangent, and its improper factor, retired along the bearings of two poses until it determines a
landmark, binding its roles on a retirement's factor, needing a proposal for the quantity it leaves
undetermined, and read on the sphere; and the hierarchical normal the [guide](../../../guide/first-model/) opens on, written
with `sample` and without it, generated, and queried with its range observed. One of its tests
checks that none of the others names `skoro-core` or `skoro-tensor` directly.

Its two examples run in CI and fail when a check of their mathematics fails.
[`pose_graph`](../../../guide/pose-graph/) estimates a hundred datasets of a pose graph on
$SE(3)$ together, by Levenberg–Marquardt, and checks that the Laplace marginals at the mode are
calibrated against the truth; [`sliding_window`](../../../guide/sliding-window/) filters an open path
with a window of five poses, retiring the oldest at every tick into a linearized factor, checks
that the number of nodes it holds stops growing, and reports its calibration and its gap to the
batch. They share a module of scenarios, the model tick by tick,
the marginals' statistics and the report, and with `SKORO_FIGURES` set they write the data that
`bin/figures` draws.
