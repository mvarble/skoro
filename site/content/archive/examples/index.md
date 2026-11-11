---
title: "Examples that run"
pages:
    - guide/pose-graph
    - guide/sliding-window
    - guide/least-squares
    - concepts/linearization
    - concepts/queries
    - concepts/rotations-and-motions
    - reference/crates
    - reference/how-skoro-is-developed
    - start/reading-these-docs
crates: [skoro, skoro-core, skoro-linearize, skoro-spatial]
---

This entry gave skoro two examples that run, in the umbrella crate, each a `cargo run --example`
target that CI runs and that fails when a check of its mathematics fails. `pose_graph` draws a
hundred datasets from a model of poses on SE(3) and the landmarks they sight, estimates them
together by Levenberg–Marquardt, and checks that the Laplace marginals at the mode contain the
truth as often as they claim to, with a tolerance fixed by the statistics and the approximation's
expected error before the example runs. `sliding_window` keeps the same model to a window of five
poses over an open path, retiring the oldest at every tick into a factor linearized at the window's
mode, and reports its calibration and its gap to the batch, which the mathematics does not bound.

Writing them put into the library what a user would need too: the Levenberg–Marquardt method in
`skoro-linearize`, with each row of a batch damped and stopped on its own, and a stop on the
decrease a step promises, which the build showed must not wait for an acceptance; the inverse step
of a layout and the observation of generated values in the core; and the rigid motions' composition
and action on the host in `skoro-spatial`. The examples write the data of their figures on request,
and `bin/figures` draws them for the README and the guide.
