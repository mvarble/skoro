---
title: "Examples that run"
pages: []
crates: []
---

This entry sets out to show what skoro does with examples that run, where the archive's could only
describe problems it was not yet able to solve. A model is built once and then estimated and
filtered without being told its structure again, and each example demonstrates that on a problem
of its own: a pose graph on SE(3), solved for its mode by Gauss–Newton and Levenberg–Marquardt
steps, with Laplace marginals from the same linearization; and a sliding-window filter that retires
one tick after another, each tick replaced by its linearized factor. Each example is a
`cargo run --example` target that CI runs, and its guide page quotes it the way the guide quotes
the tests today, so no page can describe an example that does not run. Writing the examples also
tests the public interface: whatever an example needs that only the tests have now (the gradient
of a density query laid out by its quantities, for one) is a sign that it belongs in the library.
