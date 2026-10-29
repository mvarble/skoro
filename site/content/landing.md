---
title: Home
---

Skoro is a Rust library for writing a probabilistic model once, generatively, and running
different inference algorithms on the same model. A model is built forward, the way it is thought
of: each uncertain quantity, and its conditional distribution given the quantities it depends on.
The library records the structure as it is built, and an inference algorithm reads from that
structure what it needs: it runs when the model meets its assumptions, and says what is missing
when it does not.

It is built for the model that lives over time. In sequential estimation, for robots and other
autonomous systems, a model grows as measurements arrive, retires old states as they stop
mattering, is held under several hypotheses about what the measurements came from, and is
estimated on other threads while all of that happens. Skoro's quantities may live on manifolds
such as rotations and poses, and its derivatives, of first and second order, are taken in their
tangent spaces.

> [!NOTE] Status
> Skoro is being rebuilt from its foundations. Its purpose and design are settled in
> [the charter](archive/charter/), and the library returns one piece at a time, each with a
> written record of its mathematics and design in the [archive](archive/). [The core](archive/core/)
> is back: models can be built, queried and differentiated to second order, but no distribution
> ships yet. Nothing is published on crates.io.

Start with [what skoro is](start/what-is-skoro/), and who it is for, or with
[a first model](guide/first-model/).
