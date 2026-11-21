---
title: "Least squares, motivated from MAP estimation and trust regions"
pages:
    - concepts/building
    - concepts/linearization
    - concepts/operations
    - concepts/queries
    - concepts/rotations-and-motions
    - concepts/spaces
    - concepts/terms
    - concepts/the-normal
    - concepts/the-uniform
    - guide/first-model
    - guide/least-squares
    - guide/pose-graph
    - guide/rotations-and-poses
    - guide/sliding-window
    - reference/crates
    - reference/how-skoro-is-developed
    - start/reading-these-docs
    - start/what-is-skoro
crates: []
revises: [mathematics-in-order]
---

This entry reordered the mathematics' part on least squares and Gaussian approximations so that it begins with the estimate sought and ends with the linear algebra that computes it.
Five notes are new: [maximum a posteriori estimation](../../math/map-estimation/), [Newton's method and trust regions](../../math/trust-regions/), [models of normals as least squares](../../math/nonlinear-least-squares/), [the normal equations](../../math/normal-equations/), and a table that pairs the optimizer's names with the estimator's, [optimization and estimation, term by term](../../math/optimization-and-estimation/), which closes the part.
Two more were split off the improper Gaussian factor: [what elimination leaves](../../math/integrating-out/) and [the Kalman filter](../../math/kalman-filter/).
[The linearization](../../math/linearization/) gained its reading as a model in a trust region, [the improper Gaussian factor](../../math/improper-gaussian/) its reading of the linearization as a Gaussian on the increment, and [elimination in square-root form](../../math/square-root-elimination/) moved from first to seventh, opened by the normal equations it solves without forming.
Every equation, statement and section the archive cites stayed in its note, so no done entry was amended.
Every other page listed, with the landing page and the math notes, was rewritten to one sentence to a line, as the authoring guide now asks; their text did not change.
