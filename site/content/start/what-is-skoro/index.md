---
title: What is skoro?
summary: A library for writing a probabilistic model once and running any inference algorithm on it, built for models that live over time.
order: 1
---

Skoro is a Rust library for writing probabilistic models once, generatively, and running different
inference algorithms on the same model. This page says what that means, who it is for, and the
ideas the rest of the library follows from. The argument for each of them is in
[the charter](../../archive/charter/writeup.md).

## One model, written generatively

A model is written the way it is thought of: each uncertain quantity, and the conditional
distribution it has given the quantities it depends on. Consider a range measurement $z$ between a
vehicle at $x$ and a target at $y$:

$$
	z \sim \mu(\mathrm{d}z \mid x, y)
$$

That one statement serves three situations. The vehicle and the target can be estimated jointly
from $z$; the vehicle alone can be estimated from $z$ when the target is known; and $z$ can be
simulated from given $x$ and $y$. In skoro these are not three models. The model holds the joint
distribution, and which quantities are observed is part of each question asked of it, so the same
model answers all three, from different threads at once.

Factor-graph libraries, by contrast, describe a model by the residuals of one particular posterior.
Each of the three situations needs a different set of factors, and the third, simulation, cannot be
expressed at all.

## Inference reads the model

The library records a model's structure as the model is built, much as automatic differentiation
records a computation: which quantities depend on which, what each density term scores, which of
its parts are a residual or a whitening factor, which operations can be differentiated to second
order. An inference algorithm reads what it needs from that record. If the model meets the
algorithm's assumptions, the algorithm runs; if not, it refuses before it starts and names what is
missing and where. The user never describes the model a second time for the algorithm's benefit.

> [!DECISION] No capability flags
> What a model can do is read off its structure, never declared by its user. See
> [position 6 of the charter](../../archive/charter/writeup.md#6-algorithms-refuse-by-structure).

## Models that live over time

Skoro is built for sequential estimation, where a model is never finished:

- **It grows.** Each tick appends new states and measurements, and each kind of measurement can be
  different.
- **It forgets.** States that stop mattering are marginalized onto what remains and retired.
- **It branches.** When it is uncertain which target a measurement came from, the model is held
  under several hypotheses at once, as cheap branches of one another, and an algorithm weighs them.
- **It is shared.** One writer builds while estimators run on immutable snapshots of the model,
  without waiting for each other.

Sub-models are ordinary functions of quantities, placed under namespaces by the code that calls
them, so the model of one tick is written once and reused for every tick.

## Who it is for

1. **People developing inference algorithms.** Skoro's first purpose is to make new algorithms easy
   to write against real models: what an algorithm can discover about a model matters as much as
   how a model is built.
2. **People building real-time estimators** for robots and autonomous systems, on CPU or GPU, from
   Rust, often under several hypotheses at once.
3. **People doing Bayesian inference** with quantities that live on manifolds, such as rotations
   and poses, or that need second-order derivatives.

## What it is not

- **Not a general probabilistic programming language.** Each model has a fixed structure, built
  forward. Structure that is itself random is handled one level up, over a family of models, rather
  than by programs whose shape changes as they run, as in Gen.jl.
- **Not a replacement for mature solvers.** It does not compete with GTSAM or Ceres on the maturity
  of its least-squares solvers, nor with Stan or NumPyro on the breadth of its distributions. What
  it offers is the combination: one generative model, a structure algorithms can read,
  manifold-aware derivatives of second order, and a model that lives across threads and over time.

## The ideas everything follows from

The [charter](../../archive/charter/writeup.md) argues each of these in full:

- **Conditioning belongs to the query**, never to the model.
- **Derivatives come from composition.** A few primitive operations carry hand-written derivatives,
  and everything else, distributions and manifold retractions included, is built from them; each
  operation reports the order of derivative it supports.
- **Static at the edges, dynamic inside.** Models are written with typed handles the compiler
  checks, while everything stored inside is nameable, so models can be serialized and driven from
  other languages.
- **Randomness is explicit.** Every random draw comes from a key, so one seed reproduces a run.
- **Precision belongs to the query.** A model commits to no float precision; each estimator chooses
  its own, and the solves where conditioning matters run on the host in double precision.
