# Specification: proposal-only terms

## 0. The goal

A Skoro graph encodes two densities over the same nodes. The **posterior** `p(x|z)` is known
up to a constant. The **generative model** `q(x|z)` is what the graph can sample from. The
**log-weight** `p − q` is the discrepancy an importance sampler corrects for.

Today the graph can only express `q ⊆ p`: every `LogLikelihoodNode` is a summand of `p`, and
a term is *additionally* a summand of `q` exactly when the quantity it scores is still
`Node::Stochastic`. `TermKind::{Proposal, Weight}` is that two-way partition of `p`, and
`log_weight` is a subset sum of `p` — equal to `p − q` only because `q ⊆ p`.

This change adds the missing third case: a **proposal-only** term, a summand of `q` that is
not a summand of `p`. It is the sampling mechanism for a quantity the posterior scores some
other way (or not at all), which is what lets a Monte-Carlo algorithm work over a
distribution whose density is known but which cannot be sampled from directly.

The bit cannot be derived. A joint term and a proposal-only term both have
`stochastic_id = Some(live Stochastic)`, so they are indistinguishable from stored state.
One bit per term is added.

## 1. Current state

- `LogLikelihoodNode` (`crates/skoro-graph/src/node.rs:135`) has
  `stochastic_id: Option<Id>`: `Some(x)` means "this term is the log-density of `x`", `None`
  a free-standing factor.
- `Graph::log_likelihoods: Vec<Id>` is the ordered arena of every term; marking order is
  term order. `Graph::stochastic_to_log_density` is its exact inverse over `stochastic_id`,
  hand-maintained and checked by `validate_indices`.
- `Graph::terms()` (`graph.rs:1449`) derives `TermKind` per call from the node kind of
  `scored`; nothing is cached, so `observe`/`unobserve` move a term between kinds with no
  index write.
- `log_density`, `log_density_ord1`, `log_density_ord2` each compile from, and sum over,
  `&self.log_likelihoods` unconditionally.
- `ModelServer::cleanup` (`crates/skoro-model/src/model.rs:322`) seeds its GC roots from
  `graph.log_likelihoods()`.
- `information_blocks` (`crates/skoro-model/src/information.rs:38`) and `residual_layout`
  (`crates/skoro-model/src/least_squares.rs:17`) iterate `client.factors()` and select only
  on the presence of a `FISHER_ROOT` / `RESIDUAL` binding.

## 2. Definitions

- **Term**: a `Node::LogLikelihood` in `Graph::log_likelihoods`. Unchanged.
- **In the density** (`in_density`): a stored per-term bit, `LogLikelihoodNode::in_log_density`.
  `p = Σ { term : in_density }`.
- **In the proposal** (`in_proposal`): derived, not stored — the term scores a quantity that
  is still `Node::Stochastic`. `q = Σ { term : in_proposal }`.
- **Log-weight**: `w = p − q`. No longer a subset sum.

The four cells, all reachable:

| `in_density` | `in_proposal` | meaning |
|---|---|---|
| true | true | joint: the density of a live stochastic quantity that `p` scores |
| true | false | likelihood: an observed quantity's density, or a free-standing factor |
| false | true | proposal-only: generates a quantity `p` does not score here |
| false | false | a proposal-only quantity that has since been observed — a fixed value is not sampled, so it is in neither |

## 3. Requirements

### Graph layer

**P1.** `LogLikelihoodNode` gains `pub in_log_density: bool`. `mark_log_likelihood` and
`mark_log_density` set it `true`.

**P2.** `Graph::mark_log_proposal(&mut self, id: Id, stochastic_id: Id) -> SkoroResult<()>`
behaves exactly as `mark_log_density` — same guards, same errors, same node reinsertion on
every error path — but sets `in_log_density: false`. Both are thin wrappers over one private
`mark_log_density_impl(id, stochastic_id, in_log_density)`. Promoting an existing bare
likelihood through `mark_log_proposal` sets its bit to `false`.

**P3.** `Graph::in_log_density(&self, id: Id) -> bool` reads the bit off the node, returning
`false` for any id that is not a term. `Graph::density_term_ids(&self) -> Vec<Id>` is
`log_likelihoods` filtered to that bit, in term order.

**P4.** `TermKind` is removed. `Term` becomes
`{ density: Id, scored: Option<Id>, in_density: bool, in_proposal: bool }`, with `in_density`
read off the node and `in_proposal` derived as before.

**P5.** `log_density`, `log_density_ord1` and `log_density_ord2` use `density_term_ids()` in
place of `&self.log_likelihoods` — for the `compile_backward_from` target list *and* for the
summation loop and cotangent seeds. An empty result yields zeros of shape `[batch, 1]`,
batch taken from `points` as before.

**P6.** `log_likelihoods` remains the arena of *every* term, proposal-only ones included, so
`cleanup`'s GC roots, `remove`, `unmark_log_likelihood` and `retire_density_of` are correct
unchanged. `is_scoring(id)` keeps its meaning: "`id` is a term of this graph", of `p`, of `q`,
or of both.

**P7.** `log_density_terms(points, terms)` keeps its name, its
`log_likelihoods.contains` guard and its `NotLogLikelihood` error — it sums the terms it is
given, whichever set they are, and `log_proposal` calls it with ids that are deliberately not
in `p`. This supersedes only the *doc* of R11 of
`2026-09-01-factor-bindings-and-fisher-root-spec.md`, not its signature or behaviour.

**P8.** `validate_indices` gains one clause: a term with `in_log_density == false` must have
`stochastic_id.is_some()`. A proposal-only free-standing factor would be a summand of `q`
that scores nothing, and nothing would sample it. No `SkoroError` variant is added.

**P9.** Sampling is untouched. `generate_from` compiles over all nodes with
`use_model_samplers: true` and is driven by `StochasticNode::sampler`; a proposal-only
quantity generates because it is `Stochastic`.

### Model layer

**P10.** `ModelServer::mark_proposal(&mut self, density: &Q<B, Real>, stochastic: &Q<B, T>)`
delegates to `Graph::mark_log_proposal`.

**P11.** `ModelClient::add_quantity` is factored into a private
`add_quantity_with(distr, in_log_density: bool)`; `add_quantity` and the new `add_proposal`
are one-line wrappers. The body — including the comment explaining why the write lock must be
dropped before `distr.factor(&x)` — exists exactly once.

**P12.** `ModelBuilderOps` gains
`fn add_proposal<K, D>(&self, distr: D) -> SkoroResult<Q<B, D::Qty>>`, on the builder tier
beside `add_quantity`.

**P13.** `FactorInfo` swaps `kind: TermKind` for `in_density: bool` / `in_proposal: bool`.

**P14.** `log_proposal` sums the `in_proposal` terms; `log_weight` becomes
`Σ in_density − Σ in_proposal`. Both take the read lock exactly once, closing an existing
race in which `terms()` and the evaluation locked separately.

**P15.** `information_blocks` and `residual_layout` skip terms with `!in_density`. A Gaussian
proposal binds `FISHER_ROOT` and `RESIDUAL` like any other, so without this every information
matrix in a model using `add_proposal` is silently wrong.

**P16.** `crates/skoro/src/lib.rs` drops `TermKind` from the prelude re-export.

## 4. Acceptance tests

### 4.1 `crates/skoro-graph/tests/graph_mutation.rs`

| Test | Assert |
|---|---|
| `terms_classify_density_and_proposal` (was `terms_classify_proposal_and_weight`) | stochastic `x`, observed `y`, free-standing factor: flags are `(1,1)`, `(1,0)`, `(1,0)`; after `observe(x)` the first is `(1,0)` |
| `a_proposal_only_term_is_not_in_the_log_density` | `mark_log_proposal` puts the term in `log_likelihoods()` and `terms()` with `in_proposal && !in_density`; `density_term_ids()` excludes it; `log_density` equals the value without it |
| `observing_a_proposal_only_quantity_leaves_it_in_neither` | after `observe` both flags are false; after `unobserve`, `in_proposal` is true again |
| `a_model_of_only_proposals_has_zero_log_density` | with only proposal-only terms, `log_density(points)` is zeros `[batch, 1]` |
| `marking_a_proposal_as_a_likelihood_is_refused` | `mark_log_likelihood` on a proposal-only term is `Err(MustRemainDensity)` |
| `retiring_a_proposal_only_term_restores_a_mapped_node` | `demote_to_parameter` on a proposal-only quantity retires its term, and `validate_indices` passes |

### 4.2 `crates/skoro-model/tests/terms.rs`

| Test | Assert |
|---|---|
| `proposal_plus_weight_is_the_log_density` | unchanged; now holds by construction rather than by partition |
| `observing_moves_a_term_out_of_the_proposal` | flags before/after `observe` |
| `log_weight_subtracts_a_proposal_only_term` | one likelihood `L`, one proposal-only `P`: `log_density == L`, `log_proposal == P`, `log_weight == L − P` within `1e-12` |

### 4.3 `crates/skoro-model/tests/proposal.rs` (new)

| Test | Assert |
|---|---|
| `the_weight_corrects_a_proposal_to_the_target` | target expressed as an `add_factor` term over a quantity introduced by `add_proposal`; `log_weight` equals `log_target − log_q` computed by hand within `1e-12` |
| `a_proposal_only_quantity_still_generates` | `generate(n)` returns a value for it |

### 4.4 `crates/skoro-normal/tests/information.rs`

| Test | Assert |
|---|---|
| `a_proposal_contributes_no_information` | adding an unrelated `add_proposal(Normal::new(..))` to a passing fixture leaves `information(free)` and `information_root` unchanged. Must be seen to fail before P15 lands |

## 5. Out of scope

- Derivatives of `q` or of the weight: `log_density_ord1`/`ord2` remain `p`-only. Importance
  sampling needs the weight's value, not its gradient.
- Joint proposals over several quantities at once. A proposal-only term scores exactly one
  stochastic quantity, which follows from `in_proposal` being derived. The extension is to
  store `in_proposal` beside `in_density`; `Term` already has the shape for it.
- Fusing `log_weight` into a single compiled pass. Two passes recompute the joint terms,
  which cancel to within rounding.
- Any importance sampler, resampling scheme, or weight accumulator.
