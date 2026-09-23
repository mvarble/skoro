# Roadmap

Work that two shipped specs deliberately put out of scope. None of what remains here is planned
or designed; this is a list of what was consciously left undone, not a backlog anyone has
committed to. The compiler entries have since been built and are described in
[skoro.md](skoro.md) §4.5; what is left of them is noted where they stood.

## From the proposal-only terms work

Proposal-only terms landed: a term can now be a summand of `q` without being one of `p`, so
`log_weight = Σ in_density − Σ in_proposal` is a real difference rather than a subset sum.
What that change stopped short of:

- **Derivatives of `q` or of the weight.** `log_density_ord1`/`ord2` stay `p`-only. Importance
  sampling needs the weight's value, not its gradient.
- **Joint proposals over several quantities at once.** A proposal-only term scores exactly one
  stochastic quantity, which follows from `in_proposal` being derived rather than stored. The
  extension is to store `in_proposal` beside `in_density`; `Term` already has the shape for it.
- **Any importance sampler, resampling scheme, or weight accumulator.** The weight is now
  expressible; nothing consumes it yet.

## From the factor-bindings and Fisher-root work

Bindings and Fisher roots landed: a term carries labelled reference edges to other nodes, and
`FISHER_ROOT` gives `information`/`information_root`/`information_blocks` what Fisher scoring
needs. What that change stopped short of:

- **A Fisher-scoring or Levenberg–Marquardt solver loop.** The information is assembled; no
  optimizer drives it. Its one non-obvious precondition is now machine-checkable:
  `coverage::require_role(&model, FISHER_ROOT)` refuses a model whose scored terms do not all bind
  a root, which is what keeps `information_root` and the gradient it is solved against covering the
  same terms. `docs/fisher-scoring.md` §7 states the condition and §8 the failure it prevents.
- **Block-sparse Jacobian storage, and sparse solvers.** What is left of this entry: cut
  compilation, `support(id)` and the compiled-program cache shipped (see
  [skoro.md](skoro.md) §4.5), and that work deliberately left these two out.
- **Lazily built bindings from a stored distribution object.** Designed since, and postponed;
  see [deferred-bindings.md](deferred-bindings.md).
- **Any change to sampling, `observe`, `stacked_tangent`, or the discrete kind.**

## Noted in passing

- **Family kernels in `information_blocks`.** Recognize a term by its bindings and compute
  `C · J` directly, never materializing the root: O(d²·F) with zero nodes. Rejected as a
  primary design — recognition by binding shape is implicit, and every consumer of a root would
  need its own kernel — but it remains a good later optimization inside `information_blocks`
  for known root operations.
- **Riemannian manifold HMC.** The assembly in `information_blocks` (`C · J` from the Jacobian
  passes) is not yet tape-differentiable. Making it so is what RMHMC would need, and it is the
  reason the root has to stay a differentiable graph node rather than a host-side solve.
