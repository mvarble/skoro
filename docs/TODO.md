# Open work

This file and its neighbours used to live in `docs/superpowers/`, split off from `docs/` by a
workflow that no longer exists; they were moved up and the directory deleted. Everything else
that was in it is done and committed — the two specs (factor bindings / Fisher roots, and
proposal-only terms) were implemented in full, and their plans and defect reports were removed
once their work landed. Recover any of it from `git show 2948f6d:docs/superpowers/...`.

What is actually left:

## 1. `retire_density_of` is not atomic on its error paths

`Graph::retire_density_of` (`crates/skoro-graph/src/graph.rs`) removes the
`stochastic_to_log_density` entry first and only then looks the density node up, so its two
error returns leave that index already emptied. `remove` calls it with `?` from inside its
removal loop, so the same error abandons a half-applied batch -- and, because the epoch is
bumped at the end of `remove`, abandons it without invalidating the compiled-sequence cache.

Both error returns require `stochastic_to_log_density` to name a node that is absent or is not
a `LogLikelihood`, which `validate_indices` exists to rule out, so no constructible call reaches
them today. That argument holds at *entry*, though, and `remove` calls this from inside its
deletion loop -- so the window to think about is the one where the batch is half-applied and the
indices are transiently incoherent by construction. Nothing reaches it there either, because
`remove` retires a density only for a stochastic node whose density lies outside the batch, but
that is the reasoning a fix would have to preserve. Found while auditing mutators for the cache in `graph-compiler.md` §1; the sibling
gaps in `remove` and `register_children` were reachable and were fixed there. Left alone here
because a fix without a failing test to drive it would be speculative.

## 2. Four `skoro-so3` `wgpu_f32` tests fail

Pre-existing as of `91d4270`, before the bindings work, and unrelated to it. They are f32 GPU
tolerance misses — for example a scaled error of 2.3e-6 against a 1e-6 bound. Needs a GPU to
reproduce; not verified recently. Separate `skoro-so3` work.

## 3. Deferred bindings — postponed, design kept

Not planned for some time. The finished design, the triggers that would reopen it, and the
decisions it would then need are in [deferred-bindings.md](deferred-bindings.md).

---

Further out, and deliberately unplanned: [roadmap.md](roadmap.md).
