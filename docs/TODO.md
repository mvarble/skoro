# Open work

This file and its neighbours used to live in `docs/superpowers/`, split off from `docs/` by a
workflow that no longer exists; they were moved up and the directory deleted. Everything else
that was in it is done and committed — the two specs (factor bindings / Fisher roots, and
proposal-only terms) were implemented in full, and their plans and defect reports were removed
once their work landed. Recover any of it from `git show 2948f6d:docs/superpowers/...`.

What is actually left:

## 0. The graph compiler — designed, in progress

Caching, cut compilation, and three small step improvements.
[graph-compiler.md](graph-compiler.md) holds the design, the execution order, and the
verification. It is the only item here with work scheduled against it.

## 1. `ModelServer::add_factor` cannot check model identity

`ModelClient::add_factor` (`crates/skoro-model/src/model.rs:886`) refuses a `Factor` built
against another model, via `check_factor_model`. The server method
(`crates/skoro-model/src/model.rs:200`) cannot: `ModelServer` has no `server_id` field, so it
binds a foreign factor's ids without complaint. Its docs point at the client method instead.

Storing the id on `ModelServer` would let the server check too. Small, unscheduled.

## 2. Four `skoro-so3` `wgpu_f32` tests fail

Pre-existing as of `91d4270`, before the bindings work, and unrelated to it. They are f32 GPU
tolerance misses — for example a scaled error of 2.3e-6 against a 1e-6 bound. Needs a GPU to
reproduce; not verified recently. Separate `skoro-so3` work.

## 3. Deferred bindings — postponed, design kept

Not planned for some time. The finished design, the triggers that would reopen it, and the
decisions it would then need are in [deferred-bindings.md](deferred-bindings.md).

---

Further out, and deliberately unplanned: [roadmap.md](roadmap.md).
