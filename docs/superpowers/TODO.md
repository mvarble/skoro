# Open work

Everything else that was in this directory is done and committed. The two specs
(factor bindings / Fisher roots, and proposal-only terms) were implemented in full; their
plans and defect reports were removed once their work landed. Recover any of it from
`git show 2948f6d:docs/superpowers/...`.

What is actually left:

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
