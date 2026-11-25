# skoro's build

What `/wf-build` asks of skoro besides what the skill says.

**Before anything.** Read the entry's `implementation.md` beside `plan.md`, and the `AGENTS.md`
sections on Rust. Load any skill the code calls for (for instance `skoro-diff-operation` before
writing a differentiable operation).

**Each task.** Implement exactly what the task and the implementation companion say. Add the tests
the task names, with those names, and any others the code needs. Then run `cargo fmt`,
`cargo clippy --all-targets -- -D warnings` and `cargo test` for the crates touched. A task that
cannot be done as designed (an interface does not allow it, a test shows the mathematics wrong) is
taken back to the stage it belongs to.

**At the end.** Show the user what was built, to read before they approve it. For each crate the
branch changed (the directories under `crates/` in
`git diff --name-only origin/main...HEAD -- crates/`, the base `bin/wf check` compares against),
run `bin/reading-order crates/<crate>`, which prints its modules in the order to read them and what
each uses of a module that comes later. Put each list in the reply, leaving out a crate it reports
as having nothing to order, and say that the crate's page in the reference,
`/reference/crates/<crate>/`, draws it, and that `pnpm dev` in `site/` reads it from the sources on
every load. It describes the code as it is, not a promise of the plan: nothing in it is to be
approved, and it changes no document.
