---
name: wf-build
description: Build the active entry by working through its approved plan, task by task, with tests and a commit per task, in skoro's development workflow. Use only when the user invokes /wf-build.
disable-model-invocation: true
argument-hint: '[a task to start from, or a limit such as "the first three"]'
---

# /wf-build: the build

Work through `plan.md` in order: implement each task as the implementation companion describes,
add its tests, tick its box, commit.

## Before anything

Run `bin/wf status --expect build`. If it fails, stop and tell the user what it said.

Read `.agents/reference/workflow.md` once per session, the entry's `implementation.md` and
`plan.md`, and the `AGENTS.md` sections on Rust. Load any skill the code calls for (for instance
`skoro-diff-operation` before writing a differentiable operation).

## Each task

1. Implement exactly what the task and the implementation companion say. Details they leave open
   (private helpers, local names, the order of statements) are yours to decide.
2. Add the tests the task names, with those names, and any others the code needs.
3. Run `cargo fmt`, `cargo clippy --all-targets -- -D warnings` and `cargo test` for the crates
   touched. Do not move on with anything failing.
4. Tick the task's box in `plan.md`; change nothing else in the plan, since `wf check` refuses any
   other edit to it.
5. Commit, with a message saying what the task added.

Commit often, push when your environment expects it, and report progress between tasks when the
user asked for only some of them.

## When the plan is wrong

If a task cannot be done as designed (an interface does not allow it, a test shows the
mathematics wrong, the plan misses a step), stop. Tell the user what was found and which stage it
belongs to, and suggest `/wf-reopen <stage> <reason>`. Do not deviate from the design quietly, and
never edit an approved document.

## At the end

When every box is ticked: run the plan's "Done when" checks and `bin/wf check`.

Then show the user what was built, to read before they approve it. For each crate the branch
changed (the directories under `crates/` in `git diff --name-only origin/main...HEAD -- crates/`,
the base `bin/wf check` compares against), run `bin/reading-order crates/<crate>`, which prints its
modules in the order to read them and what each uses of a module that comes later. Put each list
in the reply, leaving out a crate it reports as having nothing to order, and say that `pnpm dev`
in `site/` draws it at `/reading-order/<crate>/`, read from the sources on every load. It
describes the code as it is, not a promise of the plan: nothing in it is to be approved, and it
changes no document.

Reply that `/wf-approve` moves to publishing. Never approve yourself.
