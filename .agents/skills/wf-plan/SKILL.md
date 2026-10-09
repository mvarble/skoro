---
name: wf-plan
description: Write or revise the plan, a checklist distilled from the approved implementation companion (or, for a chore, from the request), in skoro's development workflow. Use only when the user invokes /wf-plan.
disable-model-invocation: true
argument-hint: '[what to change]'
---

# /wf-plan: the plan

The implementation companion's prose, distilled into a checklist that the build works through in
order. It should need few revisions: everything in it is already decided. A chore has no
implementation companion; its plan is distilled from what the user asked and from the code.

## Before anything

Run `bin/wf status --expect plan`. If it fails, stop and tell the user what it said.

Read `.agents/reference/workflow.md` once per session, the entry's `index.md`, and its approved
`implementation.md` (for a feature). If writing the plan shows a hole in the design, stop and
suggest `/wf-reopen implementation <reason>` rather than deciding it here.

## Writing

`site/content/archive/<slug>/plan.md`:

```markdown
---
summary: <one sentence>
depends_on: [archive/<slug>/implementation]
---

## <Group, such as a crate or a layer>

- [ ] <One task, small enough for one commit.> Tests: `<test_name>`, `<test_name>`.
- [ ] ...

## Done when

<The checks that must pass at the end: cargo test, clippy, the site build, and anything the
implementation's tests section singles out.>
```

- Every test named in the implementation's `## Tests` appears in exactly one task.
- Order tasks so that each builds on finished ones, and the code compiles and passes its tests
  after each.
- A task says what to do, not how: the how is in the implementation companion.

Commit each revision. When the plan is right, reply that `/wf-approve` starts the build. Never
approve yourself.
