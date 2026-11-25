# skoro's plan

What `/wf-plan` asks of skoro besides what the skill says.

**What it is distilled from.** A feature's plan is its approved implementation companion's prose,
distilled into a checklist: everything in it is already decided, so it should need few revisions.
Read the entry's `implementation.md`. If writing the plan shows a hole in the design, stop and
suggest `/wf-reopen implementation <reason>` rather than deciding it here. A chore has no
implementation companion; its plan is distilled from what the user asked and from the code.

**The template.**

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
- Order tasks so that the code compiles and passes its tests after each.
- A task says what to do, not how: the how is in the implementation companion.
