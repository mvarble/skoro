---
name: wf-writeup
description: Draft or revise the writeup of the active decision entry, the argument for a position skoro takes, in skoro's development workflow. Use only when the user invokes /wf-writeup.
disable-model-invocation: true
argument-hint: '[what to write, answer or change]'
---

# /wf-writeup: a decision

A decision is a position the library takes (what it is for, how its core is shaped, what it will
not do), argued in prose until the user and the agent agree on it. It is then published into the
pages that follow from it.

## Before anything

Run `bin/wf status --expect writeup`. If it fails, stop and tell the user what it said.

Read `.agents/reference/workflow.md` and `.agents/reference/authoring.md` once per session, the
entry's `index.md`, and the current draft (`bin/wf path`), if any.

## Working with the user

A decision is reached by asking. When the user asks for a decision to be worked out, interrogate
before writing: what they want, what they built before (the `archive` branch is the record), and
why. Offer alternatives where the user's choice looks weak, where a library already does this, or
where the goals and the design disagree, and say plainly which you would choose. Record the
questions still open as `> [!TODO]` callouts in the draft, and work through them with the user.

Each invocation is one revision of `site/content/archive/<slug>/writeup.md`:

```markdown
---
summary: <one sentence: the position taken>
---

## Context

What prompted the decision: the problem, the constraints, what came before.

## Decision

The position, stated plainly, then argued: what it means for the library, at the level of its
interfaces, with code only to show those.

## Consequences

What it makes easy, what it makes hard, and what it costs.

## Alternatives

What else was considered, and why it lost.
```

A long decision (the library's charter, say) may use its own sections instead, as long as it
argues rather than describes. After a reopening, keep a `## Revisions` section as in `/wf-math`.

Commit each revision. When nothing is open, reply that `/wf-approve` moves to publishing. Never
approve yourself.
