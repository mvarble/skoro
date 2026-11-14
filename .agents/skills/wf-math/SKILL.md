---
name: wf-math
description: Draft or revise the mathematics writeup of the active feature entry, the first stage of skoro's development workflow. Use only when the user invokes /wf-math.
disable-model-invocation: true
argument-hint: '[what to write, answer or change]'
---

# /wf-math: the mathematics

The first stage of a feature: a writeup of the mathematics at play, revised with the user until
both agree on it. No code is designed or written here.

## Before anything

Run `bin/wf status --expect math`. If it fails, stop and tell the user what it said, and which
step it points to. Answering a question about the mathematics needs no skill: do that in
conversation instead, without writing the document.

Read `.agents/reference/workflow.md` and `.agents/reference/authoring.md` once per session, the
entry's `index.md`, and the current draft (`bin/wf path`), if there is one.

## Writing

Each invocation is one revision. Read what the user asked, then change the document to answer it,
whether a question, an objection, a new direction, or "start". The answer to a question goes
into the document, where the next reader will need it, and is summarized in your reply.

Search before writing: the math notes (`site/content/math/`) for ideas the site already explains,
and the `archive` branch for how skoro did this before, if it did (`git grep -i <term> archive`,
`git show archive:docs/<file>`). Prior art informs the writeup; it does not bind it.

The document, `site/content/archive/<slug>/math.md` (`.svx` if it shows statements):

```markdown
---
summary: <one sentence: what is being worked out>
depends_on: [<math notes it needs>]
---

## The problem

What the feature computes or decides, in words, and why skoro wants it.

## Setting

The objects and spaces involved, and the notation, linking math notes rather than redefining.

## <The mathematics, in as many sections as it takes>

Derivations, with the equations the implementation will need numbered (`@tag`), and statements
for what is claimed and proved. Say where an approximation is made and what it costs.

## In skoro's terms

How the mathematics meets skoro's own objects (quantities, the compilation and probabilistic
graphs, manifolds, derivatives), still as mathematics: which quantities are random, which are
parameters, what is differentiated with respect to what. No types or traits; that is the implementation's job.

## Open questions

Each as a `> [!TODO]` callout, until it is answered.
```

An idea that stands on its own and is needed by more than this entry becomes a math note in
`site/content/math/<slug>/index.md` (the math notes are open at this stage), and the writeup
links to it.

After a reopening, keep a `## Revisions` section at the end: for each reopening, the date, what
changed and why, in a short paragraph. The earlier text is not kept there; git has it.

## Finishing a revision

1. `cd site && pnpm build` if the math is new or changed, and fix any `katex:` warning or broken
   link it reports.
2. Commit the revision, with a message saying what changed ("Derive the whitened residual's
   Jacobian").
3. Reply with what changed, what remains open, and, when nothing does, that `/wf-approve` moves
   to the implementation. Never approve, and never start the implementation, yourself.
