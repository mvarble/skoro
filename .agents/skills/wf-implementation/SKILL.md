---
name: wf-implementation
description: Draft or revise the implementation companion of the active feature entry, the second stage of skoro's development workflow. Use only when the user invokes /wf-implementation.
disable-model-invocation: true
argument-hint: '[what to write, answer or change]'
---

# /wf-implementation: the implementation companion

The second stage of a feature: a document that turns the approved mathematics into a design. It
should be complete enough that someone who knows Rust, the mathematics and the neighbouring
interfaces could build it without designing anything of their own, and it ends with the tests
that will show the design works.

## Before anything

Run `bin/wf status --expect implementation`. If it fails, stop and tell the user what it said.

Read `.agents/reference/workflow.md` and `.agents/reference/authoring.md` once per session, the
entry's `index.md`, its approved `math.md`, and the current draft (`bin/wf path`), if any. The
mathematics is frozen: if the design shows it to be wrong, stop and say so, and suggest
`/wf-reopen math <reason>`. Do not design around it.

## Researching the design

Read the code the feature has to live with before proposing anything: the traits it implements,
the types it consumes and produces, the crates it would sit in, and their tests. Compare with the
`archive` branch if skoro did this before (`git show archive:crates/<crate>/src/<file>`), and say
what is kept from that design and what is not, and why. Write no code outside the document.

## Writing

Each invocation is one revision, as at the math stage. The document,
`site/content/archive/<slug>/implementation.md`:

```markdown
---
summary: <one sentence: what will exist when this is built>
depends_on: [archive/<slug>/math, <math notes>]
---

## Overview

What will exist when the feature is built, and how a user will meet it, with a short example of
the calling code. Link the sections and equations of the mathematics it implements
(`[](eq:archive/<slug>/math/<eq>)`).

## Constraints

The neighbouring interfaces and what they impose: each trait or type, pasted with its file and the
commit it was read at, and what the design must do because of it. Never `{path}` quotes here.

## Design

The types, traits and functions, sketched in Rust: signatures, fields, where each lives (crate and
module), what is public, how errors are reported, and how each piece maps to the mathematics.
Bodies only where the algorithm is the point. Where a choice was made, say what the alternative
was and why it lost.

## Tests

Every test the build must add, each with what it checks and why that shows the design works:
unit tests, numerical checks against the mathematics (a derivative against finite differences, a
density against a closed form), and the properties that must hold. Name them as they will be
named in the code.

## Open questions

`> [!TODO]` callouts, until answered.
```

`## Tests` is required: `wf approve` refuses the document without it. After a reopening, keep a
`## Revisions` section as described in `/wf-math`.

## Finishing a revision

Build the site (`cd site && pnpm build`) if links or math changed, commit the revision, and reply
with what changed and what remains open. When nothing does, `/wf-approve` moves to the plan.
Never approve yourself.
