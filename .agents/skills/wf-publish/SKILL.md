---
name: wf-publish
description: Bring skoro's documentation pages up to date with what the active entry built or decided, and link the entry to them; the last stage of skoro's development workflow. Use only when the user invokes /wf-publish.
disable-model-invocation: true
argument-hint: '[pages to start with, or what to change]'
---

# /wf-publish: the pages

The entry is built (or, for a decision, argued). Now the pages that say what skoro *is* must say
it: the guide, top-down, for someone using the library; the concepts, bottom-up, for someone
who wants to know how it works; and the reference. The archive discussed what might be; the pages
tell what is.

## Before anything

Run `bin/wf status --expect publish`. If it fails, stop and tell the user what it said.

Read `.agents/reference/authoring.md` once per session, the entry's documents, and the code it
added. Read the pages that are affected (`site/content/{start,guide,concepts,reference}/`, the
landing page) before changing them.

## Writing

- **Concepts (bottom-up).** Add or rewrite the pages that build up to what the entry added, each
  resting only on pages before it in the track (`order`). Explain what is, quoting the real code
  with `{path:start..end}` fences, and link the math notes for the mathematics. A design choice a
  reader would wonder about gets a short `> [!DECISION]` linking the entry that made it.
- **Guide (top-down).** Show what a user writes, with an example that works against the code as
  it is, and link down into the concepts for the curious.
- **Reference.** The crate map (`site/src/lib/crates.ts`), glossary, notation and status, where
  they changed. Every crate the entry changed has a page at `/reference/crates/<crate>/`, which
  shows its reading order, computed from its sources, under a description of what the crate is
  for: write or revise that description, from a couple of sentences to a couple of paragraphs, in
  `site/content/reference/crates/<crate>/index.md`. It is Markdown, with links written the way the
  files sit on disk; add a crate the entry created to the crate map as well.
- **Drop what is no longer true.** A page describes only the current library. What a revised
  entry built and this one replaced disappears from the pages; it stays in the archive.
- **The entry's `index.md`.** Replace its opening paragraph with what the entry did, for a reader
  of the archive, list every page it changed under `pages`, and every crate it changed (by its
  directory under `crates/`) under `crates`, which the archive links to the crates' pages.
  `wf approve` refuses a changed page or crate that is not listed, a listed one the branch did not
  change, and a changed crate with no description.

Build the site (`cd site && pnpm build`) and fix every warning and broken link. Commit as you go.

When the pages are right, run `bin/wf check` and reply that `/wf-approve` finishes the entry,
after which the branch is ready to merge. Never approve yourself.
