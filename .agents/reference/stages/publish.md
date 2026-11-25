# skoro's publish

What `/wf-publish` asks of skoro besides what the skill says.

The pages that say what skoro _is_ are the guide, top-down, for someone using the library; the
concepts, bottom-up, for someone who wants to know how it works; and the reference. Read the
affected ones under `site/content/{start,guide,concepts,reference}/` before changing them.

- **Concepts (bottom-up).** Add or rewrite the pages that build up to what the entry added, each
  resting only on pages before it in the track (`order`). Link the math notes for the mathematics.
- **Guide (top-down).** Show what a user writes, with an example that works against the code as it
  is, and link down into the concepts for the curious.
- **Reference.** The crate map (`site/src/lib/crates.ts`), glossary, notation and status, where
  they changed. Every crate the entry changed has a page at `/reference/crates/<crate>/`, which
  shows its reading order, computed from its sources, under a description of what the crate is
  for: write or revise that description, from a couple of sentences to a couple of paragraphs, in
  `site/content/reference/crates/<crate>/index.md`. It is Markdown, with links written the way the
  files sit on disk; add a crate the entry created to the crate map as well.
- **The entry's `index.md`** lists every crate it changed (by its directory under `crates/`) under
  `crates`, beside its `pages`; the archive links them to the crates' pages. `wf approve` refuses
  a changed crate that is not listed, a listed one the branch did not change, and a changed crate
  with no description.
- **What it amends.** If the entry rewrote a math note, the archive's references into that note
  must still be true (`authoring.md`, under "The math notes"): where one had to change, change it
  in the done entry's document or statement, and list it under `amends`.
