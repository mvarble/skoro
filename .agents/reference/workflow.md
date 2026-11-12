# The development workflow

Every change to skoro's code is an **entry** in the archive (`site/content/archive/<slug>/`),
worked on in a branch of its own and moved through **stages**. A person approves each stage;
nothing advances by itself. `bin/wf` keeps the record and enforces the rules; the `wf-*` skills
do the work of each stage.

This file is the procedure. `AGENTS.md` says when it applies (only when the user invokes one of
its commands), and `authoring.md` beside this file says how to write the documents.

## Kinds and stages

| kind       | stages                                              | for                                                     |
| ---------- | --------------------------------------------------- | ------------------------------------------------------- |
| `feature`  | `math` → `implementation` → `plan` → `build` → `publish` | an algorithm, a distribution, an example                |
| `decision` | `writeup` → `publish`                               | a position the library takes, argued in prose           |
| `chore`    | `plan` → `build` → `publish`                        | a port, a refactor: no new mathematics, no new design   |

Each stage, what it produces, and the skill that works on it:

| stage            | skill                 | produces                                                                                     |
| ---------------- | --------------------- | -------------------------------------------------------------------------------------------- |
| `math`           | `/wf-math`            | `math.md`: the mathematics, revised until the user and the agent agree on it                  |
| `implementation` | `/wf-implementation`  | `implementation.md`: the design, sketched in code, constrained by the neighbouring interfaces, ending in the tests |
| `plan`           | `/wf-plan`            | `plan.md`: the implementation distilled into a checklist                                    |
| `build`          | `/wf-build`           | the code, committed as the plan's boxes are ticked                                           |
| `publish`        | `/wf-publish`         | the pages that say what now *is*, and the entry's `index.md`, linked to them                 |
| `writeup`        | `/wf-writeup`         | `writeup.md`: a decision's argument                                                          |

And the commands that move an entry, all of them the user's to give:

| command                           | `bin/wf`                                    |
| --------------------------------- | ------------------------------------------- |
| `/wf-start <slug> <kind> [about]` | `wf start <slug> --kind <kind> --title ...`  |
| `/wf-approve`                     | `wf approve`                                |
| `/wf-reopen <stage> <reason>`     | `wf reopen <stage> --reason ...`            |
| `/wf-abandon <reason>`            | `wf abandon --reason ...`                   |
| `/wf-resume <slug>`               | `wf resume <slug>`                          |
| `/wf-status`                      | `wf status`                                 |

## What the stages guarantee

`bin/wf check` verifies these on every pull request; `bin/wf guard`, a Claude Code hook in
`.claude/settings.json`, refuses an edit that breaks one before it is made.

1. **One entry per branch.** `.workflow/active` names it. A branch changes the files of one entry
   only, and merges into `main` only once that entry is done or abandoned. What the entry
   [amends](#amending-a-done-entry) in entries that are done is the one exception.
2. **An approved document is frozen.** Its hash is recorded when it is approved, and it may change
   again only after its stage is reopened. The plan is the exception that proves the rule: during
   `build` its boxes are ticked, and nothing else in it changes. Once its entry is done, a
   document changes only by an [amendment](#amending-a-done-entry).
3. **Documents come in order.** A stage's document is not written before the stage.
4. **Areas open as the entry advances.** The math notes (`site/content/math/`) open at `math` (or
   `writeup`); the code (`crates/`, `Cargo.*`) at `build`; the pages (the rest of `site/content/`)
   at `publish`. Without an active entry, code is closed and the rest is open for housekeeping.
5. **Approval is a gate, not a formality.** `wf approve` refuses a document that is uncommitted,
   empty, or still has a `> [!TODO]`; an implementation without a `## Tests` section; a plan
   without a checklist; a build with unticked boxes or uncommitted work; a publish whose changed
   pages and crates are not listed in the entry's `pages` and `crates`, whose changed crates
   have no description in `site/content/reference/crates/<crate>/index.md`, or whose changes to
   done entries are not listed in its `amends`.

Anything outside these areas (the tooling, the skills, the site's own code) is not gated.

## Going back

A stage's work sometimes shows that an earlier stage was wrong. Do not work around it: stop and
tell the user what was found. If they agree, they ask for `/wf-reopen <stage> <reason>`, which
moves the entry back, marks every approval from that stage on as needing renewal, and records the
reason in the entry's history. The reopened document gets a `## Revisions` section saying what
changed and why; each later stage is approved again on the way forward (unchanged, if nothing in
it had to change).

The story is allowed to change. It is not allowed to forget: the history shows every reopening,
and git keeps every draft.

## A finished entry

When `publish` is approved the entry is done, `.workflow/active` is removed, and the branch is
ready to merge. A done entry's argument is never edited again. When skoro later changes what an
entry built, a new entry says so with `--revises <slug>`; the site then marks the old entry as
revised, and the pages show both under "How this came to be".

An entry set aside with `/wf-abandon` keeps its documents, approved or not, and can be taken up
again with `/wf-resume`.

## Amending a done entry

The math notes are rewritten as the mathematics grows (`authoring.md`, under "The math notes"),
and the archive's documents refer to their equations, statements and sections. A reference is
kept true: what is cited stays where the notes have a place for it, the reference follows what
moves, and what leaves the notes is written into the document that cited it. The last two change
a document of a done entry, and that is the one change a done entry admits. Its argument, what it
claimed and how it weighed things, stays as it was approved.

- Only at its `publish` stage may an entry change another's files, and only a done entry's
  documents (`math`, `implementation`, `plan`, `writeup`) and statements. `bin/wf guard` refuses
  anything else.
- The entry lists what it changed under `amends` in its `index.md`: a document as
  `archive/<slug>/<name>`, a statement as `archive/<slug>/statements/<name>`.
- `wf approve` refuses the publish if a done entry's file changed and is not listed, or is listed
  and did not change. On approval it writes the amended document's new hash into the amended
  entry's `workflow.json`, with an `amend` event naming the amending entry. The commit of the
  document's own approval stays in the record, and is where its text before the amendment is.
  Commit every `workflow.json` it names.
- Nothing mechanical limits an amendment to references: the person who approves the publish reads
  the diff. Never use an amendment to change what a document argues; that is what a new entry
  with `--revises` is for.

## Revisions within a stage

A stage is a conversation. Each time the user asks a question or a change, the document is
revised, and each revision is committed: the archive keeps the final draft, and git keeps the
back-and-forth. The archive itself does not record the drafts, only the final text and any
reopenings.

## Before pushing

```sh
bin/wf check                      # the rules above, against origin/main
node --test 'tools/*/*.test.ts' && site/node_modules/.bin/tsc -p tools   # when the tooling changed
cd site && pnpm check && pnpm lint && pnpm test && pnpm build
cargo fmt --check && cargo clippy --all-targets -- -D warnings && cargo test   # once crates exist
```
