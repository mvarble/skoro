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
   only, and merges into `main` only once that entry is done or abandoned.
2. **An approved document is frozen.** Its hash is recorded when it is approved, and it may change
   again only after its stage is reopened. The plan is the exception that proves the rule: during
   `build` its boxes are ticked, and nothing else in it changes.
3. **Documents come in order.** A stage's document is not written before the stage.
4. **Areas open as the entry advances.** The math notes (`site/content/math/`) open at `math` (or
   `writeup`); the code (`crates/`, `Cargo.*`) at `build`; the pages (the rest of `site/content/`)
   at `publish`. Without an active entry, code is closed and the rest is open for housekeeping.
5. **Approval is a gate, not a formality.** `wf approve` refuses a document that is uncommitted,
   empty, or still has a `> [!TODO]`; an implementation without a `## Tests` section; a plan
   without a checklist; a build with unticked boxes or uncommitted work; a publish whose changed
   pages are not listed in the entry's `pages`.

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
ready to merge. A done entry is never edited again. When skoro later changes what an entry
built, a new entry says so with `--revises <slug>`; the site then marks the old entry as revised,
and the pages show both under "How this came to be".

An entry set aside with `/wf-abandon` keeps its documents, approved or not, and can be taken up
again with `/wf-resume`.

## Revisions within a stage

A stage is a conversation. Each time the user asks a question or a change, the document is
revised, and each revision is committed: the archive keeps the final draft, and git keeps the
back-and-forth. The archive itself does not record the drafts, only the final text and any
reopenings.

## Before pushing

```sh
bin/wf check                      # the rules above, against origin/main
node --test 'tools/wf/*.test.ts'  # when the tooling changed
cd site && pnpm check && pnpm lint && pnpm test && pnpm build
cargo fmt --check && cargo clippy --all-targets -- -D warnings && cargo test   # once crates exist
```
