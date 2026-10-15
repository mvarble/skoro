# skoro

skoro is a probabilistic modeling library in Rust. This repository started over on 2026-10-09:
everything before then is on the `archive` branch (tagged `archive-2026-10-09`), and comes back
one piece at a time, each with a written record of the mathematics and the design behind it.

## How to work here

**Conversation is the default.** Answer questions, explain the code, explore the `archive`
branch, sketch ideas in the reply: none of that needs a workflow, and none of it should start one.

**The development workflow runs only when the user invokes it**, through the `/wf-*` commands
(`.agents/skills/wf-*`). Every change to the library's code goes through an entry of the
workflow: its mathematics, its implementation companion, its plan, its build, and its pages, each
approved by the user. The procedure is in `.agents/reference/workflow.md`. In short:

| command                                         | what it does                                           |
| ----------------------------------------------- | ------------------------------------------------------ |
| `/wf-start <slug> <feature\|decision\|chore>`   | open an entry on this branch                           |
| `/wf-math`, `/wf-implementation`, `/wf-plan`    | draft and revise a feature's documents, one at a time  |
| `/wf-writeup`                                   | draft and revise a decision's argument                 |
| `/wf-build`                                     | build the approved plan, task by task                  |
| `/wf-publish`                                   | bring the pages up to date and link the entry to them  |
| `/wf-approve`, `/wf-reopen`, `/wf-abandon`, `/wf-resume` | move the entry; the user's alone to give      |
| `/wf-status`                                    | where things stand                                     |

Rules that hold whether or not a command was invoked:

- Never start, approve, reopen, abandon or resume an entry on your own judgment, and never run
  `bin/wf start|approve|reopen|abandon|resume` except inside the skill the user invoked for it.
- Never do a stage's work outside its skill. Asked to write code with no entry at the build stage,
  say so and point to the command that would get there.
- `bin/wf guard` runs before every edit in Claude Code (`.claude/settings.json`) and refuses one
  the stage does not allow. Do not work around a refusal; it is telling you which command the
  user would need to give.
- When work at one stage shows an earlier stage wrong, stop and say so; the user may reopen it.
- Not gated: the tooling (`bin/`, `tools/`), the skills, CI, and the site's own code
  (`site/src/`). Changes to those are ordinary work.

## Layout

| path                          | what it is                                                                    |
| ----------------------------- | ----------------------------------------------------------------------------- |
| `crates/`                     | the library (returns with the `core` work)                                    |
| `site/`                       | the documentation site; `site/README.md` says how it works                    |
| `site/content/`               | pages, math notes, and the archive of entries                                 |
| `bin/wf`, `tools/wf/`         | the workflow's command line and its tests                                     |
| `.workflow/active`            | the entry this branch works on, written by `bin/wf` alone                     |
| `.agents/skills/`             | the skills (linked into `.claude/skills/` for Claude Code)                    |
| `.agents/reference/`          | `workflow.md` (the procedure) and `authoring.md` (how documents are written)  |

## Writing documents

Everything in `site/content/` is written as `.agents/reference/authoring.md` says. The short of
it: pages say what is and quote real code; math notes say what is true, one idea to a note;
archive documents discuss what may be, sketch code inline, and are frozen once approved.

## The archive branch

`archive` is skoro as it stood before the restart: read it as prior art, with
`git show archive:<path>`, `git log archive -- <path>` and `git grep <term> archive`. A design
there informs a new entry; it does not bind it. Never merge from it.

## Commits

Commit often: each revision of a document, each task of a plan. The first line says what changed,
as an imperative sentence ("Derive the Jacobian of the whitened residual"); the body, when there
is one, says why. Push when your environment expects it.

## Checks

Before pushing, run what applies:

```sh
bin/wf check
cargo fmt --all --check && cargo clippy --workspace --all-targets -- -D warnings
cargo test --workspace
node --test 'tools/wf/*.test.ts' && site/node_modules/.bin/tsc -p tools
cd site && pnpm check && pnpm lint && pnpm test && pnpm build
SITE_CONTENT=test/content pnpm build      # in site/: the fixture, with a document of every kind
```

CI (`.github/workflows/ci.yml`) runs the same, and `bin/wf check --merge` on pull requests into
`main`.
