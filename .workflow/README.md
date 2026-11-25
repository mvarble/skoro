# .workflow

`active` names the archive entry this branch is working on. It exists only while an entry is
open, and only `bin/wf` writes it: `wf start` and `wf resume` create it, and `wf approve` (at the
last stage) and `wf abandon` remove it. A branch merges into `main` without one.

The entries themselves, with their stage records, are in `site/content/archive/`. The workflow is
described in `.agents/reference/mesearch/workflow.md`.

The rest is `@mvarble/mesearch-workflow`'s, which `mesearch-wf sync` writes: `engine/`, the engine
`bin/wf` runs, and `managed.json`, every file the package wrote with its hash, which `bin/wf guard`
and `bin/wf check` hold them to.
