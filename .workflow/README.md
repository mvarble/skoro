# .workflow

`active` names the archive entry this branch is working on. It exists only while an entry is
open, and only `bin/wf` writes it: `wf start` and `wf resume` create it, and `wf approve` (at the
last stage) and `wf abandon` remove it. A branch merges into `main` without one.

The entries themselves, with their stage records, are in `site/content/archive/`. The workflow is
described in `.agents/reference/workflow.md`.
