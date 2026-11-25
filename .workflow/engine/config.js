// What a project says about its workflow, in `workflow.config.ts` at the root of its repository:
// its kinds of entry, their stages, what each stage opens for editing, and what counts as its
// code. Everything else (the stage machine, the locks, the guard, amendments, `check`) is the
// engine's, and the same in every project.
//
// The engine runs from `.workflow/engine/`, vendored into the project, before anything is
// installed, so the config must load under Node alone: it may import types from this package
// (they are erased), and nothing else from `node_modules`.
// A config is written as
//
//     import type { WorkflowConfig } from '@mvarble/mesearch-workflow';
//     export default { project: 'skoro', ... } satisfies WorkflowConfig;
//
// with a type-only import, so that it loads before the package is installed.
export const CONFIG_FILE = 'workflow.config.ts';
