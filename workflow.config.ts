import type { WorkflowConfig } from '@mvarble/mesearch-workflow';

import { crateChecks } from './tools/workflow/crates.ts';

// skoro's development workflow: the kinds of entry, their stages, and what counts as the library's
// code. The engine (`bin/wf`) and the skills read it, and `mesearch-wf sync` writes
// `.agents/reference/mesearch/workflow.md` from it. It loads before anything is installed, so it
// imports nothing from a package but types.
export default {
    project: 'skoro',
    code: [/^crates\//, /^Cargo\.(toml|lock)$/, /^rust-toolchain(\.toml)?$/],
    describe: { code: '`crates/`, `Cargo.*`' },
    kinds: {
        feature: {
            stages: ['math', 'implementation', 'plan', 'build', 'publish'],
            describe:
                'an algorithm, a distribution, an example: the mathematics worked out first, then how it is built',
        },
        decision: {
            stages: ['writeup', 'publish'],
            describe: 'a position the library takes, argued in prose',
        },
        chore: {
            stages: ['plan', 'build', 'publish'],
            describe: 'a port or a refactor: no new mathematics, no new design',
        },
    },
    stages: {
        math: {
            document: true,
            unlocks: ['math'],
            describe: '`math.md`: the mathematics, revised until the user and the agent agree on it',
        },
        implementation: {
            document: true,
            sections: ['Tests'],
            describe:
                '`implementation.md`: the design, sketched in code, constrained by the neighbouring interfaces, ending in the tests',
        },
        plan: {
            document: true,
            checklist: true,
            editableDuring: ['build'],
            describe: '`plan.md`: the implementation distilled into a checklist',
        },
        build: { unlocks: ['code'], describe: "the code, committed as the plan's boxes are ticked" },
        publish: {
            unlocks: ['pages'],
            describe: "the pages that say what now _is_, and the entry's `index.md`, linked to them",
        },
        writeup: {
            document: true,
            unlocks: ['math'],
            describe: "`writeup.md`: a decision's argument",
        },
    },
    entryFields: { crates: '[]' },
    pagesOptional: ['chore'],
    checks: { publish: crateChecks },
} satisfies WorkflowConfig;
