// skoro's own condition for approving a publish: what an entry lists under `crates` exists and is
// what the branch changed, and every crate it changed is described for its page in the reference.
// `workflow.config.ts` hands it to the engine as `checks.publish`.
import fs from 'node:fs';
import path from 'node:path';

import type { CheckContext } from '@mvarble/mesearch-workflow';

// The crate a path belongs to, as its directory under `crates/`, if it is one.
export function crateOf(rel: string): string | undefined {
    return /^crates\/([^/]+)\//.exec(rel)?.[1];
}

export function crateExists(root: string, name: string): boolean {
    return fs.existsSync(path.join(root, 'crates', name, 'Cargo.toml'));
}

// A crate's description, which its page in the reference shows above its reading order.
export const crateDescription = (name: string) => `site/content/reference/crates/${name}/index.md`;

export function crateChecks({ root, index, changed, list }: CheckContext): string[] {
    const problems: string[] = [];
    const crates = list('crates') ?? [];
    for (const name of crates) {
        if (!crateExists(root, name)) {
            problems.push(`${index} lists crate '${name}', which does not exist.`);
        }
    }
    if (!changed) return problems;
    // The crates it changed, which the archive links to their reading orders.
    const edited = new Set(changed.map(crateOf).filter((name) => name !== undefined));
    for (const name of edited) {
        if (!crateExists(root, name)) continue;
        if (!crates.includes(name)) {
            problems.push(`crate '${name}' changed on this branch; list it under \`crates\` in ${index}.`);
        }
        if (!fs.existsSync(path.join(root, crateDescription(name)))) {
            problems.push(`crate '${name}' changed on this branch; describe it in ${crateDescription(name)}.`);
        }
    }
    for (const name of crates) {
        if (!edited.has(name)) {
            problems.push(`${index} lists crate '${name}', which this branch did not change.`);
        }
    }
    return problems;
}
