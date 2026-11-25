// What skoro's site adds to the pages `@mvarble/mesearch-workflow` shares: an entry's crates
// under "What it changed", and its walkthroughs in the Markdown export.
import type { MarkdownContext, SiteHooks } from '@mvarble/mesearch-workflow';
import { importedSpecs } from '@mvarble/mesearch-workflow/app/lib/server/specs.ts';
import { REPO_ROOT } from '@mvarble/mesearch-workflow/app/lib/server/repo.ts';
import { isFileRef, resolveFileRef } from '@mvarble/mesearch-markdown';

import { entryCrates } from '$lib/server/reading';
import { describeWalkthrough, staleMatches } from '$lib/walk/describe';
import type { WalkthroughSpec } from '$lib/walk/spec';

// Every `<Walkthrough>` spec, in a `*.walk.ts` file beside its page; the fixture's too.
const walkModules = import.meta.glob<Record<string, WalkthroughSpec>>(
    ['/content/**/*.walk.ts', '/test/content/**/*.walk.ts'],
    { eager: true },
);

// Replaces each `<Walkthrough spec={name}>` and its quoted code with the code and
// `describeWalkthrough` of the spec. A step whose code match finds no line of the quoted code
// fails the build: this is what keeps a walkthrough in step with the code it quotes.
function describeWalkthroughs(body: string, { doc, script, file }: MarkdownContext): string {
    const specs = importedSpecs(script, doc, walkModules);
    return body.replace(
        /<Walkthrough\s+spec=\{(\w+)\}\s*>([\s\S]*?)<\/Walkthrough>/g,
        (_, name: string, inner: string) => {
            const spec = specs.get(name);
            if (!spec) throw new Error(`${doc.filename}: \`${name}\` is no imported walkthrough.`);
            const fence = /```(\w*)\n(\{[^\n]*\})\n```/.exec(inner);
            if (!fence || !isFileRef(fence[2]!)) {
                throw new Error(`${doc.filename}: walkthrough \`${name}\` quotes no file.`);
            }
            const quoted = resolveFileRef(fence[2]!, file, REPO_ROOT, fence[1]).code;
            const stale = staleMatches(spec, quoted);
            if (stale.length) {
                throw new Error(
                    `${doc.filename}: walkthrough "${spec.title}":\n    ${stale.join('\n    ')}`,
                );
            }
            return `${inner.trim()}\n\n${describeWalkthrough(spec)}`;
        },
    );
}

export const hooks: SiteHooks = {
    entryChanges: (entry) => {
        const crates = entryCrates(entry.filename);
        if (!crates.length) return [];
        return [
            {
                intro: 'The crates it changed:',
                links: crates.map(({ crate, href }) => ({ href, label: crate, code: true })),
            },
        ];
    },
    markdown: [describeWalkthroughs],
};
