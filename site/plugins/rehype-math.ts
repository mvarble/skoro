import type { Root } from 'hast';
import type { KatexOptions } from 'katex';
import type { Plugin } from 'unified';
import type { VFile } from 'vfile';

import { renderMath } from './katex.ts';

// Replaces the blog's `rehypeCms`, which folded macros from a database. Here there are only two
// levels: the project-wide macros in `katex.config.ts` (passed as `options`), then the page's own
// `katex_macros` frontmatter, which wins on a clash. `renderMath` copies the merged object, so a
// `\gdef` in one page cannot leak into the next.
export const rehypeMath: Plugin<[KatexOptions?]> = (options) => {
    return (tree, vfile) => {
        // mdsvex sets `filename` rather than vfile's own `path`.
        const file = vfile as VFile & { filename?: string; data: { fm?: Record<string, unknown> } };
        const name = file.filename ?? file.path ?? '<unknown>';
        const local = file.data.fm?.katex_macros;
        if (local !== undefined && (typeof local != 'object' || local === null)) {
            throw new Error(`${name}: \`katex_macros\` must be a map of macro to expansion.`);
        }
        renderMath(tree as Root, name, {
            ...options,
            macros: {
                ...(options?.macros ?? {}),
                ...((local as Record<string, string>) ?? {}),
            },
        });
    };
};
