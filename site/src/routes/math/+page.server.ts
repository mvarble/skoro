import { cms } from '$cms';
import { inlineHtml, layoutGraph } from '@mvarble/mesearch-ui/server';
import type { IndexEntry } from '@mvarble/mesearch-ui';

import { hrefOf, linkOf } from '$lib/documents';
import katex from '../../../katex.config';
import type { PageServerLoad } from './$types';

// The math notes, and the map they make with the archive's documents that build on them. The map
// is laid out here, at build time, so it is drawn before any script runs.
export const load: PageServerLoad = () => {
    const macros = katex.macros as Record<string, string>;
    const graph = cms.graph();
    const entries: Record<string, IndexEntry> = {};
    for (const key of graph.nodes) {
        const doc = cms.get(key)!;
        const title =
            doc.kind == 'entry-document' ? `${cms.get(doc.entry)!.title}: ${doc.title}` : doc.title;
        entries[key] = {
            key,
            kind: doc.kind,
            url: hrefOf(key),
            title,
            titleHtml: inlineHtml(title, macros),
            summary: doc.summary,
            summaryHtml: inlineHtml(doc.summary, macros),
            created: doc.created,
            updated: doc.updated,
            readingMinutes: doc.readingMinutes,
        };
    }
    const links = graph.edges.map(({ from, to, style }) => ({ from, to, style }));
    return {
        notes: cms.math.list().map(linkOf),
        map: { nodes: layoutGraph(Object.values(entries), links), links, entries },
    };
};
