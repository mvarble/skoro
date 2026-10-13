import { cms } from '$cms';
import { inlineHtml, layoutGraph } from '@mvarble/mesearch-ui/server';
import type { GraphLink, GraphNode, IndexEntry } from '@mvarble/mesearch-ui';

import { hrefOf } from '$lib/documents';
import { MAP_TUNING } from '$lib/map';
import { DOCUMENTS } from '$lib/site';
import katex from '../../../katex.config';

export interface DocumentMapData {
    nodes: GraphNode[];
    links: GraphLink[];
    entries: Record<string, IndexEntry>;
}

// The map of the math notes and the archive's documents, linked by what each builds on and links
// to. It is laid out here, at build time, so it is drawn before any script runs. An archive
// document is named by its entry and its stage ("The core: Mathematics"), whatever its first
// heading is.
export function documentMap(): DocumentMapData {
    const macros = katex.macros as Record<string, string>;
    const stageTitles = new Map<string, string>(DOCUMENTS.map((doc) => [doc.name, doc.title]));
    const graph = cms.graph();
    const entries: Record<string, IndexEntry> = {};
    for (const key of graph.nodes) {
        const doc = cms.get(key)!;
        const title =
            doc.kind == 'entry-document'
                ? `${cms.get(doc.entry)!.title}: ${stageTitles.get(doc.name) ?? doc.title}`
                : doc.title;
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
    return { nodes: layoutGraph(Object.values(entries), links, MAP_TUNING), links, entries };
}
