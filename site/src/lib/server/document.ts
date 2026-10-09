import { error } from '@sveltejs/kit';
import { cms } from '$cms';
import { citationLabel } from '@mvarble/mesearch-cms/presets/skoro/runtime';
import { referenceHtml } from '@mvarble/mesearch-ui/server';

import { resolve } from '$app/paths';
import { hrefOf, linkOf, type DocumentLink, type SkoroDocument } from '$lib/documents';
import { ARCHIVE, MATH, SITE, TRACKS } from '$lib/site';

export interface Crumb {
    title: string;
    href: string;
}

export interface Reference {
    key: string;
    label: string;
    html: string;
}

// Everything a document's page shows besides its own body, which `+page.ts` attaches as a
// component. The lists are links, kept small: they travel with the page as data.
export function documentData(key: string) {
    const doc = cms.get(key);
    if (!doc) error(404, `No document ${key}.`);
    const links = (records: SkoroDocument[]) => records.map(linkOf);
    const keys = (list: string[]) =>
        list.flatMap((k) => {
            const record = cms.get(k);
            return record ? [linkOf(record)] : [];
        });

    let crumbs: Crumb[];
    let prev: DocumentLink | undefined;
    let next: DocumentLink | undefined;
    if (doc.kind == 'page') {
        const track = TRACKS.find((t) => t.id == doc.track)!;
        crumbs = [{ title: track.title, href: resolve('/[track=track]', { track: track.id }) }];
        const neighbours = cms.pages.neighbours(key);
        prev = neighbours.prev && linkOf(neighbours.prev);
        next = neighbours.next && linkOf(neighbours.next);
    } else if (doc.kind == 'math') {
        crumbs = [{ title: MATH.title, href: resolve('/math') }];
    } else if (doc.kind == 'entry') {
        crumbs = [{ title: ARCHIVE.title, href: resolve('/archive') }];
    } else {
        const entry = cms.get(doc.entry)!;
        crumbs = [
            { title: ARCHIVE.title, href: resolve('/archive') },
            { title: entry.title, href: hrefOf(entry.key) },
        ];
        const siblings = cms.archive.documents(doc.entry);
        const i = siblings.findIndex((sibling) => sibling.key == key);
        prev = siblings[i - 1] && linkOf(siblings[i - 1]!);
        next = siblings[i + 1] && linkOf(siblings[i + 1]!);
    }

    const references: Reference[] = cms.bibliography(key).map((citation) => ({
        key: citation.key,
        label: citationLabel(citation),
        html: referenceHtml(citation),
    }));

    return {
        doc,
        headings: cms.headings(doc.filename),
        crumbs,
        prev,
        next,
        references,
        prerequisites: links(cms.prerequisites(key)),
        dependents: links(cms.dependents(key)),
        // A page: the development behind it. An entry: what it changed, and what it revises or
        // is revised by. An entry's document: its siblings.
        entries: doc.kind == 'page' ? keys(doc.entries) : [],
        pages: doc.kind == 'entry' ? keys(doc.pages) : [],
        revises: doc.kind == 'entry' ? keys(doc.revises) : [],
        revisedBy: doc.kind == 'entry' ? keys(doc.revisedBy) : [],
        documents: doc.kind == 'entry' ? links(cms.archive.documents(key)) : [],
        source: `${SITE.repo}/blob/${SITE.branch}/site/${doc.filename}`,
    };
}

export type DocumentData = ReturnType<typeof documentData>;
