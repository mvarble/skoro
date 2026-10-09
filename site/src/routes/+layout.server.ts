import { cms } from '$cms';

import { resolve } from '$app/paths';
import { linkOf, type DocumentLink } from '$lib/documents';
import { ARCHIVE, MATH, TRACKS } from '$lib/site';
import type { LayoutServerLoad } from './$types';

export interface NavSection {
    id: string;
    title: string;
    short: string;
    href: string;
    items: DocumentLink[];
}

// The navigation: each track's pages, the math notes, and the archive's entries.
export const load: LayoutServerLoad = () => {
    const nav: NavSection[] = [
        ...TRACKS.map((track) => ({
            ...track,
            href: resolve('/[track=track]', { track: track.id }),
            items: cms.pages.list(track.id).map(linkOf),
        })),
        { ...MATH, href: resolve('/math'), items: cms.math.list().map(linkOf) },
        { ...ARCHIVE, href: resolve('/archive'), items: cms.archive.list().map(linkOf) },
    ];
    return { nav };
};
