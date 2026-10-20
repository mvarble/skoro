import { cms } from '$cms';

import { resolve } from '$app/paths';
import { linkOf, type DocumentLink } from '$lib/documents';
import { crateNames } from '$lib/server/reading';
import { ARCHIVE, MATH, READING, TRACKS } from '$lib/site';
import type { LayoutServerLoad } from './$types';

export type NavItem = Pick<DocumentLink, 'key' | 'title' | 'href' | 'status'>;

export interface NavSection {
    id: string;
    title: string;
    short: string;
    href: string;
    items: NavItem[];
}

// The navigation: each track's pages, the math notes, the archive's entries, and the crates'
// reading orders.
export const load: LayoutServerLoad = () => {
    const nav: NavSection[] = [
        ...TRACKS.map((track) => ({
            ...track,
            href: resolve('/[track=track]', { track: track.id }),
            items: cms.pages.list(track.id).map(linkOf),
        })),
        { ...MATH, href: resolve('/math'), items: cms.math.list().map(linkOf) },
        { ...ARCHIVE, href: resolve('/archive'), items: cms.archive.list().map(linkOf) },
        {
            ...READING,
            href: resolve('/reading-order'),
            items: crateNames().map((crate) => ({
                key: `reading-order/${crate}`,
                title: crate,
                href: resolve('/reading-order/[crate=slug]', { crate }),
            })),
        },
    ];
    return { nav };
};
