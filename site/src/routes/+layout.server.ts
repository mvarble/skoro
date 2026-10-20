import { cms } from '$cms';

import { resolve } from '$app/paths';
import { linkOf, type DocumentLink } from '$lib/documents';
import { crateNames } from '$lib/server/reading';
import { ARCHIVE, MATH, TRACKS } from '$lib/site';
import type { LayoutServerLoad } from './$types';

/** A link in the navigation: a document's, or a crate's page, listed under the page of the crates. */
export type NavItem = Pick<DocumentLink, 'key' | 'title' | 'href' | 'status'> & { crate?: boolean };

export interface NavSection {
    id: string;
    title: string;
    short: string;
    href: string;
    items: NavItem[];
}

// The navigation: each track's pages, with each crate's page under the reference's page of the
// crates, the math notes, and the archive's entries.
export const load: LayoutServerLoad = () => {
    const nav: NavSection[] = [
        ...TRACKS.map((track) => ({
            ...track,
            href: resolve('/[track=track]', { track: track.id }),
            items: withCrates(cms.pages.list(track.id).map(linkOf)),
        })),
        { ...MATH, href: resolve('/math'), items: cms.math.list().map(linkOf) },
        { ...ARCHIVE, href: resolve('/archive'), items: cms.archive.list().map(linkOf) },
    ];
    return { nav };
};

// Each crate's page, right after the page of the crates (or last, in content without one).
function withCrates(items: NavItem[]): NavItem[] {
    if (!items.some((item) => item.key.startsWith('reference/'))) return items;
    const crates: NavItem[] = crateNames().map((crate) => ({
        key: `reference/crates/${crate}`,
        title: crate,
        href: resolve('/reference/crates/[crate=slug]', { crate }),
        crate: true,
    }));
    const at = items.findIndex((item) => item.key == 'reference/crates');
    return at < 0
        ? [...items, ...crates]
        : [...items.slice(0, at + 1), ...crates, ...items.slice(at + 1)];
}
