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

// The navigation: the three tracks a reader reads (the guide, the concepts and the math notes),
// then the reference, with each crate's page under its page of the crates, and the archive's
// entries. The tracks of pages keep their order in `TRACKS`; the math notes go after the last
// track that is read, before the reference, which is looked up.
export const load: LayoutServerLoad = () => {
    const tracks: NavSection[] = TRACKS.map((track) => ({
        ...track,
        href: resolve('/[track=track]', { track: track.id }),
        items: withCrates(cms.pages.list(track.id).map(linkOf)),
    }));
    const math: NavSection = {
        ...MATH,
        href: resolve('/math'),
        items: cms.math.list().map(linkOf),
    };
    const reference = tracks.findIndex((track) => track.id == 'reference');
    if (reference < 0) tracks.push(math);
    else tracks.splice(reference, 0, math);
    const nav: NavSection[] = [
        ...tracks,
        { ...ARCHIVE, href: resolve('/archive'), items: cms.archive.list().map(linkOf) },
    ];
    // Where the site is hosted, when the build knows it (`SITE_URL=https://example.org`): a link
    // preview, such as LinkedIn's, takes only an absolute URL for its image.
    const origin = (process.env.SITE_URL ?? '').replace(/\/$/, '');
    return { nav, origin };
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
