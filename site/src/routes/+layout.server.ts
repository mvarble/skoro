import { resolve } from '$app/paths';
import { navigation, siteOrigin, type NavItem } from '@mvarble/mesearch-workflow/app/lib/nav.ts';

import { crateNames } from '$lib/server/reading';
import type { LayoutServerLoad } from './$types';

// The navigation: the three tracks a reader reads (the guide, the concepts and the math notes,
// which `NAV.mathBefore` places before the reference), then the reference, with each crate's page
// under its page of the crates, and the archive's entries.
export const load: LayoutServerLoad = () => ({
    nav: navigation({
        items: (track, items) => (track == 'reference' ? withCrates(items) : items),
    }),
    origin: siteOrigin(),
});

// Each crate's page, right after the page of the crates (or last, in content without one).
function withCrates(items: NavItem[]): NavItem[] {
    if (!items.some((item) => item.key.startsWith('reference/'))) return items;
    const crates: NavItem[] = crateNames().map((crate) => ({
        key: `reference/crates/${crate}`,
        title: crate,
        href: resolve('/reference/crates/[crate=slug]', { crate }),
        sub: true,
        code: true,
    }));
    const at = items.findIndex((item) => item.key == 'reference/crates');
    return at < 0
        ? [...items, ...crates]
        : [...items.slice(0, at + 1), ...crates, ...items.slice(at + 1)];
}
