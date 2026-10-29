import { error } from '@sveltejs/kit';

import { CRATES } from '$lib/crates';
import { crateNames, readingOf } from '$lib/server/reading';
import type { EntryGenerator, PageServerLoad } from './$types';

// One page for each crate in the workspace: its description and its reading order, the latter
// computed from its sources when the site is built.
export const entries: EntryGenerator = () => crateNames().map((crate) => ({ crate }));

export const load: PageServerLoad = ({ params }) => {
    const reading = readingOf(params.crate);
    if (!reading) error(404, `No crate at crates/${params.crate}.`);
    const crate = CRATES.find((c) => c.name == params.crate);
    // The description in the content being built: `SITE_CONTENT` names the fixture's.
    const content = process.env.SITE_CONTENT ?? 'content';
    const description = `/${content}/reference/crates/${params.crate}/index.md`;
    return { reading, purpose: crate?.purpose, deps: crate?.deps ?? [], description };
};
