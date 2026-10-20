import { error } from '@sveltejs/kit';

import { crateNames, readingOf } from '$lib/server/reading';
import type { EntryGenerator, PageServerLoad } from './$types';

// One page for each crate in the workspace, prerendered from its sources.
export const entries: EntryGenerator = () => crateNames().map((crate) => ({ crate }));

export const load: PageServerLoad = ({ params }) => {
    const reading = readingOf(params.crate);
    if (!reading) error(404, `No crate at crates/${params.crate}.`);
    return { reading };
};
