import { error } from '@sveltejs/kit';

import { neighbours } from '$lib/content';
import { findPage, PAGES } from '$lib/server/content';
import { TRACKS } from '$lib/site';
import type { EntryGenerator, PageServerLoad } from './$types';

export const entries: EntryGenerator = () => PAGES.map((p) => ({ track: p.track, slug: p.slug }));

export const load: PageServerLoad = ({ params }) => {
    const page = findPage(params.track, params.slug);
    if (!page) error(404, `No page ${params.track}/${params.slug}.`);
    return {
        page,
        track: TRACKS.find((t) => t.id == page.track)!,
        ...neighbours(PAGES, page),
    };
};
