import { error } from '@sveltejs/kit';

import { pagesIn } from '$lib/content';
import { PAGES } from '$lib/server/content';
import { TRACKS } from '$lib/site';
import type { EntryGenerator, PageServerLoad } from './$types';

export const entries: EntryGenerator = () => TRACKS.map((t) => ({ track: t.id }));

export const load: PageServerLoad = ({ params }) => {
    const track = TRACKS.find((t) => t.id == params.track);
    if (!track) error(404, `No section named ${params.track}.`);
    return { track, trackPages: pagesIn(PAGES, track.id) };
};
