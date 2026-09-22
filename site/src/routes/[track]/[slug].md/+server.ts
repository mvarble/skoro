import { error } from '@sveltejs/kit';

import { findPage, PAGES } from '$lib/server/content';
import { markdownOf } from '$lib/server/markdown';
import type { EntryGenerator, RequestHandler } from './$types';

export const prerender = true;

export const entries: EntryGenerator = () => PAGES.map((p) => ({ track: p.track, slug: p.slug }));

export const GET: RequestHandler = ({ params }) => {
    const page = findPage(params.track, params.slug);
    if (!page) error(404);
    return new Response(markdownOf(page), {
        headers: { 'content-type': 'text/markdown; charset=utf-8' },
    });
};
