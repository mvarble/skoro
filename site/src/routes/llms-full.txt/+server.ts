import { PAGES } from '$lib/server/content';
import { markdownOf } from '$lib/server/markdown';
import { SITE } from '$lib/site';
import type { RequestHandler } from './$types';

export const prerender = true;

export const GET: RequestHandler = () => {
    const header = `# ${SITE.name}\n\n> ${SITE.description}\n`;
    const body = PAGES.map((p) => `\n---\n\n<!-- page: ${p.path} -->\n\n${markdownOf(p)}`);
    return new Response(header + body.join(''), {
        headers: { 'content-type': 'text/plain; charset=utf-8' },
    });
};
