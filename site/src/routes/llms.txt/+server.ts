import { resolve } from '$app/paths';
import { pagesIn } from '$lib/content';
import { PAGES } from '$lib/server/content';
import { SITE, TRACKS } from '$lib/site';
import type { RequestHandler } from './$types';

export const prerender = true;

// The index format of https://llmstxt.org: a title, a one-paragraph summary, then one section per
// track listing each page's Markdown source with its summary. Links are absolute when the build
// knows where the site is hosted (`SITE_URL=https://example.org`), and root-relative otherwise.
export const GET: RequestHandler = () => {
    const origin = (process.env.SITE_URL ?? '').replace(/\/$/, '');
    const link = (path: string) => `${origin}${path}`;
    const lines = [
        `# ${SITE.name}`,
        '',
        `> ${SITE.description}`,
        '',
        'Every page below links to its Markdown source. The docs have two reading orders: the ' +
            'Guide is top-down (start from usage), and Concepts is bottom-up (one crate at a time, ' +
            `in dependency order). All pages concatenated: ${link(resolve('/llms-full.txt'))}`,
        `Source: ${SITE.repo}`,
    ];
    for (const track of TRACKS) {
        lines.push('', `## ${track.title}`, '', track.blurb, '');
        for (const p of pagesIn(PAGES, track.id)) {
            const href = link(resolve('/[track]/[slug].md', { track: p.track, slug: p.slug }));
            lines.push(`- [${p.title}](${href}): ${p.summary}`);
        }
    }
    return new Response(lines.join('\n') + '\n', {
        headers: { 'content-type': 'text/plain; charset=utf-8' },
    });
};
