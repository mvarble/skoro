import type { PageMeta } from '$lib/content';
import { TRACKS } from '$lib/site';

const metadata = import.meta.glob<Record<string, unknown> | undefined>('/src/content/*/*.svx', {
    eager: true,
    import: 'metadata',
});

// Every problem across every page is reported at once, and the build stops.
function index(): PageMeta[] {
    const problems: string[] = [];
    const pages: PageMeta[] = [];
    const tracks = new Set(TRACKS.map((t) => t.id));

    for (const [file, meta = {}] of Object.entries(metadata)) {
        const [, , , track, name] = file.split('/');
        const slug = name.replace(/\.svx$/, '').replace(/^\d+-/, '');
        const fail = (message: string) => problems.push(`${file}: ${message}`);

        if (!tracks.has(track)) fail(`\`${track}\` is not a track in \`src/lib/site.ts\`.`);
        for (const field of ['title', 'summary'] as const) {
            if (typeof meta[field] != 'string' || !meta[field]) {
                fail(`frontmatter \`${field}\` must be a non-empty string.`);
            }
        }
        if (typeof meta.order != 'number') fail('frontmatter `order` must be a number.');

        pages.push({
            track,
            slug,
            path: `${track}/${slug}`,
            title: meta.title as string,
            summary: meta.summary as string,
            order: meta.order as number,
            headings: (meta.headings as PageMeta['headings']) ?? [],
            source: `site${file}`,
        });
    }

    if (problems.length) throw new Error(`Invalid content:\n  ${problems.join('\n  ')}`);

    const rank = new Map(TRACKS.map((t, i) => [t.id, i]));
    return pages.sort((a, b) => rank.get(a.track)! - rank.get(b.track)! || a.order - b.order);
}

/** Every page, in reading order: tracks as listed in `TRACKS`, then by `order`. */
export const PAGES: PageMeta[] = index();

export function findPage(track: string, slug: string): PageMeta | undefined {
    return PAGES.find((p) => p.track == track && p.slug == slug);
}
