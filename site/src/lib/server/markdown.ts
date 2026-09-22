import path from 'node:path';

import { base } from '$app/paths';
import type { PageMeta } from '$lib/content';
import { describeGraph } from '$lib/graph/describe';
import type { GraphSpec } from '$lib/graph/spec';
import { expandFileRefs, REPO_ROOT } from '../../../plugins/file-ref';

// Server-only, so that no page's source is shipped to the browser.
const raws = import.meta.glob<string>('/src/content/*/*.svx', {
    eager: true,
    query: '?raw',
    import: 'default',
});

// Every `<ModelGraph>` spec, which by convention lives in a `*.graph.ts` file beside its page.
const graphModules = import.meta.glob<Record<string, GraphSpec>>('/src/content/**/*.graph.ts', {
    eager: true,
});

// A page as Markdown for a reader without a browser: the raw `.svx`, with its frontmatter kept so
// the page's KaTeX macros stay visible, a `# title` heading added after it, the `<script>` block
// that imports components dropped, every `<ModelGraph spec={name} />` replaced by a text
// description of the graph, and every file reference replaced by the code it quotes.
export function markdownOf(page: PageMeta): string {
    const file = path.join(REPO_ROOT, page.source);
    const raw = raws[page.source.slice('site'.length)];
    const match = /^---\n[\s\S]*?\n---\n/.exec(raw);
    const frontmatter = match ? match[0] : '';
    const rest = raw.slice(frontmatter.length);
    const script = /^\s*<script[\s\S]*?<\/script>\s*/.exec(rest)?.[0] ?? '';
    const body = describeGraphs(rest.slice(script.length).trim(), script, page);
    const markdown = relinked(expandFileRefs(body, file), page);
    return `${frontmatter}\n# ${page.title}\n\n${markdown}\n`;
}

// Resolves each `<ModelGraph spec={name} />` through the page's own `import { name } from
// './x.graph'`, and replaces the tag with `describeGraph` of that spec.
function describeGraphs(body: string, script: string, page: PageMeta): string {
    const dir = path.posix.dirname(page.source.slice('site'.length));
    const specs = new Map<string, GraphSpec>();
    for (const [, names, from] of script.matchAll(
        /import\s*\{([^}]*)\}\s*from\s*['"](\.[^'"]+)['"]/g,
    )) {
        const module = graphModules[path.posix.join(dir, from.replace(/(\.ts)?$/, '.ts'))];
        if (!module) continue;
        for (const name of names
            .split(',')
            .map((n) => n.trim())
            .filter(Boolean)) {
            specs.set(name, module[name]);
        }
    }
    return body.replace(/<ModelGraph\s+spec=\{(\w+)\}\s*\/>/g, (tag, name: string) => {
        const spec = specs.get(name);
        if (!spec) throw new Error(`${page.source}: \`${tag}\` names no imported graph spec.`);
        return describeGraph(spec);
    });
}

// A page's relative links are written for its HTML URL (`/track/slug/`). In the Markdown they are
// rewritten to absolute paths, and a link to another page points at that page's `.md` version.
function relinked(markdown: string, page: PageMeta): string {
    const origin = 'http://site';
    const from = `${origin}${base}/${page.track}/${page.slug}/`;
    return markdown.replace(/\]\((\.{1,2}\/[^)\s]*)\)/g, (_, href: string) => {
        const url = new URL(href, from);
        const inner = url.pathname.slice(base.length);
        const match = /^\/([\w-]+)\/([\w-]+)\/$/.exec(inner);
        const target = match ? `${base}/${match[1]}/${match[2]}.md` : url.pathname;
        return `](${target}${url.hash})`;
    });
}
