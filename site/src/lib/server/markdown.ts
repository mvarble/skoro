import fs from 'node:fs';
import path from 'node:path';

import { base } from '$app/paths';
import type { SkoroDocument } from '$lib/documents';
import { describeGraph } from '$lib/graph/describe';
import type { GraphSpec } from '$lib/graph/spec';
import { expandFileRefs } from '../../../plugins/file-ref';

// Every `<ModelGraph>` spec, which by convention lives in a `*.graph.ts` file beside its page.
const graphModules = import.meta.glob<Record<string, GraphSpec>>('/content/**/*.graph.ts', {
    eager: true,
});

// A document as Markdown for a reader without a browser: the raw source, with its frontmatter
// kept so its KaTeX macros stay visible, a `# title` heading added after it, the `<script>` block
// that imports components dropped, every `<ModelGraph spec={name} />` replaced by a text
// description of the graph, every file reference replaced by the code it quotes, and every link
// to another document pointed at that document's Markdown.
//
// The build runs from `site/`, which is where the content database's filenames start.
export function markdownOf(doc: SkoroDocument): string {
    const file = path.resolve(process.cwd(), doc.filename);
    const raw = fs.readFileSync(file, 'utf8');
    const match = /^---\n[\s\S]*?\n---\n/.exec(raw);
    const frontmatter = match ? match[0] : '';
    const rest = raw.slice(frontmatter.length);
    const script = /^\s*<script[\s\S]*?<\/script>\s*/.exec(rest)?.[0] ?? '';
    const body = inlineStatements(
        describeGraphs(rest.slice(script.length).trim(), script, doc),
        script,
        file,
    );
    const markdown = relinked(expandFileRefs(body, file), doc);
    return `${frontmatter}\n# ${doc.title}\n\n${markdown}\n`;
}

// Resolves each `<ModelGraph spec={name} />` through the document's own `import { name } from
// './x.graph'`, and replaces the tag with `describeGraph` of that spec.
function describeGraphs(body: string, script: string, doc: SkoroDocument): string {
    const dir = path.posix.dirname(`/${doc.filename}`);
    const specs = new Map<string, GraphSpec>();
    for (const [, names, from] of script.matchAll(
        /import\s*\{([^}]*)\}\s*from\s*['"](\.[^'"]+)['"]/g,
    )) {
        const module = graphModules[path.posix.join(dir, from!.replace(/(\.ts)?$/, '.ts'))];
        if (!module) continue;
        for (const name of names!
            .split(',')
            .map((n) => n.trim())
            .filter(Boolean)) {
            specs.set(name, module[name]!);
        }
    }
    return body.replace(/<ModelGraph\s+spec=\{(\w+)\}\s*\/>/g, (tag, name: string) => {
        const spec = specs.get(name);
        if (!spec) throw new Error(`${doc.filename}: \`${tag}\` names no imported graph spec.`);
        return describeGraph(spec);
    });
}

// Replaces each `<Statement {...name} />` with the statement it spreads in, imported as
// `import * as name from './statements/x.md'`, as a quotation headed by its kind and title.
function inlineStatements(body: string, script: string, file: string): string {
    const imports = new Map<string, string>();
    for (const [, name, from] of script.matchAll(
        /import\s*\*\s*as\s+(\w+)\s+from\s*['"](\.[^'"]+)['"]/g,
    )) {
        imports.set(name!, path.resolve(path.dirname(file), from!));
    }
    return body.replace(/<Statement\s+\{\.\.\.(\w+)\}[^>]*\/>/g, (tag, name: string) => {
        const source = imports.get(name);
        if (!source || !fs.existsSync(source)) return tag;
        const raw = fs.readFileSync(source, 'utf8');
        const frontmatter = /^---\n([\s\S]*?)\n---\n/.exec(raw);
        const field = (key: string) =>
            new RegExp(`^${key}:\\s*(.+)$`, 'm').exec(frontmatter?.[1] ?? '')?.[1]?.trim();
        const kind = field('kind') ?? 'statement';
        const title = field('title');
        const heading = `**${kind.slice(0, 1).toUpperCase()}${kind.slice(1)}${title ? ` (${title})` : ''}.**`;
        const text = raw.slice(frontmatter?.[0].length ?? 0).trim();
        return `${heading} ${text}`
            .split('\n')
            .map((line) => (line ? `> ${line}` : '>'))
            .join('\n');
    });
}

// Links are written the way the files sit on disk (`../../math/retraction/`, `plan.md`). Here they
// become absolute links to the Markdown of the document they name.
function relinked(markdown: string, doc: SkoroDocument): string {
    const dir = path.posix.dirname(doc.filename);
    return markdown.replace(/\]\((\.{0,2}[^):\s]*?)(#[^)\s]*)?\)/g, (whole, href: string, hash) => {
        if (!href || /^[a-z]+:/i.test(href) || href.startsWith('/')) return whole;
        if (/\.(?!md$|svx$)[a-z0-9]+$/i.test(href)) return whole;
        const target = path.posix.normalize(path.posix.join(dir, href));
        const key = path.posix
            .relative('content', target)
            .replace(/\/+$/, '')
            .replace(/(^|\/)index\.(md|svx)$/, '')
            .replace(/\.(md|svx)$/, '');
        if (key.startsWith('..')) return whole;
        return `](${base}/${key}.md${hash ?? ''})`;
    });
}
