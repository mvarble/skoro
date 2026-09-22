import GithubSlugger from 'github-slugger';
import type { Element, Root } from 'hast';
import type { Plugin } from 'unified';
import type { VFile } from 'vfile';

export interface Heading {
    depth: 2 | 3;
    text: string;
    slug: string;
}

function textOf(node: Element): string {
    let text = '';
    for (const child of node.children) {
        if (child.type == 'text') text += child.value;
        else if (child.type == 'element') text += textOf(child);
    }
    return text;
}

// Gives every `h2`/`h3` an `id` and a trailing anchor link, and records the list on the page's
// frontmatter as `headings`. mdsvex serializes `vfile.data.fm` into `export const metadata` only
// after every user rehype plugin has run, so the table of contents is known at build time.
//
// This must run before `rehypeMath`: afterwards, a heading's math is an opaque `{@html ..}` string.
// A heading containing math will therefore show its TeX source in the table of contents.
export const rehypeHeadings: Plugin<[]> = () => {
    return (tree, vfile) => {
        const file = vfile as VFile & { data: { fm?: Record<string, unknown> } };
        const slugger = new GithubSlugger();
        const headings: Heading[] = [];

        const visit = (node: Root | Element) => {
            for (const child of node.children) {
                if (child.type != 'element') continue;
                if (child.tagName == 'h2' || child.tagName == 'h3') {
                    const text = textOf(child).trim();
                    const slug = slugger.slug(text);
                    child.properties = { ...child.properties, id: slug };
                    child.children.push({
                        type: 'element',
                        tagName: 'a',
                        properties: {
                            className: ['heading-anchor'],
                            href: `#${slug}`,
                            ariaLabel: `Link to this section: ${text}`,
                        },
                        children: [{ type: 'text', value: '#' }],
                    });
                    headings.push({ depth: child.tagName == 'h2' ? 2 : 3, text, slug });
                    continue;
                }
                visit(child);
            }
        };
        visit(tree as Root);

        file.data.fm = { ...(file.data.fm ?? {}), headings };
    };
};
