import katex, { type KatexOptions } from 'katex';
import type { Element, Root } from 'hast';

// KaTeX is called directly rather than through `rehype-katex-svelte`. The
// wrapper is a dozen lines around `renderToString`, and it pinned its own copy
// of KaTeX: it rendered with 0.16 while this package depends on 0.18 and the
// blog loads 0.18's stylesheet. Those two disagree about class names --- 0.18
// renamed `base`, `strut` and `tag` to `katex-base`, `katex-strut` and
// `katex-tag` --- so the markup that reached the page was styled by rules that
// no longer matched it.
const isMath = (node: Element): 'inline' | 'display' | null => {
    const classes = node.properties?.className;
    if (!Array.isArray(classes)) return null;
    if (classes.includes('math-display')) return 'display';
    if (classes.includes('math-inline')) return 'inline';
    return null;
};

// The TeX source, which `remark-math` leaves as the element's text content.
function textOf(node: Element): string {
    let text = '';
    for (const child of node.children) {
        if (child.type == 'text') text += child.value;
        else if (child.type == 'element') text += textOf(child);
    }
    return text;
}

// Renders every `.math-inline` / `.math-display` element in place, replacing its
// children with the one text node mdsvex needs.
//
// `filename` only names the document in failure reports. `options` is passed to
// KaTeX as given, except that `macros` is copied first: KaTeX writes `\gdef`
// definitions back into that object, and a definition made in one document must
// not leak into the next.
export function renderMath(tree: Root, filename: string, options: KatexOptions = {}) {
    const settings = { ...options, macros: { ...(options.macros ?? {}) } };
    const failures: string[] = [];

    const render = (node: Root | Element) => {
        for (const child of node.children) {
            if (child.type != 'element') continue;
            const mode = isMath(child);
            if (!mode) {
                render(child);
                continue;
            }

            const tex = textOf(child);
            const displayMode = mode == 'display';
            let html: string;
            try {
                html = katex.renderToString(tex, { ...settings, displayMode, throwOnError: true });
            } catch (error) {
                // Rendered anyway, in KaTeX's error colour: a document being
                // drafted should still produce a page, with the broken formula
                // visible in place rather than a failed build.
                html = katex.renderToString(tex, { ...settings, displayMode, throwOnError: false });
                failures.push(`${summarise(error)}\n      in: ${tex.trim().split('\n')[0]}`);
            }
            // `JSON.stringify` is what makes this Svelte rather than HTML, and
            // `rehypeKatexBox` reads the escaping it produces back out again.
            child.children = [{ type: 'text', value: `{@html ${JSON.stringify(html)}}` }];
        }
    };
    render(tree);

    // Every failure the document currently has, every time it is compiled ---
    // not just the ones that are new. While you are editing macros to fix an
    // expression, an error that is still there is the thing you most need to be
    // told about, and silence reads as success. A production build compiles each
    // document twice, so it reports each of these twice; that is the cost of not
    // having to guess whether a stale error is still real.
    for (const failure of failures) console.warn(`katex: ${filename}\n    ${failure}`);
}

const summarise = (error: unknown) =>
    (error instanceof Error ? error.message : String(error)).replace(/ at position \d+.*$/s, '');
