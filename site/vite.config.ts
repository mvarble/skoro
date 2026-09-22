import adapter from '@sveltejs/adapter-static';
import { sveltekit } from '@sveltejs/kit/vite';
import { mdsvex, escapeSvelte } from 'mdsvex';
import remarkMath from 'remark-math';
import { createHighlighter } from 'shiki';
import { defineConfig } from 'vitest/config';

import katex from './katex.config.ts';
import { isFileRef, REPO_ROOT, resolveFileRef } from './plugins/file-ref.ts';
import { rehypeKatexBox } from './plugins/katex-box.ts';
import { rehypeHeadings } from './plugins/rehype-headings.ts';
import { rehypeMath } from './plugins/rehype-math.ts';
import { remarkCallouts } from './plugins/remark-callouts.ts';

// Where the site is served from: `/` by default, or e.g. `BASE_PATH=/skoro` for GitHub Pages.
const base = (process.env.BASE_PATH ?? '') as '' | `/${string}`;

const REPO_URL = 'https://github.com/mvarble/skoro';
const REPO_BRANCH = 'main';

const highlighter = await createHighlighter({
    themes: ['vitesse-light', 'vitesse-dark'],
    langs: ['rust', 'toml', 'sh', 'text', 'yaml', 'markdown'],
});

export default defineConfig({
    plugins: [
        sveltekit({
            compilerOptions: {
                // Force runes mode for the project, except for libraries. Can be removed in svelte 6.
                runes: ({ filename }) =>
                    filename.split(/[/\\]/).includes('node_modules') ? undefined : true,
                // mdsvex 0.12 still emits `<script context="module">` for frontmatter.
                warningFilter: (warning) => warning.code != 'script_context_deprecated',
            },
            adapter: adapter({ strict: true }),
            paths: { base, relative: false },
            prerender: {
                entries: ['*', '/llms.txt', '/llms-full.txt'],
            },
            preprocess: [
                mdsvex({
                    remarkPlugins: [remarkMath, remarkCallouts],
                    // Order matters: headings read their text before math turns it opaque.
                    rehypePlugins: [rehypeHeadings, [rehypeMath, katex], rehypeKatexBox],
                    highlight: { highlighter: highlight },
                    extensions: ['.svx', '.md'],
                }),
            ],
            extensions: ['.svelte', '.svx', '.md'],
        }),
    ],
    test: {
        include: ['src/**/*.test.ts'],
    },
    build: {
        // A math-heavy page is one chunk of pre-rendered KaTeX markup, which is large by nature.
        chunkSizeWarningLimit: 1000,
    },
    server: {
        // File references quote from `crates/` and `docs/`.
        fs: { allow: [REPO_ROOT] },
    },
});

async function highlight(code: string, lang: string | null | undefined, _: unknown, file?: string) {
    let caption = '';
    if (isFileRef(code)) {
        const ref = resolveFileRef(code, file!, lang ?? undefined);
        code = ref.code;
        lang = ref.lang;
        const href = `${REPO_URL}/blob/${REPO_BRANCH}/${ref.repoPath}#L${ref.start}-L${ref.end}`;
        caption =
            `<figcaption><a href="${href}" rel="external">` +
            `<code>${ref.repoPath}</code> <span>lines ${ref.start}–${ref.end}</span></a></figcaption>`;
    }
    const html = highlighter.codeToHtml(code, {
        lang: lang || 'text',
        themes: { light: 'vitesse-light', dark: 'vitesse-dark' },
        defaultColor: false,
    });
    const figure = `<figure class="code-block">${caption}${html}</figure>`;
    return `{@html \`${escapeSvelte(figure)}\` }`;
}
