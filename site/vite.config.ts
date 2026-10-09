import adapter from '@sveltejs/adapter-static';
import { sveltekit } from '@sveltejs/kit/vite';
import { createCms } from '@mvarble/mesearch-cms';
import { skoroPreset } from '@mvarble/mesearch-cms/presets/skoro';
import { rehypeKatex, rehypeMathBox, type PluginEntry } from '@mvarble/mesearch-markdown';
import { mdsvex, escapeSvelte } from 'mdsvex';
import remarkMath from 'remark-math';
import { createHighlighter } from 'shiki';
import { defineConfig } from 'vitest/config';

import katex from './katex.config.ts';
import { isFileRef, REPO_ROOT, resolveFileRef } from './plugins/file-ref.ts';
import { rehypeHeadingAnchors } from './plugins/rehype-heading-anchors.ts';
import { remarkCallouts } from './plugins/remark-callouts.ts';
import { ARCHIVE, DOCUMENTS, MATH, TRACKS } from './src/lib/site.ts';

// Where the site is served from: `/` by default, or e.g. `BASE_PATH=/skoro` for GitHub Pages.
const base = (process.env.BASE_PATH ?? '') as '' | `/${string}`;

const REPO_URL = 'https://github.com/mvarble/skoro';
const REPO_BRANCH = 'main';

// The content database: every document under `content/`, with its links, numbered equations and
// statements resolved, served to the routes as `$cms`.
const cms = createCms({
    preset: skoroPreset({
        // `SITE_CONTENT=test/content pnpm build` builds the fixture, which has a document of every
        // kind, instead of the real content.
        contentDir: process.env.SITE_CONTENT ?? 'content',
        tracks: TRACKS.map((track) => track.id),
        documents: Object.fromEntries(DOCUMENTS.map((doc) => [doc.name, doc.title])),
        sectionTitles: Object.fromEntries(
            [...TRACKS, MATH, ARCHIVE].map((section) => [section.id, section.title]),
        ),
        base,
    }),
    macros: katex.macros,
    virtualId: '$cms',
});

const highlighter = await createHighlighter({
    themes: ['vitesse-light', 'vitesse-dark'],
    langs: ['rust', 'toml', 'sh', 'text', 'yaml', 'markdown', 'ts', 'json'],
});

export default defineConfig({
    plugins: [
        ...cms.vite(),
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
                // A collection with no documents yet (no math notes, an empty archive) leaves its
                // document route with nothing to render.
                handleUnseenRoutes: 'ignore',
            },
            preprocess: [
                mdsvex({
                    remarkPlugins: [remarkMath, remarkCallouts, cms.remark],
                    // mdsvex runs an older unified than the shared plugins are typed against.
                    rehypePlugins: [
                        rehypeHeadingAnchors,
                        [rehypeKatex, { macros: cms.macrosFor }],
                        [rehypeMathBox, { liftTags: true }],
                    ] as PluginEntry[],
                    highlight: { highlighter: highlight },
                    extensions: ['.svx', '.md'],
                }),
            ],
            extensions: ['.svelte', '.svx', '.md'],
        }),
    ],
    test: {
        include: ['src/**/*.test.ts', 'plugins/**/*.test.ts'],
    },
    build: {
        // A math-heavy page is one chunk of pre-rendered KaTeX markup, which is large by nature.
        chunkSizeWarningLimit: 1000,
    },
    server: {
        // File references quote from anywhere in the repository.
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
