import prettier from 'eslint-config-prettier';
import path from 'node:path';
import js from '@eslint/js';
import svelte from 'eslint-plugin-svelte';
import globals from 'globals';
import ts from 'typescript-eslint';
import { defineConfig, includeIgnoreFile } from 'eslint/config';

const gitignorePath = path.resolve(import.meta.dirname, '.gitignore');

export default defineConfig(
    includeIgnoreFile(gitignorePath),
    js.configs.recommended,
    ts.configs.recommended,
    svelte.configs.recommended,
    prettier,
    svelte.configs.prettier,
    {
        languageOptions: {
            globals: { ...globals.browser, ...globals.node },
        },
        rules: {
            'no-undef': 'off',
        },
    },
    {
        files: ['**/*.svelte', '**/*.svelte.ts', '**/*.svelte.js'],
        languageOptions: {
            parserOptions: {
                projectService: true,
                extraFileExtensions: ['.svelte'],
                parser: ts.parser,
            },
        },
    },
    {
        ignores: ['./build'],
    },
    {
        // These insert KaTeX output, and text that `renderMixed` has escaped, from specs authored
        // in this repository: nothing a visitor supplies reaches them.
        files: [
            'src/lib/components/Tex.svelte',
            'src/lib/components/ModelGraph.svelte',
            'src/lib/components/Walkthrough.svelte',
        ],
        rules: { 'svelte/no-at-html-tags': 'off' },
    },
    {
        // The same for titles, summaries and references from `content/`, rendered by `renderMixed`
        // or `referenceHtml`. Their links are made by `resolve()` on the server (`hrefOf`, and the
        // navigation in `+layout.server.ts`) and arrive as data.
        files: [
            'src/lib/components/DocumentPage.svelte',
            'src/lib/components/LinkList.svelte',
            'src/lib/components/PageList.svelte',
            'src/lib/components/PrevNext.svelte',
            'src/lib/components/Sidebar.svelte',
            'src/lib/components/Toc.svelte',
            'src/lib/components/TopBar.svelte',
            'src/lib/components/TrackCard.svelte',
        ],
        rules: {
            'svelte/no-at-html-tags': 'off',
            'svelte/no-navigation-without-resolve': 'off',
        },
    },
    {
        rules: {
            'comma-dangle': [2, 'always-multiline'],
        },
    },
);
