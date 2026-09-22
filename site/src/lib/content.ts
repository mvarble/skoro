import type { Component } from 'svelte';

import type { Heading } from '../../plugins/rehype-headings';

export type { Heading };

export interface PageMeta {
    track: string;
    slug: string;
    /** `track/slug`, the page's path below the base. */
    path: string;
    title: string;
    summary: string;
    order: number;
    headings: Heading[];
    /** Path of the source file from the repository root. */
    source: string;
}

// The page index is built on the server (`$lib/server/content.ts`) and reaches the browser as the
// root layout's data. Only the components are imported here, lazily, so a page ships its own
// content and nothing else.
const components = import.meta.glob<Component>('/src/content/*/*.svx', { import: 'default' });

export function loadComponent(page: PageMeta): Promise<Component> {
    return components[page.source.slice('site'.length)]();
}

export function pagesIn(pages: PageMeta[], track: string): PageMeta[] {
    return pages.filter((p) => p.track == track);
}

/** The neighbours of a page within its own track. */
export function neighbours(
    pages: PageMeta[],
    page: PageMeta,
): { prev?: PageMeta; next?: PageMeta } {
    const siblings = pagesIn(pages, page.track);
    const i = siblings.findIndex((p) => p.path == page.path);
    return { prev: siblings[i - 1], next: siblings[i + 1] };
}
