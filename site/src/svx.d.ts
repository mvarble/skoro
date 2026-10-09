declare module '*.svx' {
    import type { Component } from 'svelte';
    const component: Component;
    export default component;
    export const metadata: Record<string, unknown>;
}

declare module '*.md' {
    import type { Component } from 'svelte';
    const component: Component;
    export default component;
    export const metadata: Record<string, unknown>;
}

// The content database, which `vite.config.ts` serves under these names.
declare module '$cms' {
    import type { SkoroCms } from '@mvarble/mesearch-cms/presets/skoro/runtime';
    export const cms: SkoroCms;
}

declare module '$cms/loaders' {
    import type { Component } from 'svelte';
    const loaders: Record<string, () => Promise<{ default: Component }>>;
    export default loaders;
}
