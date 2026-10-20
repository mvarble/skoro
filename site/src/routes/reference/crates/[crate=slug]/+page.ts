import type { Component } from 'svelte';

import type { PageLoad } from './$types';

// Each crate's description, written at publish as `content/reference/crates/<crate>/index.md`. The
// content database does not read these files as documents; this page renders them, as it would a
// page's body, from the content the server says is being built.
const descriptions = import.meta.glob<{ default: Component }>([
    '/content/reference/crates/*/index.md',
    '/test/content/reference/crates/*/index.md',
]);

export const load: PageLoad = async ({ data }) => {
    const load = descriptions[data.description];
    return { ...data, description: load ? (await load()).default : undefined };
};
