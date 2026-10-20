<script lang="ts">
    import { resolve } from '$app/paths';
    import { page as current } from '$app/state';
    import { layers, type Crate } from '$lib/crates';
    import type { NavSection } from '../../routes/+layout.server';

    const rows = layers();
    // A crate links to its page only when the content has that page, as the navigation lists it.
    const pages = $derived(
        new Set((current.data.nav as NavSection[]).flatMap((s) => s.items.map((i) => i.key))),
    );
    const linked = (crate: Crate) => (crate.page && pages.has(crate.page) ? crate.page : undefined);
    const href = (page: string) => {
        const [track, slug] = page.split('/');
        return resolve('/[track=track]/[slug=slug]', { track, slug });
    };
</script>

<ol class="crate-map" aria-label="Workspace crates, foundations first">
    {#each rows as row, depth (depth)}
        <li class="crate-layer">
            {#each row as crate (crate.name)}
                {@const page = linked(crate)}
                {#if page}
                    <a class="crate" href={href(page)} title={crate.purpose}>
                        <code>{crate.name}</code>
                        <span>{crate.purpose}</span>
                    </a>
                {:else}
                    <div class="crate umbrella" title={crate.purpose}>
                        <code>{crate.name}</code>
                        <span>{crate.purpose}</span>
                    </div>
                {/if}
            {/each}
        </li>
    {/each}
</ol>
