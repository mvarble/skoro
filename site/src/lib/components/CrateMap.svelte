<script lang="ts">
    import { resolve } from '$app/paths';
    import { layers } from '$lib/crates';

    const rows = layers();
    const href = (page: string) => {
        const [track, slug] = page.split('/');
        return resolve('/[track]/[slug]', { track, slug });
    };
</script>

<ol class="crate-map" aria-label="Workspace crates, foundations first">
    {#each rows as row, depth (depth)}
        <li class="crate-layer">
            {#each row as crate (crate.name)}
                {#if crate.page}
                    <a class="crate" href={href(crate.page)} title={crate.purpose}>
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
