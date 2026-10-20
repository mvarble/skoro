<script lang="ts">
    import { resolve } from '$app/paths';
    import { layers } from '$lib/crates';

    // Each crate links to the order in which to read its modules.
    const rows = layers();
    const href = (crate: string) => resolve('/reading-order/[crate=slug]', { crate });
</script>

<ol class="crate-map" aria-label="Workspace crates, foundations first">
    {#each rows as row, depth (depth)}
        <li class="crate-layer">
            {#each row as crate (crate.name)}
                <a class="crate" href={href(crate.name)} title={crate.purpose}>
                    <code>{crate.name}</code>
                    <span>{crate.purpose}</span>
                </a>
            {/each}
        </li>
    {/each}
</ol>
