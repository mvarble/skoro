<script lang="ts">
    import { resolve } from '$app/paths';
    import { layers } from '$lib/crates';

    // Each crate links to its page in the reference.
    const rows = layers();
    const href = (crate: string) => resolve('/reference/crates/[crate=slug]', { crate });
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
