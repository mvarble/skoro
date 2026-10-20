<script lang="ts">
    import { resolve } from '$app/paths';
    import { SITE } from '$lib/site';

    let { data } = $props();
</script>

<svelte:head>
    <title>Reading the crates · {SITE.name}</title>
    <meta
        name="description"
        content="Each crate's modules in an order that uses as little as possible before it is read."
    />
</svelte:head>

<div class="doc-layout">
    <article class="doc prose">
        <header class="doc-header">
            <h1>Reading the crates</h1>
            <p class="lede">
                Each crate's modules in an order that uses as little as possible before it is read,
                computed from the sources when the site was built.
            </p>
        </header>
        {#if data.readings.length}
            <table>
                <thead>
                    <tr><th>Crate</th><th>Modules</th><th>Uses of a later module</th></tr>
                </thead>
                <tbody>
                    {#each data.readings as r (r.crate)}
                        <tr>
                            <td>
                                <a href={resolve('/reading-order/[crate=slug]', { crate: r.crate })}
                                    ><code>{r.crate}</code></a
                                >
                            </td>
                            <td>{r.modules}</td>
                            <td>
                                {r.modules < 2
                                    ? 'nothing to order'
                                    : `${r.ahead}, naming ${r.items} item${r.items == 1 ? '' : 's'}`}
                            </td>
                        </tr>
                    {/each}
                </tbody>
            </table>
        {:else}
            <p>The workspace has no crates yet.</p>
        {/if}
    </article>
</div>
