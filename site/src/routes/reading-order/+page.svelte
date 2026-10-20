<script lang="ts">
    import { resolve } from '$app/paths';
    import { READING, SITE } from '$lib/site';

    let { data } = $props();
</script>

<svelte:head>
    <title>{READING.title} · {SITE.name}</title>
    <meta name="description" content={READING.blurb} />
</svelte:head>

<div class="doc-layout">
    <article class="doc prose">
        <header class="doc-header">
            <h1>{READING.title}</h1>
            <p class="lede">{READING.blurb}</p>
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
