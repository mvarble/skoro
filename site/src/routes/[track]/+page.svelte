<script lang="ts">
    import { resolve } from '$app/paths';
    import { SITE } from '$lib/site';

    let { data } = $props();
</script>

<svelte:head>
    <title>{data.track.title} · {SITE.name}</title>
    <meta name="description" content={data.track.blurb} />
</svelte:head>

<!-- The same frame as a documentation page, so the two line up. -->
<div class="doc-layout">
    <article class="doc track-index">
        <header class="doc-header">
            <h1>{data.track.title}</h1>
            <p class="lede">{data.track.blurb}</p>
        </header>
        <ol class="page-list">
            {#each data.trackPages as p, i (p.path)}
                <li>
                    <a href={resolve('/[track]/[slug]', { track: p.track, slug: p.slug })}>
                        <span class="index">{String(i + 1).padStart(2, '0')}</span>
                        <span class="title">{p.title}</span>
                        <span class="summary">{p.summary}</span>
                    </a>
                </li>
            {/each}
        </ol>
    </article>
</div>
