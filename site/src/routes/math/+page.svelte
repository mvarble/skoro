<script lang="ts">
    import MathMap from '$lib/components/MathMap.svelte';
    import PageList from '$lib/components/PageList.svelte';
    import { MATH, SITE } from '$lib/site';

    let { data } = $props();
</script>

<svelte:head>
    <title>{MATH.title} · {SITE.name}</title>
    <meta name="description" content={MATH.blurb} />
</svelte:head>

<div class="doc-layout">
    <article class="doc track-index">
        <header class="doc-header">
            <h1>{MATH.title}</h1>
            <p class="lede">{MATH.blurb}</p>
        </header>
        {#if data.map.nodes.length}
            <MathMap {...data.map} />
        {/if}
        <h2>Notes</h2>
        <PageList links={data.notes} empty="No notes yet." />
    </article>
</div>
