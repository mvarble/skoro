<script lang="ts">
    import ReadingOrder from '$lib/components/ReadingOrder.svelte';
    import { SITE } from '$lib/site';

    let { data } = $props();
</script>

<svelte:head>
    <title>Reading {data.crate} · {SITE.name}</title>
    <meta name="robots" content="noindex" />
</svelte:head>

<div class="doc-layout">
    <article class="doc prose">
        <header class="doc-header">
            <h1>Reading <code>{data.crate}</code></h1>
            <p class="lede">
                Its modules in an order that uses as little as possible before it is read, drawn
                from the sources as they are now. Reload after a change.
            </p>
        </header>
        <ReadingOrder spec={data.spec} />
    </article>
</div>
