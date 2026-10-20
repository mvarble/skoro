<script lang="ts">
    import { resolve } from '$app/paths';
    import ReadingOrder from '$lib/components/ReadingOrder.svelte';
    import { READING, SITE } from '$lib/site';

    let { data } = $props();
    const reading = $derived(data.reading);
</script>

<svelte:head>
    <title>Reading {reading.crate} · {SITE.name}</title>
    <meta
        name="description"
        content="The modules of {reading.crate} in an order that uses as little as possible before it is read."
    />
</svelte:head>

<div class="doc-layout">
    <article class="doc prose">
        <header class="doc-header">
            <p class="eyebrow"><a href={resolve('/reading-order')}>{READING.title}</a></p>
            <h1>Reading <code>{reading.crate}</code></h1>
            <p class="lede">
                Its modules in an order that uses as little as possible before it is read, computed
                from the sources when the site was built.
            </p>
        </header>
        {#if reading.spec}
            <ReadingOrder spec={reading.spec} />
        {:else}
            <p>
                <code>{reading.crate}</code> has {reading.modules ? 'one module' : 'no modules'} of its
                own, so there is no order to read.
            </p>
        {/if}
    </article>
</div>
