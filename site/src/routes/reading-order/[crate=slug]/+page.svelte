<script lang="ts">
    import { resolve } from '$app/paths';
    import { page } from '$app/state';
    import ReadingOrder from '$lib/components/ReadingOrder.svelte';
    import { SITE } from '$lib/site';
    import type { NavSection } from '../../+layout.server';

    let { data } = $props();
    const reading = $derived(data.reading);
    // The reference page of the crates, which links every crate here, when the content has one.
    const crates = $derived(
        (page.data.nav as NavSection[])
            .flatMap((section) => section.items)
            .find((item) => item.key == 'reference/crates'),
    );
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
            {#if crates}
                <p class="eyebrow">
                    <a
                        href={resolve('/[track=track]/[slug=slug]', {
                            track: 'reference',
                            slug: 'crates',
                        })}>{crates.title}</a
                    >
                </p>
            {/if}
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
