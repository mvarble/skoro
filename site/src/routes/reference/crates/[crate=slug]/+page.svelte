<script lang="ts">
    import { resolve } from '$app/paths';
    import { page } from '$app/state';
    import ReadingOrder from '$lib/components/ReadingOrder.svelte';
    import { SITE } from '$lib/site';
    import type { NavSection } from '../../../+layout.server';

    // The template of every crate's page: what the crate is for, from its description, and the
    // order in which to read its modules.
    let { data } = $props();
    const reading = $derived(data.reading);
    // The reference page of the crates, which links every crate here, when the content has one.
    const crates = $derived(
        (page.data.nav as NavSection[])
            .flatMap((section) => section.items)
            .find((item) => item.key == 'reference/crates'),
    );
    const href = (crate: string) => resolve('/reference/crates/[crate=slug]', { crate });
</script>

<svelte:head>
    <title>{reading.crate} · {SITE.name}</title>
    <meta property="og:title" content="{reading.crate} · {SITE.name}" />
    {#if data.purpose}
        <meta name="description" content={data.purpose} />
        <meta property="og:description" content={data.purpose} />
    {/if}
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
            <h1><code>{reading.crate}</code></h1>
            {#if data.purpose}<p class="lede">{data.purpose}.</p>{/if}
            {#if data.deps.length}
                <p class="crate-deps">
                    Depends on
                    {#each data.deps as dep, i (dep)}{i ? ', ' : ''}<a href={href(dep)}
                            ><code>{dep}</code></a
                        >{/each}.
                </p>
            {/if}
        </header>

        {#if data.description}
            <data.description />
        {/if}

        <h2 id="reading-order">Reading order</h2>
        {#if reading.spec}
            <p>
                The modules in an order that uses as little as possible before it is read, computed
                from the sources when the site was built. A dashed arc marks a module's use of one
                that comes later, with what it uses named beside it.
            </p>
            <ReadingOrder spec={reading.spec} />
        {:else}
            <p>
                <code>{reading.crate}</code> has {reading.modules ? 'one module' : 'no modules'} of its
                own, so there is no order to read.
            </p>
        {/if}
    </article>
</div>
