<script lang="ts">
    import PageList from '$lib/components/PageList.svelte';
    import { ARCHIVE, SITE } from '$lib/site';

    let { data } = $props();
</script>

<svelte:head>
    <title>{ARCHIVE.title} · {SITE.name}</title>
    <meta name="description" content={ARCHIVE.blurb} />
</svelte:head>

<div class="doc-layout">
    <article class="doc track-index">
        <header class="doc-header">
            <h1>{ARCHIVE.title}</h1>
            <p class="lede">{ARCHIVE.blurb}</p>
        </header>
        <div class="prose">
            <p>
                An entry is opened for each piece of development and moves through its stages: for a
                feature, the mathematics, then the implementation, the plan, the build, and last the
                pages that say what the feature made true. What each stage approved is kept here as
                it was approved. When the library later changes, a new entry revises the old one
                rather than rewriting it.
            </p>
        </div>
        {#if data.active.length}
            <h2>In progress</h2>
            <PageList links={data.active} empty="" />
        {/if}
        <h2>Done</h2>
        <PageList links={data.done} empty="No entry is finished yet." />
        {#if data.abandoned.length}
            <h2>Set aside</h2>
            <p class="muted">
                Worked on, then put down. Their documents stay, should they be taken up again.
            </p>
            <PageList links={data.abandoned} empty="" />
        {/if}
    </article>
</div>
