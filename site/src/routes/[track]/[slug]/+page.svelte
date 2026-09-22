<script lang="ts">
    import { resolve } from '$app/paths';
    import CopyCode from '$lib/components/CopyCode.svelte';
    import Icon from '$lib/components/Icon.svelte';
    import PrevNext from '$lib/components/PrevNext.svelte';
    import Toc from '$lib/components/Toc.svelte';
    import { SITE } from '$lib/site';

    let { data } = $props();
    let body = $state<HTMLElement>();
</script>

<svelte:head>
    <title>{data.page.title} · {SITE.name}</title>
    <meta name="description" content={data.page.summary} />
    <link
        rel="alternate"
        type="text/markdown"
        href={resolve('/[track]/[slug].md', { track: data.page.track, slug: data.page.slug })}
    />
</svelte:head>

<div class="doc-layout">
    <article class="doc">
        <header class="doc-header">
            <p class="breadcrumb">
                <a href={resolve('/[track]', { track: data.track.id })}>{data.track.title}</a>
            </p>
            <h1>{data.page.title}</h1>
            <p class="lede">{data.page.summary}</p>
        </header>

        <div class="prose" bind:this={body}>
            <data.component />
        </div>
        <CopyCode root={body} />

        <footer class="doc-footer">
            <a href="{SITE.repo}/blob/{SITE.branch}/{data.page.source}" rel="external">
                <Icon name="github" size={14} /> Edit this page
            </a>
            <a
                href={resolve('/[track]/[slug].md', {
                    track: data.page.track,
                    slug: data.page.slug,
                })}
            >
                <Icon name="file" size={14} /> View as Markdown
            </a>
        </footer>
        <PrevNext prev={data.prev} next={data.next} />
    </article>

    <aside class="doc-aside">
        <Toc headings={data.page.headings} />
    </aside>
</div>
