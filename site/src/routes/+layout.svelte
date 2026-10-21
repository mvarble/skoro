<script lang="ts">
    import '../styles/index.css';

    import { asset, resolve } from '$app/paths';
    import { page } from '$app/state';
    import Sidebar from '$lib/components/Sidebar.svelte';
    import TopBar from '$lib/components/TopBar.svelte';
    import { SITE } from '$lib/site';

    let { data, children } = $props();
    let menuOpen = $state(false);

    const landing = $derived(page.route.id === '/');
</script>

<svelte:head>
    <link rel="alternate" type="text/plain" title="llms.txt" href={resolve('/llms.txt')} />
    <link rel="icon" href={asset('/favicon.svg')} />
    <!-- The card a shared link shows, drawn by `logo/generate.py`. -->
    <meta property="og:site_name" content={SITE.name} />
    <meta property="og:type" content="website" />
    {#if data.origin}<meta property="og:url" content={data.origin + page.url.pathname} />{/if}
    <meta property="og:image" content={data.origin + asset('/og-image.png')} />
    <meta property="og:image:width" content="1200" />
    <meta property="og:image:height" content="630" />
    <meta property="og:image:alt" content="The skoro logo: a stitched eight-pointed star" />
    <meta name="twitter:card" content="summary_large_image" />
</svelte:head>

<a class="skip-link" href="#content">Skip to content</a>
<TopBar bind:menuOpen showMenu={!landing} />

<div class="shell" class:landing>
    {#if !landing}
        <Sidebar bind:open={menuOpen} />
    {/if}
    <main id="content">
        {@render children()}
    </main>
</div>

<footer class="site-footer">
    <p>
        {SITE.name} is developed at
        <a href={SITE.repo} rel="external">{SITE.repo.replace('https://', '')}</a>.
    </p>
    <p>
        For language models: <a href={resolve('/llms.txt')}>llms.txt</a> (index) ·
        <a href={resolve('/llms-full.txt')}>llms-full.txt</a> (every page, as Markdown).
    </p>
</footer>
