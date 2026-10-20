<script lang="ts">
    import { resolve } from '$app/paths';
    import CrateMap from '$lib/components/CrateMap.svelte';
    import Logo from '$lib/components/Logo.svelte';
    import TrackCard from '$lib/components/TrackCard.svelte';
    import { ARCHIVE, MATH, SITE, TRACKS } from '$lib/site';

    let { data } = $props();

    const blurb = (id: string) => TRACKS.find((t) => t.id == id)!.blurb;
</script>

<svelte:head>
    <title>{SITE.name}: probabilistic modeling in Rust</title>
    <meta name="description" content={SITE.description} />
</svelte:head>

<section class="hero">
    <p class="eyebrow">Rust · burn · second-order autodiff · manifolds</p>
    <h1><Logo variant="lockup" /><span class="visually-hidden">{SITE.name}</span></h1>
    <p class="hero-tagline">{SITE.tagline}</p>
    <div class="hero-actions">
        <a class="button primary" href={resolve('/[track=track]', { track: 'start' })}
            >Get started</a
        >
        <a class="button" href={SITE.repo} rel="external">View source</a>
    </div>
</section>

{#if data.component}
    <section class="landing-body prose">
        <data.component />
    </section>
{/if}

<section class="landing-tracks">
    <h2>Two ways to read</h2>
    <div class="card-grid">
        <TrackCard id="guide" eyebrow="Top-down" blurb={blurb('guide')} />
        <TrackCard id="concepts" eyebrow="Bottom-up" blurb={blurb('concepts')} />
    </div>
    <div class="card-grid secondary">
        <TrackCard id="math" eyebrow="What it rests on" blurb={MATH.blurb} />
        <TrackCard id="archive" eyebrow="How it came to be" blurb={ARCHIVE.blurb} />
        <TrackCard id="reference" eyebrow="Look it up" blurb={blurb('reference')} />
    </div>
</section>

{#if data.crates || data.reading}
    <section class="landing-crates">
        <h2>The workspace</h2>
        {#if data.crates}
            <p>
                Each crate depends only on the crates above it. The bottom-up track follows this
                order.
            </p>
            <CrateMap />
        {/if}
        {#if data.reading}
            <p>
                <a href={resolve('/reading-order')}>Reading the crates</a>: each crate's modules in
                an order that uses as little as possible before it is read.
            </p>
        {/if}
    </section>
{/if}
