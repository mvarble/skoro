<script lang="ts">
    import { resolve } from '$app/paths';
    import CrateMap from '$lib/components/CrateMap.svelte';
    import TrackCard from '$lib/components/TrackCard.svelte';
    import { SITE, TRACKS } from '$lib/site';
    import Landing from '../content/landing.svx';

    const track = (id: string) => TRACKS.find((t) => t.id == id)!;
</script>

<svelte:head>
    <title>{SITE.name}: probabilistic modeling in Rust</title>
    <meta name="description" content={SITE.description} />
</svelte:head>

<section class="hero">
    <p class="eyebrow">Rust · burn · second-order autodiff · manifolds</p>
    <h1>{SITE.name}</h1>
    <p class="hero-tagline">{SITE.tagline}</p>
    <div class="hero-actions">
        <a
            class="button primary"
            href={resolve('/[track]/[slug]', { track: 'start', slug: 'what-is-skoro' })}
            >Get started</a
        >
        <a class="button" href={SITE.repo} rel="external">View source</a>
    </div>
</section>

<section class="landing-body prose">
    <Landing />
</section>

<section class="landing-tracks">
    <h2>Two ways to read</h2>
    <div class="card-grid">
        <TrackCard track={track('guide')} eyebrow="Top-down" />
        <TrackCard track={track('concepts')} eyebrow="Bottom-up" />
    </div>
    <div class="card-grid secondary">
        <TrackCard track={track('design')} eyebrow="Why" />
        <TrackCard track={track('reference')} eyebrow="Look it up" />
    </div>
</section>

<section class="landing-crates">
    <h2>The workspace</h2>
    <p>Each crate depends only on the crates above it. The bottom-up track follows this order.</p>
    <CrateMap />
</section>
