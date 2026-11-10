<script lang="ts">
    import { resolve } from '$app/paths';
    import CrateMap from '$lib/components/CrateMap.svelte';
    import ExampleFigure from '$lib/components/ExampleFigure.svelte';
    import Logo from '$lib/components/Logo.svelte';
    import TrackCard from '$lib/components/TrackCard.svelte';
    import { ARCHIVE, MATH, SITE, TRACKS } from '$lib/site';

    let { data } = $props();

    const blurb = (id: string) => TRACKS.find((t) => t.id == id)!.blurb;

    // The examples' guide pages, which the showcase links to; it is shown only where they exist.
    const guidePage = (key: string) =>
        data.nav.find((section) => section.id == 'guide')?.items.find((item) => item.key == key);
    const poseGraph = $derived(guidePage('guide/pose-graph'));
    const slidingWindow = $derived(guidePage('guide/sliding-window'));
</script>

<svelte:head>
    <title>{SITE.name}: probabilistic modeling in Rust</title>
    <meta property="og:title" content="{SITE.name}: probabilistic modeling in Rust" />
    <meta name="description" content={SITE.description} />
    <meta property="og:description" content={SITE.description} />
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

{#if poseGraph && slidingWindow}
    <section class="landing-showcase">
        <ExampleFigure
            name="pose-graph"
            alt="A pose graph on SE(3): 73 poses around a closed circuit and 24 landmarks; dead reckoning drifts by 1.49 m, and the estimate stays within 34 cm of the truth, each pose inside its 90% ellipse."
        />
        <p class="showcase-lede">
            A robot's poses and the landmarks it sights, written once as a model, drawn a hundred
            times and estimated together, with uncertainties that hold the truth as often as they
            claim to.
            <a href={poseGraph.href}>Read the example →</a>
        </p>
        <div class="showcase-pair">
            <div>
                <ExampleFigure
                    name="calibration"
                    alt="Histograms of the normalized errors of 7,300 poses and 2,400 landmarks lying on the chi-squared densities they follow when the marginals are calibrated."
                />
                <p class="showcase-link">
                    <a href="{poseGraph.href}#are-the-uncertainties-honest"
                        >Are the uncertainties honest? →</a
                    >
                </p>
            </div>
            <div>
                <ExampleFigure
                    name="sliding-window"
                    alt="A sliding window over an open path of 80 poses: dead reckoning drifts by 3 m, while the window's filtered estimates and the batch stay within about a metre of the truth."
                />
                <p class="showcase-link">
                    <a href={slidingWindow.href}>A sliding window, retiring as it goes →</a>
                </p>
            </div>
        </div>
    </section>
{/if}

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

{#if data.crates}
    <section class="landing-crates">
        <h2>The workspace</h2>
        <p>
            Each crate depends only on the crates above it, and leads to the order in which to read
            its modules.
        </p>
        <CrateMap />
    </section>
{/if}
