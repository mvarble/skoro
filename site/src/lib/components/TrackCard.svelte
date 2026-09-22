<script lang="ts">
    import { resolve } from '$app/paths';
    import type { Track } from '$lib/site';
    import { page } from '$app/state';
    import { pagesIn } from '$lib/content';

    let { track, eyebrow }: { track: Track; eyebrow?: string } = $props();
    const first = $derived(pagesIn(page.data.pages, track.id)[0]);
</script>

<a
    class="card track-card"
    href={first
        ? resolve('/[track]/[slug]', { track: first.track, slug: first.slug })
        : resolve('/[track]', { track: track.id })}
>
    {#if eyebrow}<span class="eyebrow">{eyebrow}</span>{/if}
    <span class="card-title">{track.title}</span>
    <span class="card-body">{track.blurb}</span>
    {#if first}<span class="card-cta">Begin with “{first.title}” →</span>{/if}
</a>
