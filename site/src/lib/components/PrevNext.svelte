<script lang="ts">
    import { resolve } from '$app/paths';
    import type { PageMeta } from '$lib/content';
    import Icon from './Icon.svelte';

    let { prev, next }: { prev?: PageMeta; next?: PageMeta } = $props();
</script>

<nav class="prev-next" aria-label="Previous and next pages">
    {#if prev}
        <a class="prev" href={resolve('/[track]/[slug]', { track: prev.track, slug: prev.slug })}>
            <span class="label"><Icon name="arrowLeft" size={14} /> Previous</span>
            <span class="title">{prev.title}</span>
        </a>
    {/if}
    {#if next}
        <a class="next" href={resolve('/[track]/[slug]', { track: next.track, slug: next.slug })}>
            <span class="label">Next <Icon name="arrowRight" size={14} /></span>
            <span class="title">{next.title}</span>
        </a>
    {/if}
</nav>
