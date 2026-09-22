<script lang="ts">
    import { resolve } from '$app/paths';
    import { page } from '$app/state';
    import { pagesIn } from '$lib/content';
    import { TRACKS } from '$lib/site';

    let { open = $bindable(false) }: { open?: boolean } = $props();

    const here = $derived(`${page.params.track}/${page.params.slug}`);
</script>

<nav id="sidebar" class="sidebar" class:open aria-label="Documentation">
    {#each TRACKS as track (track.id)}
        <details open>
            <summary>
                <a href={resolve('/[track]', { track: track.id })}>{track.title}</a>
            </summary>
            <ol>
                {#each pagesIn(page.data.pages, track.id) as p (p.path)}
                    <li>
                        <a
                            href={resolve('/[track]/[slug]', { track: p.track, slug: p.slug })}
                            aria-current={here === p.path ? 'page' : undefined}
                            onclick={() => (open = false)}>{p.title}</a
                        >
                    </li>
                {/each}
            </ol>
        </details>
    {/each}
</nav>

{#if open}
    <button
        class="sidebar-scrim"
        type="button"
        aria-label="Close navigation"
        onclick={() => (open = false)}
    ></button>
{/if}
