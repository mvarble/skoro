<script lang="ts">
    import { resolve } from '$app/paths';
    import { page } from '$app/state';
    import { SITE, TRACKS } from '$lib/site';
    import Icon from './Icon.svelte';
    import Logo from './Logo.svelte';
    import ThemeToggle from './ThemeToggle.svelte';

    let {
        menuOpen = $bindable(false),
        showMenu = true,
    }: { menuOpen?: boolean; showMenu?: boolean } = $props();

    const current = $derived(page.params.track);
</script>

<header class="topbar">
    {#if showMenu}
        <button
            class="icon-button menu-button"
            type="button"
            aria-label={menuOpen ? 'Close navigation' : 'Open navigation'}
            aria-expanded={menuOpen}
            aria-controls="sidebar"
            onclick={() => (menuOpen = !menuOpen)}
        >
            <Icon name={menuOpen ? 'close' : 'menu'} />
        </button>
    {/if}

    <a class="wordmark" href={resolve('/')}>
        <Logo variant="icon" />
        <span>{SITE.name}</span>
        <span class="version">v{SITE.version}</span>
    </a>

    <nav class="track-nav" aria-label="Sections">
        {#each TRACKS.filter((t) => t.id != 'start') as track (track.id)}
            <a
                href={resolve('/[track]', { track: track.id })}
                aria-current={current === track.id ? 'page' : undefined}>{track.short}</a
            >
        {/each}
    </nav>

    <div class="topbar-actions">
        <a class="icon-button" href={SITE.repo} rel="external" aria-label="Source on GitHub">
            <Icon name="github" />
        </a>
        <ThemeToggle />
    </div>
</header>
