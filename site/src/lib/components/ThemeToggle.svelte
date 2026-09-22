<script lang="ts">
    import { theme } from '$lib/state/theme.svelte';
    import Icon from './Icon.svelte';

    $effect(() => {
        document.documentElement.classList.toggle('dark', theme.current === 'dark');
    });
</script>

<button
    class="icon-button"
    type="button"
    aria-label="Switch to {theme.current === 'dark' ? 'light' : 'dark'} theme"
    onclick={() => (theme.current = theme.current === 'dark' ? 'light' : 'dark')}
>
    <Icon name={theme.current === 'dark' ? 'sun' : 'moon'} />
</button>
