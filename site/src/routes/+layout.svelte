<script lang="ts">
    import '../styles/index.css';

    import { asset } from '$app/paths';
    import Shell from '@mvarble/mesearch-workflow/app/lib/components/Shell.svelte';

    let { data, children } = $props();
</script>

<Shell origin={data.origin}>
    {#snippet head()}
        <!-- The card a shared link shows, drawn by `logo/generate.py`. -->
        <meta property="og:image" content={data.origin + asset('/og-image.png')} />
        <meta property="og:image:width" content="1200" />
        <meta property="og:image:height" content="630" />
        <meta property="og:image:alt" content="The skoro logo: a stitched eight-pointed star" />
        <meta name="twitter:card" content="summary_large_image" />
    {/snippet}
    {@render children()}
</Shell>
