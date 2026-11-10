<!--
    One of the examples' figures, drawn by `bin/figures` from the data the examples write and kept in
    `static/figures/`. The figure is a self-contained card with its own surface, so it reads the
    same in both themes; the caption says what it shows in words, for a reader who cannot see it.
-->
<script lang="ts">
    import { asset } from '$app/paths';

    let { name, alt, caption }: { name: string; alt: string; caption?: string } = $props();
</script>

<figure class="example-figure">
    <img src={asset(`/figures/${name}.svg`)} {alt} loading="lazy" decoding="async" />
    {#if caption}<figcaption>{caption}</figcaption>{/if}
</figure>

<style>
    .example-figure {
        margin: var(--space-6, 2rem) 0;
    }

    img {
        display: block;
        width: 100%;
        height: auto;
        border-radius: 18px;
        box-shadow: 0 12px 32px rgb(0 0 0 / 0.18);
    }

    figcaption {
        margin-top: var(--space-3);
        font: var(--font-small);
        color: var(--fg-3);
        text-align: center;
    }
</style>
