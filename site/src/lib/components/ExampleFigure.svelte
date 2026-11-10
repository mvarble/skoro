<!--
    One of the examples' figures, drawn by `bin/figures` from the data the examples write and kept in
    `static/figures/`. The figure is a self-contained card with its own surface, so it reads the
    same in both themes; the caption says what it shows in words, for a reader who cannot see it.

    The figure is a button: clicked, or pressed with Enter or Space, it opens the figure at the
    size of the window in a modal dialog, which Escape, the close button or a click outside it
    closes. Hovering or focusing it lifts it and shows the expand mark, so that a reader sees it
    can be opened before trying.
-->
<script lang="ts">
    import { asset } from '$app/paths';
    import Icon from './Icon.svelte';

    let { name, alt, caption }: { name: string; alt: string; caption?: string } = $props();

    const src = $derived(asset(`/figures/${name}.svg`));
    let dialog: HTMLDialogElement | undefined = $state();
    // The figure's width over its height, read when it loads, so that the enlarged figure is as
    // large as the window allows without being letterboxed.
    let aspect = $state(1200 / 760);

    function measure(event: Event) {
        const img = event.currentTarget as HTMLImageElement;
        if (img.naturalWidth && img.naturalHeight) aspect = img.naturalWidth / img.naturalHeight;
    }

    function open() {
        dialog?.showModal();
    }

    // A click on the backdrop lands on the dialog itself, outside its frame.
    function closeOnBackdrop(event: MouseEvent) {
        if (event.target == dialog) dialog?.close();
    }
</script>

<figure class="example-figure">
    <button
        type="button"
        class="opener"
        onclick={open}
        aria-haspopup="dialog"
        aria-label={`Enlarge the figure: ${alt}`}
    >
        <img {src} alt="" loading="lazy" decoding="async" onload={measure} />
        <span class="badge" aria-hidden="true"><Icon name="expand" size={16} /> Enlarge</span>
    </button>
    {#if caption}<figcaption>{caption}</figcaption>{/if}
</figure>

<dialog bind:this={dialog} class="figure-dialog" aria-label={alt} onclick={closeOnBackdrop}>
    <div class="frame" style:--aspect={aspect}>
        <img {src} {alt} />
        {#if caption}<p class="dialog-caption">{caption}</p>{/if}
        <form method="dialog">
            <button class="close" aria-label="Close"><Icon name="close" size={20} /></button>
        </form>
    </div>
</dialog>

<style>
    .example-figure {
        margin: var(--space-6) 0;
    }

    .opener {
        position: relative;
        display: block;
        width: 100%;
        padding: 0;
        border: 0;
        border-radius: 18px;
        background: none;
        cursor: zoom-in;
        box-shadow: 0 8px 24px rgb(0 0 0 / 0.08);
        transition:
            transform 160ms ease,
            box-shadow 160ms ease;
    }

    .opener img {
        display: block;
        width: 100%;
        height: auto;
        border-radius: 18px;
    }

    .badge {
        position: absolute;
        top: 0.75rem;
        right: 0.75rem;
        display: inline-flex;
        align-items: center;
        gap: 0.35rem;
        padding: 0.3rem 0.6rem;
        border-radius: 999px;
        background: var(--bg-1);
        color: var(--fg-2);
        font: var(--font-small);
        box-shadow: var(--shadow-sm);
        border: 1px solid var(--border);
        opacity: 0;
        transform: translateY(-2px);
        transition:
            opacity 160ms ease,
            transform 160ms ease;
    }

    .opener:hover,
    .opener:focus-visible {
        transform: translateY(-3px);
        box-shadow:
            0 0 0 2px var(--fg-accent),
            0 16px 36px rgb(0 0 0 / 0.14);
    }

    .opener:focus-visible {
        outline: none;
    }

    .opener:hover .badge,
    .opener:focus-visible .badge {
        opacity: 1;
        transform: none;
    }

    @media (prefers-reduced-motion: reduce) {
        .opener,
        .badge {
            transition: none;
        }

        .opener:hover,
        .opener:focus-visible {
            transform: none;
        }
    }

    figcaption {
        margin-top: var(--space-3);
        font: var(--font-small);
        color: var(--fg-3);
        text-align: center;
    }

    .figure-dialog {
        /* Centred in the window: the site's reset takes the user agent's automatic margins away. */
        margin: auto;
        max-width: min(96vw, 1600px);
        max-height: 96vh;
        padding: 0;
        border: 0;
        background: transparent;
        overflow: visible;
    }

    .figure-dialog::backdrop {
        background: rgb(10 12 20 / 0.72);
        backdrop-filter: blur(3px);
    }

    .frame {
        position: relative;
    }

    .frame img {
        display: block;
        width: min(96vw, 1600px, calc(86vh * var(--aspect)));
        height: auto;
        border-radius: 18px;
        box-shadow: 0 24px 64px rgb(0 0 0 / 0.35);
    }

    .dialog-caption {
        margin: var(--space-3) auto 0;
        max-width: 60rem;
        text-align: center;
        font: var(--font-small);
        color: hsl(0 0% 92%);
    }

    form {
        position: absolute;
        top: 0.75rem;
        right: 0.75rem;
        margin: 0;
    }

    .close {
        display: grid;
        place-items: center;
        width: 2.25rem;
        height: 2.25rem;
        border-radius: 999px;
        border: 1px solid var(--border);
        background: var(--bg-1);
        color: var(--fg-1);
        cursor: pointer;
        box-shadow: var(--shadow-sm);
    }

    .close:hover,
    .close:focus-visible {
        border-color: var(--fg-accent);
        color: var(--fg-accent);
    }
</style>
