<script lang="ts">
    import type { DocumentLink } from '$lib/documents';
    import { STATUS_LABELS } from '$lib/site';
    import { renderMixed } from '$lib/tex';

    // Links to documents: a list with summaries, a row of chips (`inline`), or a grid of cards.
    let {
        links,
        inline = false,
        cards = false,
    }: { links: DocumentLink[]; inline?: boolean; cards?: boolean } = $props();
</script>

{#if inline}
    <span class="chips">
        {#each links as link (link.key)}
            <a class="chip" href={link.href}>{@html renderMixed(link.title)}</a>
        {/each}
    </span>
{:else if cards}
    <div class="card-grid">
        {#each links as link (link.key)}
            <a class="card" href={link.href}>
                <span class="card-title">{@html renderMixed(link.title)}</span>
                {#if link.summary}<span class="card-body">{@html renderMixed(link.summary)}</span
                    >{/if}
            </a>
        {/each}
    </div>
{:else}
    <ul class="link-list">
        {#each links as link (link.key)}
            <li>
                <a href={link.href}>{@html renderMixed(link.title)}</a>
                {#if link.status}
                    <span class="chip status-{link.status}">{STATUS_LABELS[link.status]}</span>
                {/if}
                {#if link.summary}<span class="summary">{@html renderMixed(link.summary)}</span
                    >{/if}
            </li>
        {/each}
    </ul>
{/if}
