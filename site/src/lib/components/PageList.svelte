<script lang="ts">
    import type { DocumentLink } from '$lib/documents';
    import { STATUS_LABELS } from '$lib/site';
    import { renderMixed } from '$lib/tex';

    // The documents of a section, one row each.
    let {
        links,
        numbered = false,
        empty,
    }: { links: DocumentLink[]; numbered?: boolean; empty: string } = $props();
</script>

{#if links.length}
    <ol class="page-list" class:numbered>
        {#each links as link, i (link.key)}
            <li>
                <a href={link.href}>
                    {#if numbered}<span class="index">{String(i + 1).padStart(2, '0')}</span>{/if}
                    <span class="title"
                        >{@html renderMixed(link.title)}{#if link.status}
                            <span class="chip status-{link.status}"
                                >{STATUS_LABELS[link.status]}</span
                            >{/if}</span
                    >
                    <span class="summary">{@html renderMixed(link.summary)}</span>
                </a>
            </li>
        {/each}
    </ol>
{:else}
    <p class="muted">{empty}</p>
{/if}
