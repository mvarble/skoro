<script lang="ts">
    import { layoutReadingOrder } from '$lib/reading/layout';
    import { itemText, type ReadingOrderSpec } from '$lib/reading/spec';

    // Draws the order in which to read a crate's modules (see `$lib/reading/spec.ts`): the modules
    // top to bottom, each use of an earlier module as an arc on the left, and each use of a later
    // one as a dashed arc on the right with its items named beside the module. Uses that a longer
    // chain implies are hidden until asked for. Hovering, focusing or tapping a module shows all of
    // its uses and what uses it. All geometry is in `$lib/reading/layout.ts`.

    let { spec }: { spec: ReadingOrderSpec } = $props();

    const layout = $derived(layoutReadingOrder(spec));
    const id = $props.id();

    let showImplied = $state(false);
    let hovered = $state<string | null>(null);
    let pinned = $state<string | null>(null);
    const focused = $derived(pinned ?? hovered);

    const at = $derived(new Map(spec.modules.map((m, i) => [m.id, i])));
    const byOrder = (ids: string[]) => [...ids].sort((a, b) => at.get(a)! - at.get(b)!);
    const uses = (m: string) => byOrder(spec.uses.filter((u) => u.from == m).map((u) => u.to));
    const users = (m: string) => byOrder(spec.uses.filter((u) => u.to == m).map((u) => u.from));
    const related = $derived(focused ? new Set([...uses(focused), ...users(focused)]) : new Set());

    const ahead = $derived(
        spec.uses
            .filter((u) => at.get(u.to)! > at.get(u.from)!)
            .sort((a, b) => at.get(a.from)! - at.get(b.from)! || at.get(a.to)! - at.get(b.to)!),
    );
    const cycleSizes = $derived([...new Set(layout.bands.map((b) => b.size))]);

    const toggle = (m: string) => (pinned = pinned == m ? null : m);
    const onKey = (event: KeyboardEvent, m: string) => {
        if (event.key == 'Enter' || event.key == ' ') {
            event.preventDefault();
            toggle(m);
        }
    };
</script>

<figure class="reading-order">
    <div class="reading-controls">
        <span class="reading-key"
            ><svg viewBox="0 0 30 10" aria-hidden="true"
                ><line x1="0" y1="5" x2="30" y2="5" class="key-earlier" /></svg
            >uses an earlier module</span
        >
        <span class="reading-key"
            ><svg viewBox="0 0 30 10" aria-hidden="true"
                ><line x1="0" y1="5" x2="30" y2="5" class="key-later" /></svg
            >uses a later module</span
        >
        {#if layout.bands.length}
            <span class="reading-key"
                ><svg viewBox="0 0 30 10" aria-hidden="true"
                    ><rect width="30" height="10" rx="2" class="reading-band" /></svg
                >a cycle ({cycleSizes.join(', ')} modules)</span
            >
        {/if}
        <label class="reading-key">
            <input type="checkbox" bind:checked={showImplied} />
            Show the uses a longer chain implies
        </label>
    </div>

    <div class="reading-canvas">
        <svg
            viewBox="0 0 {layout.width} {layout.height}"
            style:max-width="{layout.width}px"
            role="img"
            aria-labelledby="{id}-title"
            class:focusing={focused}
            class:show-implied={showImplied}
        >
            <title id="{id}-title">{spec.title}</title>
            <defs>
                <marker
                    id="{id}-earlier"
                    viewBox="0 0 8 8"
                    refX="7"
                    refY="4"
                    markerWidth="7"
                    markerHeight="7"
                    orient="auto"
                >
                    <path d="M0 0L8 4L0 8z" class="head-earlier" />
                </marker>
                <marker
                    id="{id}-later"
                    viewBox="0 0 8 8"
                    refX="7"
                    refY="4"
                    markerWidth="7"
                    markerHeight="7"
                    orient="auto"
                >
                    <path d="M0 0L8 4L0 8z" class="head-later" />
                </marker>
            </defs>

            {#each layout.bands as band, i (i)}
                <rect
                    class="reading-band"
                    x={layout.dotX + 4}
                    y={band.y}
                    width={layout.laterX - layout.dotX - 10}
                    height={band.height}
                    rx="6"
                />
            {/each}

            {#each layout.uses as use (`${use.from}\n${use.to}`)}
                <path
                    class="reading-use"
                    class:later={use.later}
                    class:implied={use.implied}
                    class:on={focused == use.from || focused == use.to}
                    d={use.d}
                    marker-end="url(#{id}-{use.later ? 'later' : 'earlier'})"
                    ><title>{use.from} uses {use.to}: {use.items.map(itemText).join(', ')}</title
                    ></path
                >
            {/each}

            {#each layout.rows as row (row.id)}
                <g
                    class="reading-row"
                    class:on={focused == row.id}
                    class:related={related.has(row.id)}
                    role="button"
                    tabindex="0"
                    aria-pressed={pinned == row.id}
                    aria-label="{row.number}. {row.id}"
                    onmouseenter={() => (hovered = row.id)}
                    onmouseleave={() => (hovered = null)}
                    onfocus={() => (hovered = row.id)}
                    onblur={() => (hovered = null)}
                    onclick={() => toggle(row.id)}
                    onkeydown={(event) => onKey(event, row.id)}
                >
                    <rect
                        class="row-hit"
                        x={layout.dotX + 4}
                        y={row.y - 12}
                        width={layout.width - layout.dotX - 8}
                        height="24"
                        rx="4"
                    />
                    <circle class="row-dot" cx={layout.dotX + 3} cy={row.y} r="3" />
                    <text class="row-number" x={layout.nameX - 10} y={row.y + 4}>{row.number}</text>
                    <text class="row-name" x={layout.nameX} y={row.y + 4}>{row.id}</text>
                    {#if row.later}
                        <text class="row-later" x={layout.itemX} y={row.y + 4}>{row.later}</text>
                    {/if}
                </g>
            {/each}
        </svg>
    </div>

    <p class="reading-info" aria-live="polite">
        {#if focused}
            <code>{focused}</code> uses
            {#each uses(focused) as m, i (m)}{i ? ', ' : ''}<code>{m}</code>{:else}nothing here{/each}.
            It is used by
            {#each users(focused) as m, i (m)}{i ? ', ' : ''}<code>{m}</code>{:else}nothing here{/each}.
        {:else}
            Hover over or tap a module to see everything it uses and everything that uses it.
        {/if}
    </p>

    {#if spec.caption}<figcaption>{spec.caption}</figcaption>{/if}

    {#if ahead.length}
        <table class="reading-ahead">
            <thead>
                <tr><th>Module</th><th>Uses, before it is read</th><th>From</th></tr>
            </thead>
            <tbody>
                {#each ahead as use (`${use.from}\n${use.to}`)}
                    <tr>
                        <td><code>{use.from}</code></td>
                        <td
                            >{#each use.items as item, i (item)}{i ? ', ' : ''}<code
                                    >{itemText(item)}</code
                                >{/each}</td
                        >
                        <td><code>{use.to}</code></td>
                    </tr>
                {/each}
            </tbody>
        </table>
    {/if}
</figure>
