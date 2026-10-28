<script lang="ts">
    import { onMount, untrack, type Snippet } from 'svelte';

    import Icon from '$lib/components/Icon.svelte';
    import { layoutGraph, type TextLayout } from '$lib/graph/geometry';
    import { edgeKindClasses, nodeKindClasses, type NodeSpec } from '$lib/graph/spec';
    import { renderMixed } from '$lib/tex';
    import { frameAt, kindsUsed } from '$lib/walk/frames';
    import { codeMatches, edgeId, matchLines, type WalkthroughSpec } from '$lib/walk/spec';

    // A graph figure built up step by step (see `$lib/walk/spec.ts`) above the code that builds
    // it: the page's own file-reference fence, passed as children, whose lines each step points at.
    // The drawing follows `<ModelGraph>`: geometry from `$lib/graph/geometry.ts`, text in an HTML
    // layer over the SVG, the same palette. Every node, edge and region is always in the SVG, and a
    // step only changes classes on it, so CSS transitions animate what appears, fades or changes
    // kind.
    //
    // The prerendered page shows the last step and lists every step's note, which is the whole
    // walkthrough without JavaScript. Once mounted, it starts at the first step.

    let { spec, children }: { spec: WalkthroughSpec; children?: Snippet } = $props();

    const id = $props.id();
    const last = $derived(spec.steps.length - 1);
    let step = $state(untrack(() => spec.steps.length - 1));
    let playing = $state(false);
    let mounted = $state(false);
    // Off for the first frame after mounting, so the jump to the first step is not animated.
    let animate = $state(false);
    let stepsOpen = $state(true);
    let code: HTMLElement | undefined = $state();

    const frame = $derived(frameAt(spec, step));
    const layout = $derived(layoutGraph(frame.graph));
    const ids = $derived(spec.edges.map(edgeId));

    const text = (source: string) => renderMixed(source, spec.katexMacros);
    // A note: TeX between dollars, as `renderMixed` does, and `code` between backticks.
    const note = (source: string) =>
        source
            .split(/(`[^`]*`)/)
            .map((part) =>
                /^`[^`]*`$/.test(part)
                    ? `<code>${renderMixed(part.slice(1, -1).replace(/\$/g, '\\$'))}</code>`
                    : text(part),
            )
            .join('');

    // The classes a node, edge or region gets from the frame.
    const flags = (key: string) => ({
        'wt-hidden': !frame.visible.has(key),
        'graph-muted': frame.muted.has(key),
        'wt-emphasis': frame.emphasized.has(key),
    });

    const edgeClasses = (kind: string | undefined) =>
        edgeKindClasses(kind === undefined ? undefined : spec.edgeKinds?.[kind]);

    const nodeClass = (node: NodeSpec) =>
        ['graph-node', nodeKindClasses(frame.graph.nodeKinds[node.kind]!), node.class]
            .filter(Boolean)
            .join(' ');

    const SHIFT_X = { start: '0%', middle: '-50%', end: '-100%' };
    const SHIFT_Y = { hanging: '0%', middle: '-50%', auto: '-100%' };
    const place = (t: TextLayout) =>
        `left: ${(t.x / layout.width) * 100}%; top: ${(t.y / layout.height) * 100}%; ` +
        `transform: translate(${SHIFT_X[t.anchor]}, ${SHIFT_Y[t.baseline]});`;
    const centered = (x: number, y: number) =>
        place({ x, y, anchor: 'middle', baseline: 'middle' });

    // Regions in SVG coordinates, +Y down.
    const regions = $derived(
        (spec.regions ?? []).map((region) => ({
            spec: region,
            x: region.x,
            y: spec.height - region.y - region.height,
        })),
    );

    const legend = $derived.by(() => {
        const used = kindsUsed(spec);
        return {
            nodes: Object.entries(spec.nodeKinds).filter(([key]) => used.nodes.has(key)),
            edges: Object.entries(spec.edgeKinds ?? {}).filter(([key]) => used.edges.has(key)),
        };
    });

    const go = (to: number) => {
        step = Math.min(Math.max(to, 0), last);
    };

    const togglePlay = () => {
        if (!playing && step == last) step = 0;
        playing = !playing;
    };

    // Long enough to read the note.
    const dwell = (i: number) => Math.max(2600, spec.steps[i]!.note.length * 45);

    $effect(() => {
        if (!playing) return;
        if (step >= last) {
            playing = false;
            return;
        }
        const timer = setTimeout(() => step++, dwell(step));
        return () => clearTimeout(timer);
    });

    const keydown = (event: KeyboardEvent) => {
        if (event.target instanceof HTMLInputElement) return;
        const moves: Record<string, () => void> = {
            ArrowLeft: () => go(step - 1),
            ArrowRight: () => go(step + 1),
            Home: () => go(0),
            End: () => go(last),
        };
        const move = moves[event.key];
        if (!move) return;
        event.preventDefault();
        playing = false;
        move();
    };

    // The quoted code's lines, and the lines each step points at.
    let lines: HTMLElement[] = $state([]);
    const pointed = $derived.by(() => {
        const texts = lines.map((line) => line.textContent ?? '');
        return spec.steps.map(
            (s) => new Set(codeMatches(s).flatMap((match) => matchLines(texts, match))),
        );
    });

    onMount(() => {
        lines = [...(code?.querySelectorAll<HTMLElement>('.line') ?? [])];
        step = 0;
        stepsOpen = false;
        mounted = true;
        requestAnimationFrame(() => requestAnimationFrame(() => (animate = true)));
    });

    $effect(() => {
        const current = pointed[step];
        if (!current) return;
        lines.forEach((line, i) => line.classList.toggle('wt-line', current.has(i)));
        const first = lines.find((_, i) => current.has(i));
        const scroller = code?.querySelector('pre');
        if (!first || !scroller) return;
        const top = first.offsetTop - scroller.clientHeight / 3;
        const smooth = animate && !matchMedia('(prefers-reduced-motion: reduce)').matches;
        scroller.scrollTo({ top, behavior: smooth ? 'smooth' : 'auto' });
    });
</script>

<figure class="model-graph walkthrough" class:wt-animate={animate} aria-labelledby="{id}-caption">
    <div
        class="graph-canvas"
        style:max-width="{layout.width}px"
        style:--graph-width={layout.width}
        style:--graph-font={layout.fontSize}
    >
        <svg viewBox="0 0 {layout.width} {layout.height}" role="img" aria-labelledby="{id}-title">
            <title id="{id}-title">{spec.title}</title>

            {#each regions as region (region.spec.id)}
                <rect
                    class={[
                        'wt-region',
                        `hue-${region.spec.hue ?? 'gray'}`,
                        `line-${region.spec.line ?? 'solid'}`,
                        flags(region.spec.id),
                    ]}
                    x={region.x}
                    y={region.y}
                    width={region.spec.width}
                    height={region.spec.height}
                    rx="8"
                />
            {/each}

            {#each layout.edges as edge, i (ids[i])}
                <g
                    class={[
                        'graph-edge',
                        edgeClasses(edge.spec.kind),
                        edge.spec.class,
                        flags(ids[i]!),
                        frame.flow.has(ids[i]!) && 'wt-flow',
                        frame.flowBack.has(ids[i]!) && 'wt-flow-back',
                    ]}
                    style={edge.spec.style}
                >
                    <path d={edge.d} />
                    {#each edge.arrows as points, j (j)}
                        <polygon {points} />
                    {/each}
                </g>
            {/each}

            {#each layout.nodes as node (node.spec.id)}
                <g
                    class={[nodeClass(node.spec), flags(node.spec.id)]}
                    style="transform-origin: {node.cx}px {node.cy}px; {node.spec.style ?? ''}"
                >
                    {#if node.spec.title}<title>{node.spec.title}</title>{/if}
                    {#if node.shape == 'square'}
                        <rect
                            x={node.cx - node.r}
                            y={node.cy - node.r}
                            width={2 * node.r}
                            height={2 * node.r}
                            rx="3"
                        />
                    {:else}
                        <circle cx={node.cx} cy={node.cy} r={node.r} />
                    {/if}
                </g>
            {/each}
        </svg>

        <div class="graph-labels">
            {#each regions as region (region.spec.id)}
                {@const label = frame.regionLabels.get(region.spec.id)}
                {#if label}
                    <span
                        class={[
                            'wt-region-label',
                            `hue-${region.spec.hue ?? 'gray'}`,
                            flags(region.spec.id),
                        ]}
                        style={place({
                            x: region.x + 8,
                            y: region.y + 6,
                            anchor: 'start',
                            baseline: 'hanging',
                        })}>{@html text(label)}</span
                    >
                {/if}
            {/each}
            {#each layout.edges as edge, i (ids[i])}
                {#if edge.label}
                    <span
                        class={['edge-label', edgeClasses(edge.spec.kind), flags(ids[i]!)]}
                        style={centered(edge.label.x, edge.label.y)}
                        >{@html text(edge.spec.label!)}</span
                    >
                {/if}
            {/each}
            {#each layout.nodes as node (node.spec.id)}
                {#if node.spec.label}
                    <span
                        class={['node-label', flags(node.spec.id)]}
                        style={centered(node.cx, node.cy)}>{@html text(node.spec.label)}</span
                    >
                {/if}
                {#if node.annotation}
                    <span
                        class={['node-annotation', flags(node.spec.id)]}
                        style={place(node.annotation)}>{@html text(node.spec.annotation!)}</span
                    >
                {/if}
            {/each}
        </div>
    </div>

    {#if spec.legend ?? true}
        <ul class="graph-legend" aria-label="Legend">
            {#each legend.nodes as [key, kind] (key)}
                <li>
                    <svg viewBox="-9 -9 18 18" width="14" height="14" aria-hidden="true">
                        <g class="graph-node {nodeKindClasses(kind)}">
                            {#if kind.shape == 'square'}
                                <rect x="-7" y="-7" width="14" height="14" rx="2" />
                            {:else}
                                <circle r="7" />
                            {/if}
                        </g>
                    </svg>
                    <span>{@html text(kind.name)}</span>
                </li>
            {/each}
            {#each legend.edges as [key, kind] (key)}
                <li>
                    <svg viewBox="0 0 28 10" width="28" height="10" aria-hidden="true">
                        <g class="graph-edge {edgeKindClasses(kind)}">
                            <path d="M 1 5 L 20 5" />
                            <polygon points="27,5 19,1.5 19,8.5" />
                        </g>
                    </svg>
                    <span>{@html text(kind.name)}</span>
                </li>
            {/each}
        </ul>
    {/if}

    {#if mounted}
        <div
            class="wt-controls"
            role="toolbar"
            aria-label="Steps"
            tabindex="-1"
            onkeydown={keydown}
        >
            <button
                class="wt-button"
                onclick={() => ((playing = false), go(0))}
                disabled={step == 0}
            >
                <Icon name="restart" size={16} /><span class="visually-hidden">First step</span>
            </button>
            <button
                class="wt-button"
                onclick={() => ((playing = false), go(step - 1))}
                disabled={step == 0}
            >
                <Icon name="arrowLeft" size={16} /><span class="visually-hidden">Previous step</span
                >
            </button>
            <button class="wt-button wt-play" onclick={togglePlay} aria-pressed={playing}>
                <Icon name={playing ? 'pause' : 'play'} size={16} />
                <span class="visually-hidden">{playing ? 'Pause' : 'Play'}</span>
            </button>
            <button
                class="wt-button"
                onclick={() => ((playing = false), go(step + 1))}
                disabled={step == last}
            >
                <Icon name="arrowRight" size={16} /><span class="visually-hidden">Next step</span>
            </button>
            <input
                type="range"
                min="0"
                max={last}
                bind:value={step}
                oninput={() => (playing = false)}
                aria-label="Step"
                aria-valuetext="Step {step + 1} of {last + 1}"
            />
            <span class="wt-count">{step + 1} / {last + 1}</span>
        </div>
        <p class="wt-note" aria-live="polite">{@html note(spec.steps[step]!.note)}</p>
    {/if}

    <div class="wt-code" bind:this={code}>
        {@render children?.()}
    </div>

    <details class="wt-steps" bind:open={stepsOpen}>
        <summary>Every step</summary>
        <ol>
            {#each spec.steps as s, i (i)}
                <li aria-current={mounted && i == step ? 'step' : undefined}>
                    {#if mounted}
                        <button onclick={() => ((playing = false), go(i))}
                            >{@html note(s.note)}</button
                        >
                    {:else}
                        {@html note(s.note)}
                    {/if}
                </li>
            {/each}
        </ol>
    </details>

    <figcaption id="{id}-caption">
        <strong>{spec.title}.</strong>
        {#if spec.caption}{@html text(spec.caption)}{/if}
    </figcaption>
</figure>
