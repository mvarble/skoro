<!--
    Two diagrams for the pose graph's guide page, drawn in the site's own colours so that they follow
    its theme. `kind="rows"` is the batch of rows: a matrix whose rows are the datasets and whose
    columns are the model's quantities. `kind="jacobians"` is one row's problem: the sparsity of its
    Jacobian, whose rows are residual components and whose columns are tangent coordinates, for the
    batch over the whole trajectory beside a sliding window over its last poses.
-->
<script lang="ts">
    let { kind }: { kind: 'rows' | 'jacobians' } = $props();

    // ---- The batch of rows ----------------------------------------------------------------------

    type Column = { label: string; hue: 'pose' | 'landmark' | 'gap' };
    const quantities: Column[] = [
        { label: 'x₀', hue: 'pose' },
        { label: 'x₁', hue: 'pose' },
        { label: 'x₂', hue: 'pose' },
        { label: '⋯', hue: 'gap' },
        { label: 'x₇₂', hue: 'pose' },
        { label: 'm₀', hue: 'landmark' },
        { label: 'm₁', hue: 'landmark' },
        { label: '⋯', hue: 'gap' },
        { label: 'm₂₃', hue: 'landmark' },
    ];
    const datasets = ['0', '1', '2', '⋮', '99'];
    const cell = { w: 48, h: 28, x: 110, y: 74 };

    // ---- One row's problem -----------------------------------------------------------------------

    // A small pose graph: eight poses and four landmarks, the last pose sighting the first landmark
    // again, as the circuit closes its loop.
    type Term = { group: string; involves: string[] };
    const motion = (k: number): Term[] => [
        { group: 'motion and odometry', involves: [`x${k}`, `x${k + 1}`] },
        { group: 'motion and odometry', involves: [`x${k}`, `x${k + 1}`] },
    ];
    const sightings = (pairs: [number, number][]): Term[] =>
        pairs.map(([k, j]) => ({ group: 'sightings', involves: [`x${k}`, `m${j}`] }));
    const landmarkPriors = (js: number[]): Term[] =>
        js.map((j) => ({ group: 'landmark priors', involves: [`m${j}`] }));

    const batch = {
        title: 'The batch: the whole trajectory',
        columns: [
            ...[0, 1, 2, 3, 4, 5, 6, 7].map((k) => `x${k}`),
            ...[0, 1, 2, 3].map((j) => `m${j}`),
        ],
        terms: [
            { group: 'first pose', involves: ['x0'] },
            ...[0, 1, 2, 3, 4, 5, 6].flatMap(motion),
            ...landmarkPriors([0, 1, 2, 3]),
            ...sightings([
                [0, 0],
                [1, 0],
                [1, 1],
                [2, 1],
                [3, 1],
                [3, 2],
                [4, 2],
                [5, 2],
                [5, 3],
                [6, 3],
                [7, 3],
                [7, 0],
            ]),
        ] as Term[],
    };
    const sliding = {
        title: 'A sliding window: its last five poses',
        columns: [...[3, 4, 5, 6, 7].map((k) => `x${k}`), ...[1, 2, 3].map((j) => `m${j}`)],
        terms: [
            { group: 'the past, replaced', involves: ['x3', 'm1'] },
            ...[3, 4, 5, 6].flatMap(motion),
            ...landmarkPriors([1, 2, 3]),
            ...sightings([
                [3, 1],
                [3, 2],
                [4, 2],
                [5, 2],
                [5, 3],
                [6, 3],
                [7, 3],
            ]),
        ] as Term[],
    };

    const size = 16;
    const top = 82;
    /** The columns of each kind, as `[label, first column, columns]`. */
    const groups = (columns: string[]): [string, number, number][] => {
        const poses = columns.filter((c) => c.startsWith('x')).length;
        return [
            ['poses', 0, poses],
            ['landmarks', poses, columns.length - poses],
        ];
    };

    /** The runs of equal groups, as `[group, first row, rows]`. */
    function runs(terms: Term[]): [string, number, number][] {
        const out: [string, number, number][] = [];
        terms.forEach((t, i) => {
            const last = out[out.length - 1];
            if (last && last[0] == t.group) last[2] += 1;
            else out.push([t.group, i, 1]);
        });
        return out;
    }

    const hueOf = (column: string) => (column.startsWith('x') ? 'pose' : 'landmark');
</script>

{#if kind == 'rows'}
    <figure class="matrices">
        <svg
            viewBox="0 0 640 330"
            role="img"
            aria-label="A matrix whose rows are the hundred datasets and whose columns are the model's quantities, 73 poses and 24 landmarks. Row 0 is outlined: one dataset, solved on its own."
        >
            <text
                class="axis"
                x={cell.x + (quantities.length * cell.w) / 2}
                y="22"
                text-anchor="middle">columns: the model's quantities</text
            >
            {#each quantities as q, c (c)}
                <text
                    class="head"
                    x={cell.x + c * cell.w + cell.w / 2}
                    y={cell.y - 12}
                    text-anchor="middle">{q.label}</text
                >
            {/each}
            <text
                class="axis"
                transform="translate(26 {cell.y + (datasets.length * cell.h) / 2}) rotate(-90)"
                text-anchor="middle">rows: datasets</text
            >
            {#each datasets as r, i (i)}
                <text
                    class="head"
                    x={cell.x - 14}
                    y={cell.y + i * cell.h + cell.h / 2 + 5}
                    text-anchor="end">{r}</text
                >
                {#each quantities as q, c (c)}
                    {#if q.hue == 'gap' || r == '⋮'}
                        <text
                            class="muted"
                            x={cell.x + c * cell.w + cell.w / 2}
                            y={cell.y + i * cell.h + cell.h / 2 + 5}
                            text-anchor="middle"
                            >{q.hue == 'gap' && r == '⋮' ? '⋱' : q.hue == 'gap' ? '⋯' : '⋮'}</text
                        >
                    {:else}
                        <rect
                            class={q.hue}
                            x={cell.x + c * cell.w + 3}
                            y={cell.y + i * cell.h + 3}
                            width={cell.w - 6}
                            height={cell.h - 6}
                            rx="4"
                        />
                    {/if}
                {/each}
            {/each}
            <rect
                class="selected"
                x={cell.x - 2}
                y={cell.y}
                width={quantities.length * cell.w + 4}
                height={cell.h}
                rx="7"
            />
            <text class="note" x={cell.x} y={cell.y + datasets.length * cell.h + 34}
                >Row 0 is one dataset: its own truth, measurements, start and estimate.</text
            >
            <text class="note" x={cell.x} y={cell.y + datasets.length * cell.h + 56}
                >Each column holds one value per row: a pose's 7 numbers, a landmark's 3.</text
            >
            <text class="note" x={cell.x} y={cell.y + datasets.length * cell.h + 78}
                >The rows share programs and passes, and nothing else.</text
            >
        </svg>
    </figure>
{:else}
    <figure class="matrices">
        <svg
            viewBox="0 0 720 {top + batch.terms.length * size + 70}"
            role="img"
            aria-label="Two sparsity patterns. Left, the batch: rows for every term of a small pose graph, columns for every pose and landmark, including the sighting that closes the loop. Right, a sliding window: columns for its last five poses and the landmarks they see, and a dense replacement row for everything retired."
        >
            {#each [{ m: batch, x: 170 }, { m: sliding, x: 520 }] as { m, x } (m.title)}
                <text
                    class="title"
                    x={x + (m.columns.length * size) / 2}
                    y="18"
                    text-anchor="middle">{m.title}</text
                >
                {#each groups(m.columns) as [label, first, count] (label)}
                    <text
                        class={label == 'poses' ? 'tiny pose-text' : 'tiny landmark-text'}
                        x={x + (first + count / 2) * size}
                        y="46"
                        text-anchor="middle">{label}</text
                    >
                    <path
                        class={label == 'poses' ? 'pose-rule' : 'landmark-rule'}
                        d="M{x + first * size + 2} 54h{count * size - 4}"
                    />
                {/each}
                {#each m.columns as c, i (c)}
                    <text class="tiny" x={x + i * size + size / 2} y="72" text-anchor="middle"
                        >{c.slice(1)}</text
                    >
                {/each}
                {#each runs(m.terms) as [group, first, count] (first)}
                    <path
                        class="bracket"
                        d="M{x - 8} {top + first * size + 2}h-4v{count * size - 4}h4"
                    />
                    <text
                        class="tiny"
                        x={x - 16}
                        y={top + (first + count / 2) * size + 4}
                        text-anchor="end">{group}</text
                    >
                {/each}
                <rect
                    class="frame"
                    x={x - 1}
                    y={top - 1}
                    width={m.columns.length * size + 2}
                    height={m.terms.length * size + 2}
                    rx="3"
                />
                {#each m.terms as t, r (r)}
                    {#each m.columns as c, i (c)}
                        {#if t.involves.includes(c)}
                            <rect
                                class={t.group == 'the past, replaced' ? 'replacement' : hueOf(c)}
                                x={x + i * size + 1.5}
                                y={top + r * size + 1.5}
                                width={size - 3}
                                height={size - 3}
                                rx="2.5"
                            />
                        {/if}
                    {/each}
                {/each}
            {/each}
            <text class="axis" x="40" y={top + batch.terms.length * size + 34}
                >rows: residual components, a block per term</text
            >
            <text class="axis" x="40" y={top + batch.terms.length * size + 56}
                >columns: tangent coordinates, a block per quantity: 6 for a pose, 3 for a landmark</text
            >
        </svg>
    </figure>
{/if}

<style>
    .matrices {
        margin: var(--space-6) 0;
        padding: var(--space-4);
        border: 1px solid var(--border);
        border-radius: var(--radius-lg);
        background: var(--bg-2);
    }

    svg {
        display: block;
        width: 100%;
        height: auto;
        font: var(--font-small);
    }

    text {
        fill: var(--fg-2);
    }

    .title {
        fill: var(--fg-1);
        font-weight: 600;
    }

    .axis {
        fill: var(--fg-1);
        font-weight: 600;
    }

    .head {
        fill: var(--fg-2);
    }

    .muted,
    .note {
        fill: var(--fg-3);
    }

    .tiny {
        font-size: 12px;
        fill: var(--fg-3);
    }

    .pose-text {
        fill: var(--graph-blue);
        font-weight: 600;
    }

    .landmark-text {
        fill: var(--graph-green);
        font-weight: 600;
    }

    .pose-rule,
    .landmark-rule {
        fill: none;
        stroke-width: 2;
        stroke-linecap: round;
    }

    .pose-rule {
        stroke: var(--graph-blue);
    }

    .landmark-rule {
        stroke: var(--graph-green);
    }

    .pose {
        fill: color-mix(in srgb, var(--graph-blue) 70%, transparent);
    }

    .landmark {
        fill: color-mix(in srgb, var(--graph-green) 70%, transparent);
    }

    .replacement {
        fill: color-mix(in srgb, var(--graph-purple) 75%, transparent);
    }

    .selected {
        fill: none;
        stroke: var(--fg-accent);
        stroke-width: 2.5;
    }

    .frame {
        fill: var(--bg-1);
        stroke: var(--border-strong);
    }

    .bracket {
        fill: none;
        stroke: var(--fg-4);
    }
</style>
