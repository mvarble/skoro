import { defineGraph } from '$lib/graph/spec';

// The core's running example, a vehicle at x₀ then x₁ measuring a landmark m from each, queried
// with the measurements, the IMU reading ζ₁ and the scale σ observed, so that the free continuous
// quantities are x₀, x₁ and m. Its factor structure, and the elimination tree of the order
// x₀, x₁, m.

export const factors = defineGraph({
    title: 'The factor structure of the example',
    caption: String.raw`A query of the core's example free in $\randomPoint_0$, $\randomPoint_1$ and $\bfm$, with $\bfz_0$, $\bfz_1$, $\bfzeta_1$ and $\bfsigma$ observed. Each term reached from a free quantity is a factor, joined to the free quantities among those it involves; the observed quantities are not variables, and the score $t_7$ of $\bfsigma$, reached from no free quantity, is a constant of the query and no factor.`,
    width: 640,
    height: 300,
    nodeKinds: {
        variable: { name: 'Free quantity', hue: 'orange' },
        factor: { name: 'Factor', hue: 'purple', shape: 'square' },
    },
    edgeKinds: {
        touches: { name: 'has a block for' },
    },
    nodes: [
        { id: 't1', x: 60, y: 150, kind: 'factor', label: String.raw`$t_1$` },
        { id: 'x0', x: 180, y: 150, kind: 'variable', label: String.raw`$\randomPoint_0$` },
        { id: 't3', x: 320, y: 80, kind: 'factor', label: String.raw`$t_3$` },
        { id: 't6', x: 320, y: 20, kind: 'factor', label: String.raw`$t_6$` },
        { id: 't4', x: 320, y: 220, kind: 'factor', label: String.raw`$t_4$` },
        { id: 'x1', x: 460, y: 80, kind: 'variable', label: String.raw`$\randomPoint_1$` },
        { id: 'm', x: 460, y: 220, kind: 'variable', label: String.raw`$\bfm$` },
        { id: 't5', x: 580, y: 150, kind: 'factor', label: String.raw`$t_5$` },
        { id: 't2', x: 580, y: 270, kind: 'factor', label: String.raw`$t_2$` },
    ],
    edges: [
        { from: 't1', to: 'x0', kind: 'touches', arrow: 'none' },
        { from: 't3', to: 'x0', kind: 'touches', arrow: 'none' },
        { from: 't3', to: 'x1', kind: 'touches', arrow: 'none' },
        { from: 't6', to: 'x0', kind: 'touches', arrow: 'none' },
        { from: 't6', to: 'x1', kind: 'touches', arrow: 'none' },
        { from: 't4', to: 'x0', kind: 'touches', arrow: 'none' },
        { from: 't4', to: 'm', kind: 'touches', arrow: 'none' },
        { from: 't5', to: 'x1', kind: 'touches', arrow: 'none' },
        { from: 't5', to: 'm', kind: 'touches', arrow: 'none' },
        { from: 't2', to: 'm', kind: 'touches', arrow: 'none' },
    ],
});

export const tree = defineGraph({
    title: 'The elimination tree of the example',
    caption: String.raw`The order $\randomPoint_0, \randomPoint_1, \bfm$. Eliminating $\randomPoint_0$ collects $t_1$, $t_3$, $t_4$ and $t_6$ and leaves a new factor on its separator $\{\randomPoint_1, \bfm\}$; its parent is $\randomPoint_1$, the first of them eliminated after it. Eliminating $\randomPoint_1$ collects $t_5$ and that factor, with the separator $\{\bfm\}$. An arrow runs from a parent to its child, so the root $\bfm$, eliminated last, has no incoming arrow.`,
    width: 640,
    height: 140,
    nodeKinds: {
        variable: { name: 'Free quantity', hue: 'orange' },
    },
    edgeKinds: {
        parent: { name: 'parent of' },
    },
    nodes: [
        { id: 'm', x: 100, y: 70, kind: 'variable', label: String.raw`$\bfm$`, annotation: 'root', annotationAngle: 90 },
        { id: 'x1', x: 320, y: 70, kind: 'variable', label: String.raw`$\randomPoint_1$` },
        { id: 'x0', x: 540, y: 70, kind: 'variable', label: String.raw`$\randomPoint_0$` },
    ],
    edges: [
        { from: 'm', to: 'x1', kind: 'parent' },
        { from: 'x1', to: 'x0', kind: 'parent' },
    ],
});
