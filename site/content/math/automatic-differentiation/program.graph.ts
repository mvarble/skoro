import { defineGraph, type EdgeSpec, type NodeKindSpec, type NodeSpec } from '$lib/graph/spec';

// One small program, f = x₁x₂ + exp(x₁x₂), drawn twice: as its computation graph, and as the
// backpropagation graph its reverse pass runs on. The slot s₃ has two consumers, which is where the
// reverse pass accumulates.

const nodeKinds: Record<string, NodeKindSpec> = {
    leaf: { name: 'Leaf (an input)', hue: 'blue' },
    slot: { name: 'Slot written by a step', hue: 'green' },
};

const nodes: NodeSpec[] = [
    { id: 'x1', x: 70, y: 200, kind: 'leaf', label: String.raw`$\point_1$` },
    { id: 'x2', x: 70, y: 70, kind: 'leaf', label: String.raw`$\point_2$` },
    {
        id: 's3',
        x: 250,
        y: 135,
        kind: 'slot',
        label: String.raw`$s_3$`,
        annotation: String.raw`$\map_1 = {\odot}$`,
    },
    {
        id: 's4',
        x: 420,
        y: 215,
        kind: 'slot',
        label: String.raw`$s_4$`,
        annotation: String.raw`$\map_2 = \exp$`,
        annotationAngle: 90,
    },
    {
        id: 'f',
        x: 590,
        y: 135,
        kind: 'slot',
        label: String.raw`$f$`,
        annotation: String.raw`$\map_3 = {+}$`,
    },
];

const forward: EdgeSpec[] = [
    { from: 'x1', to: 's3', kind: 'input' },
    { from: 'x2', to: 's3', kind: 'input' },
    { from: 's3', to: 's4', kind: 'input' },
    { from: 's3', to: 'f', kind: 'input' },
    { from: 's4', to: 'f', kind: 'input' },
];

export const computation = defineGraph({
    title: 'A computation graph',
    caption: String.raw`The program $s_3 \leftarrow \point_1 \odot \point_2$, $s_4 \leftarrow \exp s_3$, $f \leftarrow s_3 + s_4$. Each node is a slot; each edge runs from an input of a step to its output. The leaves $\point_1$ and $\point_2$ are the program's inputs, and $f$ is its target.`,
    width: 660,
    height: 270,
    nodeKinds,
    edgeKinds: { input: { name: 'input of a step' } },
    nodes,
    edges: forward,
});

const labels: Record<string, string> = {
    's3>x1': String.raw`$\partial_1 \map_1\transpose$`,
    's3>x2': String.raw`$\partial_2 \map_1\transpose$`,
    's4>s3': String.raw`$\partial \map_2\transpose$`,
    'f>s3': String.raw`$\partial_1 \map_3\transpose$`,
    'f>s4': String.raw`$\partial_2 \map_3\transpose$`,
};

export const backpropagation = defineGraph({
    title: 'Its backpropagation graph',
    caption: String.raw`The same slots with every edge reversed and labelled by the pull-back it applies. Cotangents enter at the target $f$ and end at the leaves; $s_3$ has two incoming edges, and its cotangent is their sum.`,
    width: 660,
    height: 270,
    nodeKinds,
    edgeKinds: { pullback: { name: 'pull-back', hue: 'purple' } },
    nodes,
    edges: forward.map(({ from, to }) => ({
        from: to,
        to: from,
        kind: 'pullback',
        label: labels[`${to}>${from}`],
    })),
});
