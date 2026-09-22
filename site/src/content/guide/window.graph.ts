import { defineGraph, type EdgeSpec, type NodeSpec } from '$lib/graph/spec';

// Example 3: one slide of a three-pose window, before and after the oldest pose is retired.

const chain = (retired: boolean): NodeSpec[] => [
    { id: 'u0', x: 230, y: 220, kind: 'constant', label: '$u_0$', muted: retired },
    { id: 'u1', x: 510, y: 220, kind: 'constant', label: '$u_1$' },
    { id: 'x0', x: 90, y: 140, kind: 'stochastic', label: '$x_0$', muted: retired },
    { id: 'm1', x: 230, y: 140, kind: 'mapped', label: '$m_1$', muted: retired },
    {
        id: 'x1',
        x: 370,
        y: 140,
        kind: retired ? 'parameter' : 'stochastic',
        label: '$x_1$',
        annotation: retired ? 'demoted' : undefined,
        annotationAngle: 30,
    },
    { id: 'm2', x: 510, y: 140, kind: 'mapped', label: '$m_2$' },
    { id: 'x2', x: 650, y: 140, kind: 'stochastic', label: '$x_2$' },
    { id: 'z0', x: 90, y: 40, kind: 'observed', label: '$z_0$', muted: retired },
    { id: 'z1', x: 370, y: 40, kind: 'observed', label: '$z_1$' },
    { id: 'z2', x: 650, y: 40, kind: 'observed', label: '$z_2$' },
];

const chainEdges = (retired: boolean): EdgeSpec[] => [
    { from: 'u0', to: 'm1', muted: retired },
    { from: 'x0', to: 'm1', muted: retired },
    { from: 'm1', to: 'x1', muted: retired },
    { from: 'u1', to: 'm2' },
    { from: 'x1', to: 'm2' },
    { from: 'm2', to: 'x2' },
    { from: 'x0', to: 'z0', muted: retired },
    { from: 'x1', to: 'z1' },
    { from: 'x2', to: 'z2' },
];

export const windowBefore = defineGraph({
    title: 'The window before it slides',
    caption:
        'Three poses, each measured once. $x_0$ is about to leave the window, and $x_1$, the only ' +
        'pose it touches, is its Markov blanket.',
    width: 740,
    height: 260,
    nodes: chain(false),
    edges: chainEdges(false),
});

export const windowAfter = defineGraph({
    title: 'The window after it slides',
    caption:
        '$x_1$ is demoted to a parameter, which severs its edge from $m_1$. A joint factor ' +
        '$\\tilde p$ over $x_1$, anchored at the frozen estimate $\\hat x_1$, stands in for ' +
        'everything that was known about $x_0$. The faded nodes are retired by remove_closure ' +
        'and reclaimed by cleanup.',
    width: 740,
    height: 350,
    nodes: [
        ...chain(true),
        { id: 'anchor', x: 230, y: 305, kind: 'constant', label: '$\\hat x_1$', annotation: 'frozen', annotationAngle: 180 },
        { id: 'r', x: 370, y: 305, kind: 'mapped', label: '$\\tilde r$', annotation: '$W(x_1 \\ominus \\hat x_1)$', annotationAngle: 90 },
        { id: 'pt', x: 540, y: 305, kind: 'log-likelihood', label: '$\\tilde p$', annotation: 'joint factor', annotationAngle: 0 },
    ],
    edges: [
        ...chainEdges(true),
        { from: 'anchor', to: 'r' },
        { from: 'x1', to: 'r' },
        { from: 'r', to: 'pt' },
        { from: 'pt', to: 'r', kind: 'binding', role: 'residual', bend: 35 },
    ],
});
