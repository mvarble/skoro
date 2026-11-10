import { defineGraph } from '$lib/graph/spec';

// A star: a hub variable h, three leaves, a factor joining the hub to each leaf and a prior on the
// hub. Drawn first as its factor structure, then as the elimination tree of the order that
// eliminates the leaves first.

export const starFactors = defineGraph({
    title: 'The factor structure of a star',
    caption: String.raw`Each variable is a circle and each factor a square, with a line from a factor to every variable it involves. Eliminating $h$ first would join $l_1$, $l_2$ and $l_3$ in one new factor; eliminating each leaf first joins it to nothing new.`,
    width: 560,
    height: 300,
    nodeKinds: {
        variable: { name: 'Variable', hue: 'orange' },
        factor: { name: 'Factor', hue: 'purple', shape: 'square' },
    },
    edgeKinds: {
        touches: { name: 'involves' },
    },
    nodes: [
        { id: 'p', x: 60, y: 150, kind: 'factor', label: String.raw`$f_0$` },
        { id: 'h', x: 180, y: 150, kind: 'variable', label: String.raw`$h$` },
        { id: 'f1', x: 330, y: 250, kind: 'factor', label: String.raw`$f_1$` },
        { id: 'f2', x: 330, y: 150, kind: 'factor', label: String.raw`$f_2$` },
        { id: 'f3', x: 330, y: 50, kind: 'factor', label: String.raw`$f_3$` },
        { id: 'l1', x: 480, y: 250, kind: 'variable', label: String.raw`$l_1$` },
        { id: 'l2', x: 480, y: 150, kind: 'variable', label: String.raw`$l_2$` },
        { id: 'l3', x: 480, y: 50, kind: 'variable', label: String.raw`$l_3$` },
    ],
    edges: [
        { from: 'p', to: 'h', kind: 'touches', arrow: 'none' },
        { from: 'f1', to: 'h', kind: 'touches', arrow: 'none' },
        { from: 'f2', to: 'h', kind: 'touches', arrow: 'none' },
        { from: 'f3', to: 'h', kind: 'touches', arrow: 'none' },
        { from: 'f1', to: 'l1', kind: 'touches', arrow: 'none' },
        { from: 'f2', to: 'l2', kind: 'touches', arrow: 'none' },
        { from: 'f3', to: 'l3', kind: 'touches', arrow: 'none' },
    ],
});

export const starTree = defineGraph({
    title: 'The elimination tree of a star',
    caption: String.raw`The order $l_1, l_2, l_3, h$. Each leaf's separator is $\{h\}$, so $h$ is the parent of all three, and an arrow runs from a parent to each of its children. The root $h$, eliminated last, has no incoming arrow; the leaves of the tree, eliminated first, have no outgoing one.`,
    width: 560,
    height: 250,
    nodeKinds: {
        variable: { name: 'Variable', hue: 'orange' },
    },
    edgeKinds: {
        parent: { name: 'parent of' },
    },
    nodes: [
        {
            id: 'h',
            x: 280,
            y: 200,
            kind: 'variable',
            label: String.raw`$h$`,
            annotation: 'root',
            annotationAngle: 0,
        },
        { id: 'l1', x: 130, y: 60, kind: 'variable', label: String.raw`$l_1$` },
        { id: 'l2', x: 280, y: 60, kind: 'variable', label: String.raw`$l_2$` },
        { id: 'l3', x: 430, y: 60, kind: 'variable', label: String.raw`$l_3$` },
    ],
    edges: [
        { from: 'h', to: 'l1', kind: 'parent' },
        { from: 'h', to: 'l2', kind: 'parent' },
        { from: 'h', to: 'l3', kind: 'parent' },
    ],
});
