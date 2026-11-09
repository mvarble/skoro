import { defineGraph } from '$lib/graph/spec';

// A small map: three poses x1, x2, x3 with a prior on x1, odometry between consecutive poses and a
// loop closure between x1 and x3, and two landmarks, l1 seen from x1 and x2, l2 from x2 and x3.
// Drawn as its factor structure, as the elimination tree of the order l1, l2, x1, x2, x3 with each
// block's separator and its clique, and as the junction tree of those cliques.

export const mapFactors = defineGraph({
    title: 'The factor structure of a small map',
    caption: String.raw`Three poses $x_1, x_2, x_3$ and two landmarks $l_1, l_2$. Each variable is a circle and each factor a square, with a line from a factor to every variable it involves: a prior on $x_1$, odometry between consecutive poses, a loop closure between $x_1$ and $x_3$, and two sightings of each landmark.`,
    width: 560,
    height: 360,
    nodeKinds: {
        variable: { name: 'Variable', hue: 'orange' },
        factor: { name: 'Factor', hue: 'purple', shape: 'square' },
    },
    edgeKinds: {
        involves: { name: 'involves' },
    },
    nodes: [
        { id: 'x1', x: 120, y: 170, kind: 'variable', label: String.raw`$x_1$` },
        { id: 'x2', x: 300, y: 170, kind: 'variable', label: String.raw`$x_2$` },
        { id: 'x3', x: 480, y: 170, kind: 'variable', label: String.raw`$x_3$` },
        { id: 'l1', x: 210, y: 300, kind: 'variable', label: String.raw`$l_1$` },
        { id: 'l2', x: 390, y: 300, kind: 'variable', label: String.raw`$l_2$` },
        { id: 'p', x: 30, y: 170, kind: 'factor', r: 9, title: 'prior on x1' },
        { id: 'o12', x: 210, y: 170, kind: 'factor', r: 9, title: 'odometry x1 to x2' },
        { id: 'o23', x: 390, y: 170, kind: 'factor', r: 9, title: 'odometry x2 to x3' },
        { id: 'c13', x: 300, y: 60, kind: 'factor', r: 9, title: 'loop closure x1 to x3' },
        { id: 's11', x: 165, y: 235, kind: 'factor', r: 9, title: 'l1 seen from x1' },
        { id: 's21', x: 255, y: 235, kind: 'factor', r: 9, title: 'l1 seen from x2' },
        { id: 's22', x: 345, y: 235, kind: 'factor', r: 9, title: 'l2 seen from x2' },
        { id: 's32', x: 435, y: 235, kind: 'factor', r: 9, title: 'l2 seen from x3' },
    ],
    edges: [
        { from: 'p', to: 'x1', kind: 'involves', arrow: 'none' },
        { from: 'o12', to: 'x1', kind: 'involves', arrow: 'none' },
        { from: 'o12', to: 'x2', kind: 'involves', arrow: 'none' },
        { from: 'o23', to: 'x2', kind: 'involves', arrow: 'none' },
        { from: 'o23', to: 'x3', kind: 'involves', arrow: 'none' },
        { from: 'c13', to: 'x1', kind: 'involves', arrow: 'none' },
        { from: 'c13', to: 'x3', kind: 'involves', arrow: 'none' },
        { from: 's11', to: 'x1', kind: 'involves', arrow: 'none' },
        { from: 's11', to: 'l1', kind: 'involves', arrow: 'none' },
        { from: 's21', to: 'x2', kind: 'involves', arrow: 'none' },
        { from: 's21', to: 'l1', kind: 'involves', arrow: 'none' },
        { from: 's22', to: 'x2', kind: 'involves', arrow: 'none' },
        { from: 's22', to: 'l2', kind: 'involves', arrow: 'none' },
        { from: 's32', to: 'x3', kind: 'involves', arrow: 'none' },
        { from: 's32', to: 'l2', kind: 'involves', arrow: 'none' },
    ],
});

export const mapTree = defineGraph({
    title: 'The elimination tree of the map, grouped into cliques',
    caption: String.raw`The order $l_1, l_2, x_1, x_2, x_3$, with each block's separator beside it and an arrow from each parent to its children. The colours are the cliques: $x_2$ and $x_3$ each have a child whose separator is their own with themselves added, so the chain $x_1, x_2, x_3$ is one clique, eliminated as one block; $l_1$ and $l_2$ are cliques of one block each.`,
    width: 560,
    height: 400,
    nodeKinds: {
        poses: { name: String.raw`Clique of $x_1, x_2, x_3$`, hue: 'blue' },
        first: { name: String.raw`Clique of $l_1$`, hue: 'green' },
        second: { name: String.raw`Clique of $l_2$`, hue: 'amber' },
    },
    edgeKinds: {
        parent: { name: 'parent of' },
    },
    nodes: [
        {
            id: 'x3',
            x: 280,
            y: 350,
            kind: 'poses',
            label: String.raw`$x_3$`,
            annotation: String.raw`$S = \varnothing$`,
            annotationAngle: 0,
        },
        {
            id: 'x2',
            x: 280,
            y: 250,
            kind: 'poses',
            label: String.raw`$x_2$`,
            annotation: String.raw`$S = \{x_3\}$`,
            annotationAngle: 0,
        },
        {
            id: 'x1',
            x: 170,
            y: 150,
            kind: 'poses',
            label: String.raw`$x_1$`,
            annotation: String.raw`$S = \{x_2, x_3\}$`,
            annotationAngle: 180,
        },
        {
            id: 'l2',
            x: 390,
            y: 150,
            kind: 'second',
            label: String.raw`$l_2$`,
            annotation: String.raw`$S = \{x_2, x_3\}$`,
            annotationAngle: 0,
        },
        {
            id: 'l1',
            x: 170,
            y: 50,
            kind: 'first',
            label: String.raw`$l_1$`,
            annotation: String.raw`$S = \{x_1, x_2\}$`,
            annotationAngle: 180,
        },
    ],
    edges: [
        { from: 'x3', to: 'x2', kind: 'parent' },
        { from: 'x2', to: 'x1', kind: 'parent' },
        { from: 'x2', to: 'l2', kind: 'parent' },
        { from: 'x1', to: 'l1', kind: 'parent' },
    ],
});

export const mapJunction = defineGraph({
    title: 'The junction tree of the map',
    caption: String.raw`Each clique is a box listing the blocks it eliminates, then, after the bar, its separator. An arrow runs from each clique to its children, labelled with the child's separator: the blocks the two cliques share. Each clique is eliminated by one factorization, the children before the root.`,
    width: 560,
    height: 300,
    nodeKinds: {
        poses: { name: String.raw`Clique of $x_1, x_2, x_3$`, hue: 'blue', shape: 'square' },
        first: { name: String.raw`Clique of $l_1$`, hue: 'green', shape: 'square' },
        second: { name: String.raw`Clique of $l_2$`, hue: 'amber', shape: 'square' },
    },
    edgeKinds: {
        parent: { name: 'parent of' },
    },
    nodes: [
        {
            id: 'c1',
            x: 280,
            y: 230,
            r: 50,
            kind: 'poses',
            label: String.raw`$x_1, x_2, x_3$`,
            annotation: 'root',
            annotationAngle: 0,
        },
        { id: 'c2', x: 130, y: 75, r: 50, kind: 'first', label: String.raw`$l_1 \mid x_1, x_2$` },
        { id: 'c3', x: 430, y: 75, r: 50, kind: 'second', label: String.raw`$l_2 \mid x_2, x_3$` },
    ],
    edges: [
        { from: 'c1', to: 'c2', kind: 'parent', label: String.raw`$\{x_1, x_2\}$` },
        { from: 'c1', to: 'c3', kind: 'parent', label: String.raw`$\{x_2, x_3\}$` },
    ],
});
