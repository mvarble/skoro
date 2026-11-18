import { defineGraph, type EdgeSpec, type NodeKindSpec, type NodeSpec } from '$lib/graph/spec';

// The running example of "Graphical representations", drawn four ways. Two roots X₁ and X₂ are the
// parents of X₃, which is the parent of X₄ and X₅, and a likelihood factor L reads X₂ and X₄. The
// dependence graph has the quantities alone; the probabilistic graph adds the factors of the joint
// density, with the direction of each kernel; the factor graph forgets the directions; and the last
// figure picks out the Markov blanket of X₂ in the factor graph.

const quantity: NodeKindSpec = { name: 'Quantity', hue: 'orange' };
const kernel: NodeKindSpec = { name: 'Kernel density', hue: 'purple', shape: 'square' };
const likelihood: NodeKindSpec = {
    name: 'Likelihood factor',
    hue: 'purple',
    shape: 'square',
    outline: 'dashed',
};

const quantities: NodeSpec[] = [
    { id: 'x1', x: 160, y: 90, kind: 'quantity', label: String.raw`$\randomPoint_1$` },
    { id: 'x2', x: 160, y: 270, kind: 'quantity', label: String.raw`$\randomPoint_2$` },
    { id: 'x3', x: 400, y: 180, kind: 'quantity', label: String.raw`$\randomPoint_3$` },
    { id: 'x4', x: 650, y: 270, kind: 'quantity', label: String.raw`$\randomPoint_4$` },
    { id: 'x5', x: 650, y: 90, kind: 'quantity', label: String.raw`$\randomPoint_5$` },
];

// Each factor, where it sits, where its formula is written, the quantities it reads and the one
// it scores.
const factors: {
    node: NodeSpec;
    reads: string[];
    scores?: string;
}[] = [
    {
        node: {
            id: 'p1',
            x: 50,
            y: 90,
            kind: 'kernel',
            label: String.raw`$\density_1$`,
            annotation: String.raw`$\density_1(\point_1)$`,
        },
        reads: [],
        scores: 'x1',
    },
    {
        node: {
            id: 'p2',
            x: 50,
            y: 270,
            kind: 'kernel',
            label: String.raw`$\density_2$`,
            annotation: String.raw`$\density_2(\point_2)$`,
            annotationAngle: 90,
        },
        reads: [],
        scores: 'x2',
    },
    {
        node: {
            id: 'p3',
            x: 285,
            y: 180,
            kind: 'kernel',
            label: String.raw`$\density_3$`,
            annotation: String.raw`$\density_3(\point_3 \mid \point_1, \point_2)$`,
        },
        reads: ['x1', 'x2'],
        scores: 'x3',
    },
    {
        node: {
            id: 'p4',
            x: 530,
            y: 270,
            kind: 'kernel',
            label: String.raw`$\density_4$`,
            annotation: String.raw`$\density_4(\point_4 \mid \point_3)$`,
            annotationAngle: -60,
        },
        reads: ['x3'],
        scores: 'x4',
    },
    {
        node: {
            id: 'p5',
            x: 530,
            y: 90,
            kind: 'kernel',
            label: String.raw`$\density_5$`,
            annotation: String.raw`$\density_5(\point_5 \mid \point_3)$`,
        },
        reads: ['x3'],
        scores: 'x5',
    },
    {
        node: {
            id: 'L',
            x: 405,
            y: 380,
            kind: 'likelihood',
            label: String.raw`$L$`,
            annotation: String.raw`$L(\point_2, \point_4)$`,
            annotationAngle: 90,
        },
        reads: ['x2', 'x4'],
    },
];

const nodes = [...quantities, ...factors.map((factor) => factor.node)];

export const dependence = defineGraph({
    title: 'A dependence graph',
    caption: String.raw`The quantities of the running example, with an arrow from each parent to its child: $\operatorname{pa}(3) = \{1, 2\}$ and $\operatorname{pa}(4) = \operatorname{pa}(5) = \{3\}$. The roots $\randomPoint_1$ and $\randomPoint_2$ may be drawn in either order, and so may $\randomPoint_4$ and $\randomPoint_5$, so $1, 2, 3, 4, 5$ and $2, 1, 3, 5, 4$ are both ancestral orders. The likelihood factor generates nothing, and does not appear.`,
    width: 660,
    height: 360,
    fontSize: 22,
    nodeRadius: 25,
    nodeKinds: { quantity },
    edgeKinds: { parent: { name: 'is a parent of' } },
    nodes: quantities.map((node) => ({ ...node, x: node.x - 80, y: node.y })),
    edges: [
        { from: 'x1', to: 'x3', kind: 'parent' },
        { from: 'x2', to: 'x3', kind: 'parent' },
        { from: 'x3', to: 'x4', kind: 'parent' },
        { from: 'x3', to: 'x5', kind: 'parent' },
    ],
});

export const probabilistic = defineGraph({
    title: 'A probabilistic graph',
    caption: String.raw`The running example with its terms. Each quantity is a circle and each term a square: a kernel density, solid, or a likelihood factor, dashed. A solid arrow runs from a quantity to each term that reads it, and a dotted arrow from a kernel density to the quantity it is the density of. Contracting each kernel density into arrows from what it reads to what it scores, and dropping the likelihood factor, gives back the dependence graph.`,
    width: 760,
    height: 450,
    fontSize: 22,
    nodeRadius: 25,
    nodeKinds: { quantity, kernel, likelihood },
    edgeKinds: {
        reads: { name: 'is read by' },
        scores: { name: 'is the density of', line: 'dotted' },
    },
    nodes,
    edges: factors.flatMap(({ node, reads, scores }) => [
        ...reads.map((from): EdgeSpec => ({ from, to: node.id, kind: 'reads' })),
        ...(scores ? [{ from: node.id, to: scores, kind: 'scores' }] : []),
    ]),
});

// Each term joined to every quantity it involves, without direction.
const involvements = (muted: (factor: string) => boolean = () => false) =>
    factors.flatMap(({ node, reads, scores }) =>
        [...reads, ...(scores ? [scores] : [])].map((quantity): EdgeSpec => ({
            from: node.id,
            to: quantity,
            kind: 'involves',
            arrow: 'none',
            muted: muted(node.id),
        })),
    );

const unannotated = nodes.map((node) => ({ ...node, annotation: undefined }));

export const factorGraph = defineGraph({
    title: 'A factor graph',
    caption: String.raw`The probabilistic graph of the running example with its directions forgotten: a line joins each term to every quantity it reads or scores. The terms that involve $\randomPoint_3$ are $\density_3$, which scores it, and $\density_4$ and $\density_5$, which read it; no other term reads it.`,
    width: 760,
    height: 450,
    fontSize: 22,
    nodeRadius: 25,
    nodeKinds: { quantity, kernel, likelihood },
    edgeKinds: { involves: { name: 'involves' } },
    nodes: unannotated,
    edges: involvements(),
});

// The terms that involve X₂, and the quantities they involve besides it.
const touching = new Set(['p2', 'p3', 'L']);
const blanketOf2 = new Set(['x1', 'x3', 'x4']);

export const markovBlanket = defineGraph({
    title: 'A Markov blanket in the factor graph',
    caption: String.raw`The Markov blanket of $\randomPoint_2$, the quantities two lines away from it in the factor graph. The terms that involve it are $\density_2$, $\density_3$ and $L$; besides $\randomPoint_2$ they involve $\randomPoint_1$, the other parent of its child; $\randomPoint_3$, its child; and $\randomPoint_4$, which only the likelihood factor ties it to. Resampling, optimizing or integrating out $\randomPoint_2$ reads only those three terms, and nothing outside $\{\randomPoint_1, \randomPoint_3, \randomPoint_4\}$.`,
    width: 760,
    height: 450,
    fontSize: 22,
    nodeRadius: 25,
    nodeKinds: {
        focus: { name: String.raw`The quantity $\randomPoint_2$`, hue: 'blue', fill: 'shaded' },
        blanket: { name: 'Its Markov blanket', hue: 'orange', fill: 'shaded' },
        quantity,
        kernel,
        likelihood,
    },
    edgeKinds: { involves: { name: 'involves' } },
    nodes: unannotated.map((node) => {
        if (node.id == 'x2') return { ...node, kind: 'focus' };
        if (blanketOf2.has(node.id)) return { ...node, kind: 'blanket' };
        if (node.kind == 'quantity') return { ...node, muted: true };
        return { ...node, muted: !touching.has(node.id) };
    }),
    edges: involvements((factor) => !touching.has(factor)),
});
