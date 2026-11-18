import { defineGraph, type EdgeSpec, type NodeKindSpec, type NodeSpec } from '$lib/graph/spec';

// One small model built forward, drawn twice. Two roots X₁ and X₂ are the parents of X₃, which is
// the parent of X₄ and X₅, and a likelihood factor L reads X₂ and X₄. The probabilistic graph keeps
// the direction of each kernel, from the quantities it reads to the quantity it gives the density
// of; the factor graph forgets it, and its neighbours of neighbours are the Markov blankets.

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

export const probabilistic = defineGraph({
    title: 'A probabilistic graph',
    caption: String.raw`Each quantity is a circle and each factor of the joint density a square: a kernel density, solid, or a likelihood factor, dashed. A solid arrow runs from a quantity to each factor that reads it, and a dotted arrow from a kernel density to the quantity it is the density of. Contracting each kernel density into arrows from what it reads to what it scores, and dropping the likelihood factor, gives the dependence graph, with $\operatorname{pa}(3) = \{1, 2\}$ and $\operatorname{pa}(4) = \operatorname{pa}(5) = \{3\}$.`,
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
        ...reads.map((from): EdgeSpec => ({
            from,
            to: node.id,
            kind: 'reads',
        })),
        ...(scores ? [{ from: node.id, to: scores, kind: 'scores' }] : []),
    ]),
});

export const factorGraph = defineGraph({
    title: 'The factor graph of the same model',
    caption: String.raw`The same factors with their directions forgotten: a line joins each factor to every quantity it reads or scores. The Markov blanket of a quantity is the set of quantities two lines away from it. That of $\randomPoint_3$ is $\{\randomPoint_1, \randomPoint_2, \randomPoint_4, \randomPoint_5\}$, its parents and its children. That of $\randomPoint_2$ is $\{\randomPoint_1, \randomPoint_3, \randomPoint_4\}$: its child, the other parent $\randomPoint_1$ of that child, and $\randomPoint_4$, which only the likelihood factor ties it to.`,
    width: 760,
    height: 450,
    fontSize: 22,
    nodeRadius: 25,
    nodeKinds: { quantity, kernel, likelihood },
    edgeKinds: { involves: { name: 'involves' } },
    nodes: nodes.map((node) => ({ ...node, annotation: undefined })),
    edges: factors.flatMap(({ node, reads, scores }) =>
        [...reads, ...(scores ? [scores] : [])].map((quantity): EdgeSpec => ({
            from: node.id,
            to: quantity,
            kind: 'involves',
            arrow: 'none',
        })),
    ),
});
