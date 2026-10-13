import { defineGraph, type NodeKindSpec } from '$lib/graph/spec';

// A small sequential model drawn two ways. A vehicle at x₀ then x₁ measures a landmark m once from
// each position (z₀ and z₁). The probabilistic graph shows its quantities and terms, labelled by the
// quantities themselves and by the regular conditional distributions the terms give densities of,
// since it holds structure rather than values; the compilation graph shows how one of those terms,
// the density of z₀, is computed from values.

const quantity: NodeKindSpec = { name: 'Quantity', hue: 'orange' };

export const probabilistic = defineGraph({
    title: 'A probabilistic graph',
    caption: String.raw`Each term is labelled by the regular conditional distribution it gives the density of. An arrow into a term comes from a quantity it is conditioned on; a dotted arrow out of a term points to the quantity it scores. Retiring $\randomPoint_0$ and $\bfz_0$ replaces the terms of $\randomPoint_0$, $\randomPoint_1$ and $\bfz_0$ by one term over the Markov blanket $\{\randomPoint_1, \bfm\}$.`,
    width: 720,
    height: 300,
    // Larger than the default, so that the subscripts of the distributions stay legible.
    fontSize: 20,
    nodeKinds: {
        quantity,
        term: { name: 'Term', hue: 'purple', shape: 'square' },
    },
    edgeKinds: {
        conditions: { name: 'conditions a term' },
        scores: { name: 'scores a quantity', line: 'dotted' },
    },
    nodes: [
        {
            id: 'tx0',
            x: 50,
            y: 240,
            kind: 'term',
            label: String.raw`$t_1$`,
            annotation: String.raw`$\rmP_{\randomPoint_0}$`,
        },
        {
            id: 'tm',
            x: 50,
            y: 60,
            kind: 'term',
            label: String.raw`$t_2$`,
            annotation: String.raw`$\rmP_{\bfm}$`,
            annotationAngle: 90,
        },
        { id: 'x0', x: 170, y: 240, kind: 'quantity', label: String.raw`$\randomPoint_0$` },
        { id: 'm', x: 170, y: 60, kind: 'quantity', label: String.raw`$\bfm$` },
        {
            id: 'tx1',
            x: 300,
            y: 240,
            kind: 'term',
            label: String.raw`$t_3$`,
            annotation: String.raw`$\rmP_{\randomPoint_1 \mid \randomPoint_0}$`,
            annotationAngle: 90,
        },
        {
            id: 'tz0',
            x: 300,
            y: 150,
            kind: 'term',
            label: String.raw`$t_4$`,
            annotation: String.raw`$\rmP_{\bfz_0 \mid \randomPoint_0, \bfm}$`,
            annotationAngle: -60,
        },
        { id: 'x1', x: 430, y: 240, kind: 'quantity', label: String.raw`$\randomPoint_1$` },
        { id: 'z0', x: 430, y: 150, kind: 'quantity', label: String.raw`$\bfz_0$` },
        {
            id: 'tz1',
            x: 560,
            y: 150,
            kind: 'term',
            label: String.raw`$t_5$`,
            annotation: String.raw`$\rmP_{\bfz_1 \mid \randomPoint_1, \bfm}$`,
            annotationAngle: 90,
        },
        { id: 'z1', x: 670, y: 150, kind: 'quantity', label: String.raw`$\bfz_1$` },
    ],
    edges: [
        { from: 'tx0', to: 'x0', kind: 'scores' },
        { from: 'tm', to: 'm', kind: 'scores' },
        { from: 'x0', to: 'tx1', kind: 'conditions' },
        { from: 'tx1', to: 'x1', kind: 'scores' },
        { from: 'x0', to: 'tz0', kind: 'conditions' },
        { from: 'm', to: 'tz0', kind: 'conditions' },
        { from: 'tz0', to: 'z0', kind: 'scores' },
        { from: 'x1', to: 'tz1', kind: 'conditions' },
        { from: 'm', to: 'tz1', kind: 'conditions', via: [[430, 60]], curve: 'bezier' },
        { from: 'tz1', to: 'z1', kind: 'scores' },
    ],
});

export const compilation = defineGraph({
    title: 'A compilation graph',
    caption: String.raw`The log-density of the term $t_4$, $\logterm_4 = -\tfrac12 \sum_i (z_0 - (m - \point_0))_i^2$, as it is computed. Each edge runs from an input of an operation to its output. The roots are the quantities $\point_0$, $m$, $z_0$ and the constant $-\tfrac12$; every other node is a function of them.`,
    width: 780,
    height: 280,
    nodeKinds: {
        quantity,
        constant: { name: 'Constant', hue: 'gray' },
        mapped: { name: 'Mapped value', hue: 'green' },
        logDensity: { name: "A term's log-density", hue: 'purple', shape: 'square' },
    },
    edgeKinds: { input: { name: 'input of an operation' } },
    nodes: [
        { id: 'x0', x: 60, y: 90, kind: 'quantity', label: String.raw`$\point_0$` },
        { id: 'm', x: 60, y: 190, kind: 'quantity', label: String.raw`$m$` },
        { id: 'z0', x: 190, y: 245, kind: 'quantity', label: String.raw`$z_0$` },
        { id: 'h', x: 190, y: 140, kind: 'mapped', annotation: String.raw`$m - \point_0$` },
        { id: 'r', x: 320, y: 190, kind: 'mapped', annotation: String.raw`$z_0 - (\cdot)$` },
        { id: 'sq', x: 450, y: 190, kind: 'mapped', annotation: String.raw`$r \odot r$` },
        { id: 'sum', x: 580, y: 190, kind: 'mapped', annotation: String.raw`$\textstyle\sum_i$` },
        { id: 'c', x: 580, y: 70, kind: 'constant', label: String.raw`$-\tfrac12$` },
        { id: 'l', x: 710, y: 130, kind: 'logDensity', label: String.raw`$\logterm_4$` },
    ],
    edges: [
        { from: 'm', to: 'h', kind: 'input' },
        { from: 'x0', to: 'h', kind: 'input' },
        { from: 'z0', to: 'r', kind: 'input' },
        { from: 'h', to: 'r', kind: 'input' },
        { from: 'r', to: 'sq', kind: 'input', bend: 20 },
        { from: 'r', to: 'sq', kind: 'input', bend: -20 },
        { from: 'sq', to: 'sum', kind: 'input' },
        { from: 'sum', to: 'l', kind: 'input' },
        { from: 'c', to: 'l', kind: 'input' },
    ],
});
