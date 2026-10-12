import { defineGraph } from '$lib/graph/spec';

// A small sequential model drawn two ways. A vehicle at x₀ then x₁ measures a landmark m once from
// each position (z₀ and z₁). The probabilistic graph shows its quantities and terms; the
// compilation graph shows how one of those terms, the density of z₀, is computed.

export const probabilistic = defineGraph({
    title: 'A probabilistic graph',
    caption: String.raw`Circles are quantities and squares are terms. An arrow into a term comes from a quantity it is conditioned on; a dotted arrow out of a term points to the quantity it scores. Retiring $\point_0$ and $z_0$ replaces the terms of $\point_0$, $\point_1$ and $z_0$ by one term over the Markov blanket $\{\point_1, m\}$.`,
    width: 720,
    height: 300,
    legend: false,
    nodes: [
        { id: 'tx0', x: 50, y: 240, kind: 'log-likelihood', label: String.raw`$t_1$`, annotation: String.raw`$\density(\point_0)$` },
        { id: 'tm', x: 50, y: 60, kind: 'log-likelihood', label: String.raw`$t_2$`, annotation: String.raw`$\density(m)$`, annotationAngle: 90 },
        { id: 'x0', x: 170, y: 240, kind: 'stochastic', label: String.raw`$\point_0$` },
        { id: 'm', x: 170, y: 60, kind: 'stochastic', label: String.raw`$m$` },
        {
            id: 'tx1',
            x: 300,
            y: 240,
            kind: 'log-likelihood',
            label: String.raw`$t_3$`,
            annotation: String.raw`$\density(\point_1 \mid \point_0)$`,
            annotationAngle: 90,
        },
        {
            id: 'tz0',
            x: 300,
            y: 150,
            kind: 'log-likelihood',
            label: String.raw`$t_4$`,
            annotation: String.raw`$\density(z_0 \mid \point_0, m)$`,
            annotationAngle: -60,
        },
        { id: 'x1', x: 430, y: 240, kind: 'stochastic', label: String.raw`$\point_1$` },
        { id: 'z0', x: 430, y: 150, kind: 'stochastic', label: String.raw`$z_0$` },
        {
            id: 'tz1',
            x: 560,
            y: 150,
            kind: 'log-likelihood',
            label: String.raw`$t_5$`,
            annotation: String.raw`$\density(z_1 \mid \point_1, m)$`,
            annotationAngle: 90,
        },
        { id: 'z1', x: 670, y: 150, kind: 'stochastic', label: String.raw`$z_1$` },
    ],
    edges: [
        { from: 'tx0', to: 'x0', kind: 'scores' },
        { from: 'tm', to: 'm', kind: 'scores' },
        { from: 'x0', to: 'tx1' },
        { from: 'tx1', to: 'x1', kind: 'scores' },
        { from: 'x0', to: 'tz0' },
        { from: 'm', to: 'tz0' },
        { from: 'tz0', to: 'z0', kind: 'scores' },
        { from: 'x1', to: 'tz1' },
        { from: 'm', to: 'tz1', via: [[430, 60]], curve: 'bezier' },
        { from: 'tz1', to: 'z1', kind: 'scores' },
    ],
});

export const compilation = defineGraph({
    title: 'A compilation graph',
    caption: String.raw`The log-density of the term $t_4$, $\logterm_4 = -\tfrac12 \sum_i (z_0 - (m - \point_0))_i^2$, as it is computed. Each node is a value and each edge runs from an input of an operation to its output. The leaves are the quantities $\point_0$, $m$, $z_0$ and the constant $-\tfrac12$; every other node is a function of them.`,
    width: 780,
    height: 280,
    legend: false,
    nodes: [
        { id: 'x0', x: 60, y: 90, kind: 'stochastic', label: String.raw`$\point_0$` },
        { id: 'm', x: 60, y: 190, kind: 'stochastic', label: String.raw`$m$` },
        { id: 'z0', x: 190, y: 245, kind: 'stochastic', label: String.raw`$z_0$` },
        { id: 'h', x: 190, y: 140, kind: 'mapped', annotation: String.raw`$m - \point_0$` },
        { id: 'r', x: 320, y: 190, kind: 'mapped', annotation: String.raw`$z_0 - (\cdot)$` },
        { id: 'sq', x: 450, y: 190, kind: 'mapped', annotation: String.raw`$r \odot r$` },
        { id: 'sum', x: 580, y: 190, kind: 'mapped', annotation: String.raw`$\textstyle\sum_i$` },
        { id: 'c', x: 580, y: 70, kind: 'constant', label: String.raw`$-\tfrac12$` },
        {
            id: 'l',
            x: 710,
            y: 130,
            kind: 'log-likelihood',
            label: String.raw`$\logterm_4$`,
        },
    ],
    edges: [
        { from: 'm', to: 'h' },
        { from: 'x0', to: 'h' },
        { from: 'z0', to: 'r' },
        { from: 'h', to: 'r' },
        { from: 'r', to: 'sq', bend: 20 },
        { from: 'r', to: 'sq', bend: -20 },
        { from: 'sq', to: 'sum' },
        { from: 'sum', to: 'l' },
        { from: 'c', to: 'l' },
    ],
});
