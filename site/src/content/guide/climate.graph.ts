import { defineGraph } from '$lib/graph/spec';

// Example 2: a two-layer energy-balance model, integrated step by step, observed each year.
export const climate = defineGraph({
    title: 'An energy-balance model, unrolled',
    caption:
        'The parameters $\\theta$ and the noise factor $L$ have priors, so both are stochastic. ' +
        'Each state $s_k = (T_k, T_{D,k})$ is one integration step from the last; the forcing ' +
        '$F_k$ enters each step as a constant (not drawn). Each year is observed with noise $LL^\\top$.',
    width: 780,
    height: 370,
    nodes: [
        { id: 'theta', x: 510, y: 330, kind: 'stochastic', label: '$\\theta$', annotation: '$(\\lambda, \\gamma, C, C_D)$, positive', annotationAngle: 0 },
        { id: 's0', x: 220, y: 220, kind: 'constant', label: '$s_0$', annotation: 'pre-industrial', annotationAngle: 180 },
        { id: 's1', x: 370, y: 220, kind: 'mapped', label: '$s_1$' },
        { id: 's2', x: 510, y: 220, kind: 'mapped', label: '$s_2$' },
        { id: 's3', x: 650, y: 220, kind: 'mapped', label: '$s_3$', annotation: '$\\Phi_\\theta(s_2, F_3)$', annotationAngle: 0 },
        { id: 'y1', x: 370, y: 110, kind: 'observed', label: '$y_1$' },
        { id: 'y2', x: 510, y: 110, kind: 'observed', label: '$y_2$' },
        { id: 'y3', x: 650, y: 110, kind: 'observed', label: '$y_3$', annotation: '$(T, N)$ observed', annotationAngle: 0 },
        { id: 'L', x: 440, y: 25, kind: 'stochastic', label: '$L$', annotation: 'Tril', annotationAngle: 180 },
    ],
    edges: [
        { from: 's0', to: 's1' },
        { from: 's1', to: 's2' },
        { from: 's2', to: 's3' },
        { from: 'theta', to: 's1' },
        { from: 'theta', to: 's2' },
        { from: 'theta', to: 's3' },
        { from: 's1', to: 'y1' },
        { from: 's2', to: 'y2' },
        { from: 's3', to: 'y3' },
        { from: 'L', to: 'y1' },
        { from: 'L', to: 'y2' },
        { from: 'L', to: 'y3' },
    ],
});
