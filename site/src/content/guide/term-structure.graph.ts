import { defineGraph } from '$lib/graph/spec';

// Example 4: a Gaussian affine term-structure model as a state-space model.
export const termStructure = defineGraph({
    title: 'An affine yield-curve model, three days of it',
    caption:
        'The risk-neutral parameters and $\\Sigma$ determine the loadings through the Riccati ODE; ' +
        'the physical dynamics move the factors $x_t$ from day to day; each day’s curve $y_t$ is the ' +
        'loadings applied to that day’s factors. $\\Sigma$ also scales every transition (only the ' +
        'first is drawn), and the measurement noise $R$ is not drawn.',
    width: 800,
    height: 400,
    nodes: [
        { id: 'q', x: 110, y: 360, kind: 'parameter', label: '$\\vartheta^{\\mathbb Q}$', annotation: '$K^{\\mathbb Q}, \\theta^{\\mathbb Q}, \\delta_0, \\delta_1$', annotationAngle: 180 },
        { id: 'S', x: 110, y: 250, kind: 'parameter', label: '$\\Sigma$', annotation: 'Tril', annotationAngle: 180 },
        { id: 'AB', x: 390, y: 360, r: 26, kind: 'mapped', label: '$A, B$', annotation: 'Riccati ODE, unrolled over $\\tau$', annotationAngle: 0 },
        { id: 'y1', x: 300, y: 250, kind: 'observed', label: '$y_{t-1}$' },
        { id: 'y2', x: 480, y: 250, kind: 'observed', label: '$y_t$' },
        { id: 'y3', x: 660, y: 250, kind: 'observed', label: '$y_{t+1}$', annotation: 'yields at $m$ maturities', annotationAngle: 0 },
        { id: 'x1', x: 300, y: 120, kind: 'stochastic', label: '$x_{t-1}$' },
        { id: 'x2', x: 480, y: 120, kind: 'stochastic', label: '$x_t$' },
        { id: 'x3', x: 660, y: 120, kind: 'stochastic', label: '$x_{t+1}$', annotation: 'latent factors', annotationAngle: 0 },
        { id: 'P', x: 480, y: 25, kind: 'parameter', label: '$\\vartheta^{\\mathbb P}$', annotation: '$\\mu, \\Phi$', annotationAngle: 0 },
    ],
    edges: [
        { from: 'q', to: 'AB' },
        { from: 'S', to: 'AB' },
        { from: 'AB', to: 'y1' },
        { from: 'AB', to: 'y2' },
        { from: 'AB', to: 'y3' },
        { from: 'x1', to: 'y1' },
        { from: 'x2', to: 'y2' },
        { from: 'x3', to: 'y3' },
        { from: 'x1', to: 'x2' },
        { from: 'x2', to: 'x3' },
        { from: 'P', to: 'x1' },
        { from: 'P', to: 'x2' },
        { from: 'P', to: 'x3' },
        { from: 'S', to: 'x1' },
    ],
});
