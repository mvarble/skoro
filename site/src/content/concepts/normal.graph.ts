import { defineGraph } from '$lib/graph/spec';

// What `add_quantity(Normal::new(μ, L))` adds, following `Normal::factor` in
// `crates/skoro-normal/src/distribution.rs`. The normalizer's chain (diag, abs, log, sum) is drawn as
// one node.
export const normal = defineGraph({
    title: 'A Gaussian quantity, as the graph stores it',
    caption:
        'Solid arrows are the parent edges the compiler follows. The log-density term $p$ scores ' +
        '$x$ (dotted), and its dashed bindings name the nodes an estimator reads back. The model ' +
        'also binds ARGUMENT to $x$, $\\mu$ and $L$, and the factor binds $L$ as ' +
        'NORMALIZER_PARENT; those edges are left out here.',
    width: 760,
    height: 430,
    nodes: [
        { id: 'mu', x: 70, y: 300, kind: 'parameter', label: '$\\mu$', annotation: 'mean' , annotationAngle: 180 },
        { id: 'L', x: 70, y: 120, kind: 'constant', label: '$L$', annotation: 'Tril', annotationAngle: 180 },
        { id: 'x', x: 230, y: 300, kind: 'stochastic', label: '$x$' },
        { id: 'W', x: 230, y: 120, kind: 'mapped', label: '$W$', annotation: '$L^{-1}$' },
        { id: 'v', x: 370, y: 300, kind: 'mapped', label: '$v$', annotation: '$x \\ominus \\mu$', annotationAngle: 90 },
        { id: 'r', x: 500, y: 230, kind: 'mapped', label: '$r$', annotation: '$W v$', annotationAngle: 60 },
        { id: 'logdet', x: 400, y: 120, kind: 'mapped', label: '$n$', annotation: '$\\log\\lvert\\det W\\rvert$', annotationAngle: -140 },
        { id: 'C', x: 400, y: 40, kind: 'mapped', label: '$C$', annotation: 'Fisher root', annotationAngle: 0 },
        { id: 'p', x: 650, y: 175, kind: 'log-likelihood', label: '$p$', annotation: '$\\log p(x \\mid \\mu, L)$', annotationAngle: -40 },
    ],
    edges: [
        // The generative model: x is drawn given μ and L.
        { from: 'mu', to: 'x' },
        { from: 'L', to: 'x' },
        // The factor's computation.
        { from: 'L', to: 'W' },
        { from: 'x', to: 'v' },
        { from: 'mu', to: 'v', via: [[230, 375]], curve: 'bezier' },
        { from: 'v', to: 'r' },
        { from: 'W', to: 'r' },
        { from: 'W', to: 'logdet' },
        { from: 'W', to: 'C', fromAngle: -60, curve: 'bezier' },
        { from: 'r', to: 'p' },
        { from: 'logdet', to: 'p' },
        // What the term scores.
        { from: 'p', to: 'x', kind: 'scores', via: [[650, 405], [230, 405]], curve: 'bezier', fromAngle: 90, toAngle: 90 },
        // Its bindings.
        { from: 'p', to: 'r', kind: 'binding', role: 'residual', bend: 30 },
        { from: 'p', to: 'C', kind: 'binding', role: 'fisher_root', bend: -25 },
        { from: 'p', to: 'L', kind: 'binding', role: 'whitening', via: [[560, 10], [110, 10]], curve: 'bezier', fromAngle: -90, toAngle: -100 },
        { from: 'p', to: 'mu', kind: 'binding', role: 'location', via: [[715, 300], [640, 418], [70, 418]], curve: 'bezier', fromAngle: 20, toAngle: 90 },
    ],
});
