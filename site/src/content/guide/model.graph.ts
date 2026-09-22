import { defineGraph } from '$lib/graph/spec';

// The graph built by the umbrella crate's doctest: `let c = a + b;`.
export const aPlusB = defineGraph({
    title: 'The graph behind a + b',
    caption:
        'Adding a parameter and a constant adds a third, mapped node. Nothing is computed yet: ' +
        'values flow along the parent edges only when the model is evaluated.',
    width: 360,
    height: 170,
    nodes: [
        { id: 'a', x: 70, y: 115, kind: 'parameter', label: '$a$', annotation: 'Real', annotationAngle: 180 },
        { id: 'b', x: 70, y: 45, kind: 'constant', label: '$b$', annotation: '$2.0$', annotationAngle: 180 },
        { id: 'c', x: 250, y: 80, kind: 'mapped', label: '$c$', annotation: '$a + b$', annotationAngle: 0 },
    ],
    edges: [
        { from: 'a', to: 'c' },
        { from: 'b', to: 'c' },
    ],
});
