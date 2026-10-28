import { defineWalkthrough } from '$lib/walk/spec';

export const building = defineWalkthrough({
    title: 'A scope builds a graph',
    caption: 'Each line adds a node; the scope publishes them as a snapshot.',
    width: 420,
    height: 220,
    nodeKinds: {
        constant: { name: 'Constant', hue: 'gray', shape: 'square' },
        quantity: { name: 'Quantity', hue: 'orange' },
        mapped: { name: 'Mapped value', hue: 'blue' },
    },
    edgeKinds: { input: { name: 'input of' } },
    nodes: [
        { id: 'half', x: 90, y: 150, kind: 'constant', label: '½' },
        { id: 'x', x: 90, y: 70, kind: 'quantity', label: '$x$' },
        { id: 'y', x: 290, y: 110, kind: 'mapped', label: '$y$', annotation: 'mul' },
    ],
    edges: [
        { from: 'half', to: 'y', kind: 'input' },
        { from: 'x', to: 'y', kind: 'input' },
    ],
    regions: [
        { id: 'scope', x: 20, y: 20, width: 380, height: 180, label: 'scope', line: 'dashed' },
    ],
    steps: [
        { note: 'The writer opens a scope.', code: 'writer.build', show: ['scope'] },
        { note: 'A constant.', code: 'let half', show: ['half'] },
        { note: 'A quantity, at `x`.', code: 'let x', show: ['x'] },
        {
            note: 'A product of the two, with its input edges.',
            code: 'let y',
            show: ['y'],
            emphasize: ['y'],
        },
        {
            note: 'The scope closes and publishes a snapshot.',
            code: { from: '    (x, y)', to: '})?' },
            label: { scope: 'snapshot' },
            emphasize: ['scope'],
        },
    ],
});
