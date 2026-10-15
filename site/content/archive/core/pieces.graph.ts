import { defineGraph } from '$lib/graph/spec';

// How the core's types refer to one another, from the description of a computation, which names no
// device, to what is made for one target when a query is compiled. Each node is a type and each
// edge a field or a method of its source. The blue types are made by the compiler for one target;
// the rest exist on no device. The cycle Operation → Lowered → Composite → Operation is the
// expansion of composites, and the cycle Program ⇄ Kernel the boundary kernels of an embedded type,
// which hold programs of its retraction and inverse.

export const pieces = defineGraph({
    title: 'How the pieces fit',
    caption: String.raw`Each node is a type of the core, and each arrow a field or a method of the type it leaves. The types in blue are made by the compiler for one target; every other commits to no device. Lowering an operation gives a kernel, or a composite that expands back into operations, until only kernels remain; a program is a list of steps, each a kernel, and the kernels of an embedded type's boundary operations hold programs of their own.`,
    width: 880,
    height: 600,
    nodeRadius: 50,
    fontSize: 16,
    nodeKinds: {
        data: { name: 'Plain data', hue: 'gray', shape: 'square' },
        description: { name: 'A computation, on no device', hue: 'green', shape: 'square' },
        model: { name: 'The model', hue: 'orange', shape: 'square' },
        compiled: { name: 'Made for one target', hue: 'blue', shape: 'square' },
    },
    edgeKinds: {
        holds: { name: 'holds' },
        returns: { name: 'a method returns', line: 'dashed' },
    },
    nodes: [
        { id: 'embedded', x: 80, y: 500, kind: 'description', label: String.raw`$\texttt{Embedded}$` },
        {
            id: 'composite',
            x: 320,
            y: 500,
            kind: 'description',
            label: String.raw`$\texttt{Composite}$`,
            annotation: 'a named subgraph',
            annotationAngle: 90,
        },
        { id: 'space', x: 80, y: 300, kind: 'data', label: String.raw`$\texttt{Space}$` },
        { id: 'operation', x: 320, y: 300, kind: 'description', label: String.raw`$\texttt{Operation}$` },
        {
            id: 'param',
            x: 490,
            y: 170,
            kind: 'data',
            label: String.raw`$\texttt{Param}$`,
            annotation: 'Integer, Float, Text, Space, List',
        },
        {
            id: 'model',
            x: 80,
            y: 80,
            kind: 'model',
            label: String.raw`$\texttt{Model}$`,
            annotation: 'in a Snapshot',
        },
        { id: 'lowered', x: 650, y: 500, kind: 'compiled', label: String.raw`$\texttt{Lowered}$` },
        {
            id: 'kernel',
            x: 770,
            y: 300,
            kind: 'compiled',
            label: String.raw`$\texttt{Kernel}$`,
            annotation: 'Zero, One or Two',
            annotationAngle: 180,
        },
        { id: 'program', x: 650, y: 80, kind: 'compiled', label: String.raw`$\texttt{Program}$` },
    ],
    edges: [
        { from: 'space', to: 'embedded', kind: 'holds', label: 'Embedded(…)' },
        { from: 'embedded', to: 'composite', kind: 'holds', label: 'R and Q' },
        { from: 'space', to: 'composite', kind: 'returns', label: 'retraction()' },
        { from: 'composite', to: 'operation', kind: 'holds', label: 'its body' },
        { from: 'operation', to: 'space', kind: 'returns', label: 'output()' },
        { from: 'operation', to: 'param', kind: 'returns', label: 'params()', labelAt: 0.6 },
        { from: 'operation', to: 'lowered', kind: 'returns', label: 'lower(…, target)' },
        { from: 'lowered', to: 'composite', kind: 'holds', label: 'Composite(…)' },
        { from: 'lowered', to: 'kernel', kind: 'holds', label: 'Kernel(…)' },
        { from: 'model', to: 'space', kind: 'holds', label: "every node's", labelAt: 0.4 },
        {
            from: 'model',
            to: 'operation',
            kind: 'holds',
            label: "every mapped node's",
            labelAt: 0.65,
        },
        { from: 'model', to: 'program', kind: 'returns', label: 'compile(query, target)' },
        { from: 'program', to: 'kernel', kind: 'holds', label: 'its steps', bend: 40 },
        {
            from: 'kernel',
            to: 'program',
            kind: 'holds',
            label: 'R and Q, if boundary',
            labelAt: 0.55,
            bend: 40,
        },
    ],
});
