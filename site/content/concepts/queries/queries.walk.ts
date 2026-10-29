import type { NodeSpec } from '$lib/graph/spec';
import { defineWalkthrough, type WalkEdgeSpec } from '$lib/walk/spec';

// The walkthroughs of the page on queries: what compiling folds, how a composite is expanded, the
// passes over the program that results, the snapshot's cache of programs, and an estimator's step
// by the cached retractions.

const quantity = { name: 'Quantity', hue: 'orange' } as const;
const mapped = { name: 'Mapped value', hue: 'blue' } as const;
const free = {
    name: 'Free root: an input of the program',
    hue: 'orange',
    outline: 'dashed',
} as const;
const constantRoot = { name: 'Constant root', hue: 'gray', shape: 'square' } as const;
const step = { name: 'Step', hue: 'green' } as const;

/** `observed_values_fold_once_per_program`, in `tests/queries.rs`. */
export const folding = defineWalkthrough({
    title: 'Compiling folds what the observations fix',
    caption:
        'On the left, the compilation graph of the snapshot; on the right, the computation graph of the program compiled from it.',
    width: 720,
    height: 330,
    nodeKinds: {
        quantity,
        mapped,
        free,
        constantRoot,
        step,
        folded: {
            name: 'Folded step: a constant root',
            hue: 'gray',
            shape: 'square',
            fill: 'shaded',
        },
    },
    edgeKinds: { input: { name: 'input of' } },
    nodes: [
        { id: 'o', x: 80, y: 220, kind: 'quantity', label: '$o$' },
        { id: 'x', x: 80, y: 100, kind: 'quantity', label: '$x$' },
        { id: 'c', x: 190, y: 220, kind: 'mapped', label: '$c$', annotation: 'counted' },
        { id: 'cx', x: 200, y: 100, kind: 'mapped', label: '×' },
        { id: 'f', x: 300, y: 100, kind: 'mapped', label: '$f$', annotation: 'Σ' },
        { id: 'po', x: 440, y: 220, kind: 'constantRoot', label: '$o$' },
        { id: 'px', x: 440, y: 100, kind: 'free', label: '$x$' },
        { id: 'pc', x: 550, y: 220, kind: 'step', label: '$c$', annotation: 'counted' },
        { id: 'pcx', x: 560, y: 100, kind: 'step', label: '×' },
        { id: 'pf', x: 660, y: 100, kind: 'step', label: '$f$', annotation: 'Σ' },
    ],
    edges: [
        { from: 'o', to: 'c', kind: 'input' },
        { from: 'c', to: 'cx', kind: 'input' },
        { from: 'x', to: 'cx', kind: 'input' },
        { from: 'cx', to: 'f', kind: 'input' },
        { from: 'po', to: 'pc', kind: 'input' },
        { from: 'pc', to: 'pcx', kind: 'input' },
        { from: 'px', to: 'pcx', kind: 'input' },
        { from: 'pcx', to: 'pf', kind: 'input' },
    ],
    regions: [
        { id: 'snapshot', x: 20, y: 20, width: 330, height: 290, label: 'snapshot', hue: 'green' },
        {
            id: 'program',
            x: 380,
            y: 20,
            width: 320,
            height: 290,
            label: 'program',
            hue: 'sky',
            line: 'dashed',
        },
    ],
    steps: [
        {
            note: '`counted` is the identity, with a kernel that counts the times it runs.',
            code: 'let (operation, calls) = counted();',
        },
        {
            note: String.raw`A model in which $c$ is computed from the quantity $o$ alone, and $f = \Sigma(c \odot x)$.`,
            code: { from: 'let (o, f) = writer', to: '.unwrap();' },
            show: ['snapshot', 'o', 'x', 'c', 'cx', 'f'],
        },
        {
            note: 'The observations give $o$ a value. They belong to the query: the model does not change.',
            code: 'let observed = Observations::new().observe(o',
            annotate: { o: 'observed' },
            emphasize: ['o'],
        },
        {
            note: 'Compiling walks back from the target $f$ through the compilation graph, and reaches every node $f$ is computed from.',
            code: '.compile(&Query::of([f]).given(observed)',
            emphasize: ['f', 'cx', 'c', 'x', 'o'],
        },
        {
            note: 'Each quantity reached becomes a root of the program. $o$ is observed, so it is a constant, loaded on the target; $x$ is not, so it is free: the passes take a value of it, and differentiate with respect to it.',
            code: '.compile(&Query::of([f]).given(observed)',
            show: ['program', 'po', 'px'],
        },
        {
            note: "Each operation reached is lowered to a kernel for the target, and becomes a step reading its inputs' slots.",
            code: '.compile(&Query::of([f]).given(observed)',
            show: ['pc', 'pcx', 'pf'],
        },
        {
            note: "Every input of $c$'s step is a constant, so the step folds: its kernel runs now, once, and its output becomes a constant root.",
            code: '"once, when the program is compiled"',
            kind: { pc: 'folded' },
            annotate: { pc: 'counted: ran once' },
            flow: ['po->pc'],
        },
        {
            note: "No step reads $o$'s value now, so the prune drops it: a program keeps what its targets need, and its free roots.",
            code: '"once, when the program is compiled"',
            hide: ['po'],
        },
        {
            note: 'Each evaluation runs the steps that are left, from the value of $x$ it is given and the folded constant.',
            code: 'program.forward(&at)',
            flow: ['pc->pcx', 'px->pcx', 'pcx->pf'],
        },
        {
            note: 'Two evaluations later, the counted kernel has still run once.',
            code: '"and never again"',
            emphasize: ['pc'],
        },
    ],
});

// The model of `tests/passes.rs`: `f`, a call of the composite `mix`, and `g = x ⊙ y + sin y`.
// The program's nodes are shared by the expansion and the passes; the passes draw them alone,
// shifted left.
const programNodes: NodeSpec[] = [
    { id: 'ps', x: 350, y: 500, kind: 'free', label: '$s$' },
    { id: 'px', x: 350, y: 360, kind: 'free', label: '$x$' },
    { id: 'py', x: 350, y: 180, kind: 'free', label: '$y$' },
    { id: 'sx', x: 440, y: 420, kind: 'step', label: 'sin' },
    { id: 'xx', x: 440, y: 350, kind: 'step', label: '×' },
    { id: 'p1', x: 440, y: 280, kind: 'step', label: '×' },
    { id: 'ssx', x: 520, y: 490, kind: 'step', label: '·' },
    { id: 'e', x: 520, y: 280, kind: 'step', label: 'exp' },
    { id: 'a1', x: 595, y: 400, kind: 'step', label: '+' },
    { id: 'a2', x: 660, y: 330, kind: 'step', label: '+' },
    { id: 'pf', x: 725, y: 400, kind: 'step', label: '$f$', annotation: 'Σ', annotationAngle: 90 },
    { id: 'm2', x: 440, y: 190, kind: 'step', label: '×' },
    { id: 'sy', x: 440, y: 95, kind: 'step', label: 'sin' },
    { id: 'pg', x: 540, y: 140, kind: 'step', label: '$g$', annotation: '+' },
];

const programEdges: WalkEdgeSpec[] = [
    { from: 'px', to: 'p1', kind: 'input' },
    { from: 'py', to: 'p1', kind: 'input' },
    { from: 'p1', to: 'e', kind: 'input' },
    { from: 'px', to: 'sx', kind: 'input' },
    { from: 'sx', to: 'ssx', kind: 'input' },
    { from: 'ps', to: 'ssx', kind: 'input' },
    { from: 'e', to: 'a1', kind: 'input' },
    { from: 'ssx', to: 'a1', kind: 'input' },
    { from: 'px', to: 'xx', kind: 'input' },
    { from: 'a1', to: 'a2', kind: 'input' },
    { from: 'xx', to: 'a2', kind: 'input' },
    { from: 'a2', to: 'pf', kind: 'input' },
    { from: 'px', to: 'm2', kind: 'input' },
    { from: 'py', to: 'm2', kind: 'input' },
    { from: 'py', to: 'sy', kind: 'input' },
    { from: 'm2', to: 'pg', kind: 'input' },
    { from: 'sy', to: 'pg', kind: 'input' },
];

const body = ['p1', 'e', 'sx', 'ssx', 'xx', 'a1', 'a2', 'pf'];

/** The steps expanded from `f`, shifted left by `dx`. */
const mixRegion = (dx: number, label: string) =>
    ({
        id: 'mix',
        x: 408 - dx,
        y: 245,
        width: 344,
        height: 295,
        label,
        hue: 'purple',
        line: 'dashed',
    }) as const;

/** The model of `tests/passes.rs`, compiled for `[f, g]`. */
export const expansion = defineWalkthrough({
    title: 'A composite is expanded into the program',
    caption:
        'The body of `mix` is $\\Sigma(e^{x \\odot y} + s \\sin x + x \\odot x)$. On the left, the compilation graph; on the right, the computation graph of the program for $[f, g]$.',
    width: 760,
    height: 560,
    nodeKinds: {
        quantity,
        mapped,
        call: { name: 'A call of a composite', hue: 'blue', outline: 'dashed' },
        free,
        step,
    },
    edgeKinds: { input: { name: 'input of' } },
    nodes: [
        { id: 's', x: 60, y: 480, kind: 'quantity', label: '$s$' },
        { id: 'x', x: 60, y: 350, kind: 'quantity', label: '$x$' },
        { id: 'y', x: 60, y: 190, kind: 'quantity', label: '$y$' },
        {
            id: 'f',
            x: 200,
            y: 440,
            kind: 'call',
            label: '$f$',
            annotation: 'mix.call',
            annotationAngle: 90,
        },
        { id: 'm', x: 170, y: 260, kind: 'mapped', label: '×' },
        { id: 'sin', x: 170, y: 120, kind: 'mapped', label: 'sin' },
        { id: 'g', x: 245, y: 190, kind: 'mapped', label: '$g$', annotation: '+' },
        ...programNodes,
    ],
    edges: [
        { from: 'x', to: 'f', kind: 'input' },
        { from: 'y', to: 'f', kind: 'input' },
        { from: 's', to: 'f', kind: 'input' },
        { from: 'x', to: 'm', kind: 'input' },
        { from: 'y', to: 'm', kind: 'input' },
        { from: 'y', to: 'sin', kind: 'input' },
        { from: 'm', to: 'g', kind: 'input' },
        { from: 'sin', to: 'g', kind: 'input' },
        ...programEdges,
    ],
    regions: [
        { id: 'snapshot', x: 16, y: 16, width: 270, height: 530, label: 'snapshot', hue: 'green' },
        {
            id: 'program',
            x: 300,
            y: 16,
            width: 456,
            height: 530,
            label: 'program for [f, g]',
            hue: 'sky',
            line: 'dashed',
        },
        mixRegion(0, 'from $f$, a call of `mix`'),
    ],
    steps: [
        {
            note: '`mix` defines a composite: a named subgraph, with three inputs and one output, built once by the code a model would use.',
            code: { from: 'fn mix()', to: '.unwrap()' },
        },
        {
            note: 'A model of three quantities.',
            code: ['let x = quantity', 'let y = quantity', 'let s = quantity'],
            show: ['snapshot', 'x', 'y', 's'],
        },
        {
            note: 'A call of `mix` is one node of the model: the snapshot holds the call, and the composite holds its body.',
            code: 'let f = mix.call',
            show: ['f'],
        },
        {
            note: '$g$ is built of primitives, a node for each.',
            code: 'let g = x * y + y.sin();',
            show: ['m', 'sin', 'g'],
        },
        {
            note: 'The tests compile the query of $[f, g]$, which walks back from both targets; each quantity it reaches is free.',
            show: ['program', 'px', 'py', 'ps'],
        },
        {
            note: "$f$'s operation lowers to its composite, which is expanded in place: the body's operations become steps of the program, reading the slots of the call's arguments. Each step is attributed to $f$, so that a pass which refuses one names the node that was written.",
            code: 'let f = mix.call',
            show: ['mix', ...body],
            emphasize: ['mix'],
        },
        {
            note: "$g$'s operations lower to a kernel each. Its product $x \\odot y$ is a step apart from the one in `mix`: they are two nodes, and each node is emitted once.",
            code: 'let g = x * y + y.sin();',
            show: ['m2', 'sy', 'pg'],
        },
        {
            note: 'In the program, $x$ fans out to four steps: three from the body of `mix`, one from $g$.',
            emphasize: ['px', 'px->p1', 'px->sx', 'px->xx', 'px->m2'],
        },
    ],
});

const shift = 290;
const forward = programEdges.map((e) => e.id ?? `${e.from}->${e.to}`);

/** The passes of `Program`, over the program of the expansion. */
export const passes = defineWalkthrough({
    title: 'The forward passes, and the cotangent pass back',
    caption:
        'The program for $[f, g]$ drawn above. The cotangent pass runs over its backpropagation graph: the same nodes, every edge reversed.',
    width: 470,
    height: 560,
    nodeKinds: { free, step },
    edgeKinds: { input: { name: 'input of' } },
    nodes: programNodes.map((node) => ({ ...node, x: node.x - shift })),
    edges: programEdges,
    regions: [mixRegion(shift, 'from $f$')],
    steps: [
        {
            note: "`forward` runs every step once, in order, from the free values it is given to the targets' values; a value is dropped after the last step that reads it, unless it is a target. `tangent` runs along the same edges, each step's kernel pushing its inputs' tangent frames forward to its output as well.",
            code: ['pub fn forward', 'pub fn tangent'],
            show: 'all',
            flow: forward,
        },
        {
            note: 'A cotangent pass is seeded with a cotangent frame at each target: here, at $f$ and $g$.',
            code: 'pub fn cotangent',
            emphasize: ['pf', 'pg'],
        },
        {
            note: "It runs the program forward, recording each step's pull-back, and then pulls the seeds back along every edge, against the forward order.",
            code: 'pub fn cotangent',
            flowBack: forward,
        },
        {
            note: "Where a value fans out, the pull-backs along its edges add: $x$'s cotangent is the sum of four. The pass gives a cotangent at every free continuous quantity, in its tangent chart.",
            code: 'pub fn cotangent',
            emphasize: ['px'],
            flowBack: ['px->p1', 'px->sx', 'px->xx', 'px->m2'],
        },
        {
            note: "A program whose one target is a `Real`, such as a log-density, also has `gradient`: the cotangent pass seeded with one. This program has two targets, so its seeds are the caller's.",
            code: 'pub fn gradient',
        },
    ],
});

/** `same_observations_reuse_the_program`, in `tests/queries.rs`. */
export const cache = defineWalkthrough({
    title: "The snapshot's cache of programs",
    caption:
        "A program is kept under its key: the query's targets, the identity of its set of observations, and the target.",
    width: 720,
    height: 320,
    nodeKinds: {
        set: { name: 'A set of observations', hue: 'amber' },
        key: { name: 'A key', hue: 'gray', shape: 'square' },
        program: { name: 'A program', hue: 'sky', shape: 'square' },
    },
    edgeKinds: {
        keys: { name: 'keys' },
        holds: { name: 'is kept with', hue: 'sky' },
    },
    nodes: [
        { id: 'observed', x: 100, y: 220, kind: 'set', label: '$i$', annotation: 'observed' },
        { id: 'separately', x: 100, y: 90, kind: 'set', label: '$j$', annotation: 'separately' },
        {
            id: 'k1',
            x: 340,
            y: 220,
            kind: 'key',
            label: '$k_1$',
            annotation: '(density, $i$, host)',
        },
        {
            id: 'k2',
            x: 340,
            y: 90,
            kind: 'key',
            label: '$k_2$',
            annotation: '(density, $j$, host)',
        },
        { id: 'a', x: 580, y: 220, kind: 'program', label: '$a$', annotation: '$= b$' },
        { id: 'c', x: 580, y: 90, kind: 'program', label: '$c$' },
    ],
    edges: [
        { from: 'observed', to: 'k1', kind: 'keys' },
        { from: 'separately', to: 'k2', kind: 'keys' },
        { from: 'k1', to: 'a', kind: 'holds' },
        { from: 'k2', to: 'c', kind: 'holds' },
    ],
    regions: [
        {
            id: 'cache',
            x: 250,
            y: 20,
            width: 450,
            height: 280,
            label: "the snapshot's programs: the most recently used last, 32 at most",
            hue: 'green',
        },
    ],
    steps: [
        {
            note: "The running example; its writer's snapshot keeps the programs compiled from it.",
            code: 'let e = example();',
            show: ['cache'],
        },
        {
            note: 'Observing a quantity extends the empty set into a new set, with a fresh identity: $i$.',
            code: 'let observed = Observations::new().observe',
            show: ['observed'],
        },
        {
            note: 'A clone is the same set, and shares its identity.',
            code: 'let query = Query::log_density().given(observed.clone());',
            emphasize: ['observed'],
        },
        {
            note: "`compile` looks up the key of the query's targets, its set's identity and the target. Nothing is kept under it, so `compile` compiles the program and keeps it there.",
            code: 'e.compile(&query),',
            show: ['k1', 'a'],
        },
        {
            note: 'The same targets, the same identity and the same target make the same key: the same program comes back, and becomes the most recently used.',
            code: 'e.compile(&Query::log_density().given(observed)),',
            flow: ['observed->k1', 'k1->a'],
            emphasize: ['a'],
        },
        {
            note: 'A set built apart from the first, with the same value, has an identity of its own: $j$.',
            code: 'let separately',
            show: ['separately'],
        },
        {
            note: 'So its key is another, and `compile` compiles a second program. The cache compares identities, never values, which is why a caller keeps one set and passes it to every compile.',
            code: 'let c = e.compile(',
            show: ['k2', 'c'],
        },
        {
            note: 'When a 33rd program is kept, the least recently used one is dropped.',
            emphasize: ['cache'],
        },
    ],
});

/** `flat_step_is_the_product_retraction`, in `tests/queries.rs`. */
export const retraction = defineWalkthrough({
    title: "An estimator's step, by cached retractions",
    caption: String.raw`The free quantities $x \in \bbR^2$, $p \in \positive$ and $y \in \bbR^{2 \times 1}$, moved together along one tangent vector $t \in \bbR^5$.`,
    width: 760,
    height: 460,
    nodeKinds: {
        quantity: { name: "A quantity's value", hue: 'orange' },
        tangent: { name: 'A tangent vector', hue: 'amber' },
        point: { name: 'A point of the product', hue: 'orange', fill: 'shaded' },
        composite: { name: 'A retraction, as a composite', hue: 'purple', shape: 'square' },
        program: { name: 'Its program, compiled for a target', hue: 'sky', shape: 'square' },
    },
    edgeKinds: {
        input: { name: 'input of' },
        calls: { name: 'calls', hue: 'purple', line: 'dashed' },
        compiles: { name: 'is compiled into', hue: 'sky', line: 'dotted' },
    },
    nodes: [
        { id: 'x', x: 70, y: 380, kind: 'quantity', label: '$x$' },
        { id: 'p', x: 70, y: 300, kind: 'quantity', label: '$p$' },
        { id: 'y', x: 70, y: 220, kind: 'quantity', label: '$y$' },
        { id: 'point', x: 200, y: 300, kind: 'point', label: '·', annotation: 'stacked' },
        { id: 't', x: 200, y: 190, kind: 'tangent', label: '$t$' },
        {
            id: 'R',
            x: 380,
            y: 250,
            kind: 'program',
            label: 'R',
            annotation: 'host',
            annotationAngle: 0,
        },
        { id: 'moved', x: 540, y: 300, kind: 'point', label: '·', annotation: 'moved' },
        { id: 'x2', x: 680, y: 380, kind: 'quantity', label: "$x'$" },
        { id: 'p2', x: 680, y: 300, kind: 'quantity', label: "$p'$" },
        { id: 'y2', x: 680, y: 220, kind: 'quantity', label: "$y'$" },
        {
            id: 'product',
            x: 230,
            y: 60,
            annotationAngle: 90,
            kind: 'composite',
            label: '⊕',
            annotation: 'product',
        },
        {
            id: 'vector',
            x: 400,
            y: 60,
            annotationAngle: 90,
            kind: 'composite',
            label: '⊕',
            annotation: 'Vector(2)',
        },
        {
            id: 'positive',
            x: 530,
            y: 60,
            annotationAngle: 90,
            kind: 'composite',
            label: '⊕',
            annotation: 'Positive(1)',
        },
        {
            id: 'matrix',
            x: 660,
            y: 60,
            annotationAngle: 90,
            kind: 'composite',
            label: '⊕',
            annotation: 'Matrix(2, 1)',
        },
    ],
    edges: [
        { from: 'x', to: 'point', kind: 'input' },
        { from: 'p', to: 'point', kind: 'input' },
        { from: 'y', to: 'point', kind: 'input' },
        { from: 'point', to: 'R', kind: 'input' },
        { from: 't', to: 'R', kind: 'input' },
        { from: 'R', to: 'moved', kind: 'input' },
        { from: 'moved', to: 'x2', kind: 'input' },
        { from: 'moved', to: 'p2', kind: 'input' },
        { from: 'moved', to: 'y2', kind: 'input' },
        { from: 'product', to: 'vector', kind: 'calls' },
        { from: 'product', to: 'positive', kind: 'calls', via: [[400, 26]], curve: 'bezier' },
        { from: 'product', to: 'matrix', kind: 'calls', via: [[470, 18]], curve: 'bezier' },
        { from: 'product', to: 'R', kind: 'compiles', via: [[345, 115]], curve: 'bezier' },
    ],
    regions: [
        {
            id: 'global',
            x: 140,
            y: 12,
            width: 600,
            height: 140,
            label: 'every retraction, by space and map',
            hue: 'purple',
            line: 'dashed',
        },
        {
            id: 'layout',
            x: 315,
            y: 175,
            width: 130,
            height: 130,
            label: "the layout's programs",
            hue: 'sky',
            line: 'dashed',
        },
    ],
    steps: [
        {
            note: 'A model of three free quantities, of three spaces, and a program of a function of them.',
            code: ['let (x, p, y, f) = writer', '.compile(&Query::of([f.untyped()]), &target)'],
            show: ['x', 'p', 'y'],
        },
        {
            note: "The program's layout: its free continuous quantities, in ancestral order, as one point of the product of their spaces, of dimension 5.",
            code: 'let layout = program.layout();',
            emphasize: ['x', 'p', 'y'],
        },
        {
            note: "An estimator's step is a tangent vector of that dimension, the blocks of the quantities laid end to end.",
            code: 'let t = [0.1',
            show: ['t'],
        },
        {
            note: "`retract` stacks the quantities' values into one point of the product.",
            code: '.retract(&at, &Points::from_host(&t',
            show: ['point'],
        },
        {
            note: "It needs the product's retraction. `Space::retraction` keeps one composite for each space and map, for the life of the process: the first time, it defines the product's, whose body slices each factor out and calls that factor's retraction, so those are defined and kept too.",
            code: '.retract(&at, &Points::from_host(&t',
            show: ['global', 'product', 'vector', 'positive', 'matrix'],
        },
        {
            note: 'The layout compiles the composite for the target, once, and keeps the program; the calls of the factors are expanded into it.',
            code: '.retract(&at, &Points::from_host(&t',
            show: ['layout', 'R'],
            flow: ['product->R'],
        },
        {
            note: 'The program moves the point along $t$, and the result is sliced back into the quantities.',
            code: '.retract(&at, &Points::from_host(&t',
            show: ['moved', 'x2', 'p2', 'y2'],
            flow: ['point->R', 't->R', 'R->moved', 'moved->x2', 'moved->p2', 'moved->y2'],
        },
        {
            note: String.raw`Factor by factor: a Euclidean factor adds its block of $t$, and the positive one is multiplied by the exponential of its block, $2 e^{-0.3}$.`,
            code: [
                'assert_eq!(host(&step, x)',
                'assert_eq!(host(&step, p)',
                'assert_eq!(host(&step, y)',
            ],
            annotate: { x2: '[0.6, −0.8]', p2: String.raw`$2e^{-0.3}$`, y2: '[1.25, 1]' },
        },
        {
            note: 'The next step, on the same target, finds the composite and its program both kept, and compiles nothing.',
            emphasize: ['product', 'R'],
        },
    ],
});

/** The model of `tests/order.rs`, and the passes its tests ask of it. */
export const refusal = defineWalkthrough({
    title: 'Which steps a pass is on',
    caption:
        '$f = \\Sigma(S(y) + x \\odot x)$ and $g = \\Sigma x$, where $S$ squares its input by a kernel of the order a test chooses. A muted step is off the pass, and runs forward alone.',
    width: 600,
    height: 340,
    nodeKinds: {
        free,
        step,
        refused: { name: 'Refused: its kernel falls short', hue: 'red' },
    },
    edgeKinds: { input: { name: 'input of' } },
    nodes: [
        { id: 'x', x: 70, y: 220, kind: 'free', label: '$x$' },
        { id: 'y', x: 70, y: 80, kind: 'free', label: '$y$' },
        { id: 'xx', x: 220, y: 170, kind: 'step', label: '×' },
        { id: 's', x: 220, y: 80, kind: 'step', label: '$S$', annotation: 'order 1' },
        { id: 'a', x: 370, y: 125, kind: 'step', label: '+' },
        { id: 'f', x: 510, y: 125, kind: 'step', label: '$f$', annotation: 'Σ' },
        {
            id: 'g',
            x: 370,
            y: 285,
            kind: 'step',
            label: '$g$',
            annotation: 'Σ',
            annotationAngle: 0,
        },
    ],
    edges: [
        { from: 'x', to: 'xx', kind: 'input' },
        { from: 'x', to: 'g', kind: 'input' },
        { from: 'y', to: 's', kind: 'input' },
        { from: 'xx', to: 'a', kind: 'input' },
        { from: 's', to: 'a', kind: 'input' },
        { from: 'a', to: 'f', kind: 'input' },
    ],
    steps: [
        {
            note: 'The first test compiles the program for $[f, g]$, with $S$ of order one: five steps over two free quantities.',
            code: ['fn model(', 'let program = m.program(&[m.f, m.g]);'],
            show: 'all',
        },
        {
            note: 'A Hessian pass seeds velocities at $x$ and $y$ and a cotangent at $g$ alone. From the free quantities, every step is reached.',
            code: 'let (seeded, targets) =',
            flow: ['x->xx', 'x->g', 'y->s', 'xx->a', 's->a', 'a->f'],
        },
        {
            note: 'Back from the target the pass seeds, only $g$ is reached, and $x$ through it.',
            code: 'let (seeded, targets) =',
            emphasize: ['g'],
            flowBack: ['x->g'],
        },
        {
            note: "A step is on the pass when it is reached both ways: here $g$'s sum alone. $S$ runs forward only, which asks order zero of it, so its kernel of order one holds nothing back and the pass is admitted.",
            code: 'assert!(program.admits(&pass).is_ok());',
            mute: ['xx', 's', 'a', 'f'],
        },
        {
            note: 'Another test compiles $f$ alone, with $S$ of order zero. A forward pass asks order zero of every step, and runs.',
            code: ['fn zero_order_step_on_the_pass', 'program.forward(&m.at()).is_ok()'],
            hide: ['g'],
            unmute: ['xx', 's', 'a', 'f'],
            annotate: { s: 'order 0' },
            flow: ['x->xx', 'y->s', 'xx->a', 's->a', 'a->f'],
        },
        {
            note: 'A gradient seeds $f$. Every step is reached from a free quantity and reaches $f$, so every step is on the pass, and each needs order one.',
            code: 'program.gradient(&m.at()).unwrap_err()',
            emphasize: ['xx', 's', 'a', 'f'],
            flowBack: ['x->xx', 'y->s', 'xx->a', 's->a', 'a->f'],
        },
        {
            note: '$S$ has order zero, so the pass is refused before any kernel runs, with `OrderTooLow` naming its operation, the composites it was expanded from (none here) and its node.',
            code: ['program.gradient(&m.at()).unwrap_err()', 'operation: "test.square0"'],
            kind: { s: 'refused' },
        },
        {
            note: 'With $S$ of order one, a Hessian pass with a velocity at $x$ asks order two of the steps fed from $x$, and order one of $S$, which is fed from $y$ alone: it is admitted.',
            code: '.hessian_products(&m.at(), &m.velocity(m.x))',
            kind: { s: 'step' },
            annotate: { s: 'order 1' },
            emphasize: ['xx', 'a', 'f'],
            flow: ['x->xx', 'xx->a', 'a->f'],
        },
        {
            note: 'With the velocity at $y$ instead, $S$ is fed from it and needs order two, so the pass is refused, naming `test.square1`.',
            code: ['.hessian_products(&m.at(), &m.velocity(m.y))', 'operation: "test.square1"'],
            kind: { s: 'refused' },
            flow: ['y->s', 's->a', 'a->f'],
        },
    ],
});
