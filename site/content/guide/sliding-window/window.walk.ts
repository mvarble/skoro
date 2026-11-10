import { defineWalkthrough } from '$lib/walk/spec';

// The walkthrough of the sliding window's guide page: the window's probabilistic graph over four
// ticks with a window of two poses, appending a tick, solving, and retiring the oldest pose into a
// replacement over its Markov blanket, twice: once a pose alone, once a pose with the landmark
// that only it still sees. The example keeps five poses; two keep the figure small.

const X = [110, 320, 530, 740];
const between = (k: number) => (X[k - 1]! + X[k]!) / 2;

/** The window's loop, in `examples/sliding_window.rs`. */
export const window = defineWalkthrough({
    title: 'A window of two poses, retiring as it goes',
    caption: String.raw`The window's probabilistic graph. A solid arrow runs from a quantity to a term conditioned on it, a dotted arrow from a term to the quantity it scores. The odometry $\otherRandomPoint_k$ and the sightings $\bfz_{kj}$ are observed; the poses $\randomPoint_k$ and the landmarks $\bfm_j$ are free. A replacement $\psi$ is a likelihood factor over the blanket of what it replaced.`,
    width: 860,
    height: 490,
    fontSize: 18,
    nodeKinds: {
        quantity: { name: 'Free quantity', hue: 'orange' },
        observed: { name: 'Observed quantity', hue: 'gray' },
        term: { name: 'Term scoring a quantity', hue: 'purple', shape: 'square' },
        factor: { name: 'Replacement', hue: 'purple', shape: 'square', outline: 'dashed' },
        retiring: { name: 'Retiring quantity', hue: 'red' },
        retiringObserved: { name: 'Retiring measurement', hue: 'red', fill: 'shaded' },
        joined: { name: 'Term joined to them', hue: 'purple', shape: 'square', fill: 'shaded' },
        joinedFactor: {
            name: 'Replacement joined to them',
            hue: 'purple',
            shape: 'square',
            outline: 'dashed',
            fill: 'shaded',
        },
        blanket: { name: 'Quantity in the blanket', hue: 'teal' },
    },
    edgeKinds: {
        conditions: { name: 'conditions a term' },
        scores: { name: 'scores a quantity', line: 'dotted' },
    },
    nodes: [
        // The poses, the first one's prior, and the motion terms between them.
        { id: 't0', x: 30, y: 220, kind: 'term', label: String.raw`$t_0$` },
        ...X.map((x, k) => ({
            id: `x${k}`,
            x,
            y: 220,
            kind: 'quantity',
            label: String.raw`$\randomPoint_${k}$`,
        })),
        ...[1, 2, 3].map((k) => ({
            id: `tu${k}`,
            x: between(k),
            y: 220,
            kind: 'term',
            label: String.raw`$t^u_${k}$`,
        })),
        // The odometry between the poses, and its terms.
        ...[1, 2, 3].flatMap((k) => [
            {
                id: `ty${k - 1}`,
                x: between(k),
                y: 130,
                kind: 'term',
                label: String.raw`$t^y_${k - 1}$`,
            },
            {
                id: `y${k - 1}`,
                x: between(k),
                y: 45,
                kind: 'observed',
                label: String.raw`$\otherRandomPoint_${k - 1}$`,
            },
        ]),
        // The sightings: x₀ and x₁ see m₁, x₂ and x₃ see m₂.
        ...[
            [0, 1],
            [1, 1],
            [2, 2],
            [3, 2],
        ].flatMap(([k, j]) => [
            {
                id: `tz${k}${j}`,
                x: X[k!]!,
                y: 310,
                kind: 'term',
                label: String.raw`$t^z_{${k}${j}}$`,
            },
            {
                id: `z${k}${j}`,
                x: X[k!]!,
                y: 395,
                kind: 'observed',
                label: String.raw`$\bfz_{${k}${j}}$`,
            },
        ]),
        // The landmarks and their priors.
        { id: 'm1', x: between(1), y: 440, kind: 'quantity', label: String.raw`$\bfm_1$` },
        { id: 'tm1', x: between(1) - 85, y: 455, kind: 'term', label: String.raw`$t^m_1$` },
        { id: 'm2', x: between(3), y: 440, kind: 'quantity', label: String.raw`$\bfm_2$` },
        { id: 'tm2', x: between(3) - 85, y: 455, kind: 'term', label: String.raw`$t^m_2$` },
        // The replacements, one per retirement.
        { id: 'psi1', x: between(1), y: 310, kind: 'factor', label: String.raw`$\psi_1$` },
        { id: 'psi2', x: between(2), y: 310, kind: 'factor', label: String.raw`$\psi_2$` },
    ],
    edges: [
        { from: 't0', to: 'x0', kind: 'scores' },
        ...[1, 2, 3].flatMap((k) => [
            { from: `x${k - 1}`, to: `tu${k}`, kind: 'conditions' },
            { from: `tu${k}`, to: `x${k}`, kind: 'scores' },
            { from: `x${k - 1}`, to: `ty${k - 1}`, kind: 'conditions' },
            { from: `x${k}`, to: `ty${k - 1}`, kind: 'conditions' },
            { from: `ty${k - 1}`, to: `y${k - 1}`, kind: 'scores' },
        ]),
        ...[
            [0, 1],
            [1, 1],
            [2, 2],
            [3, 2],
        ].flatMap(([k, j]) => [
            { from: `x${k}`, to: `tz${k}${j}`, kind: 'conditions' },
            { from: `m${j}`, to: `tz${k}${j}`, kind: 'conditions' },
            { from: `tz${k}${j}`, to: `z${k}${j}`, kind: 'scores' },
        ]),
        { from: 'tm1', to: 'm1', kind: 'scores' },
        { from: 'tm2', to: 'm2', kind: 'scores' },
        { from: 'x1', to: 'psi1', kind: 'conditions' },
        { from: 'm1', to: 'psi1', kind: 'conditions' },
        { from: 'x2', to: 'psi2', kind: 'conditions' },
    ],
    steps: [
        {
            note: 'Tick 0: the window appends the first pose with its prior, and its sighting of the landmark $\\bfm_1$, seen for the first time, with its prior.',
            code: 'let appended = window.build(',
            show: ['t0', 'x0', 'tz01', 'z01', 'm1', 'tm1'],
        },
        {
            note: 'Tick 1: the next pose, the motion term from the last, the odometry between them, and a second sighting of $\\bfm_1$. The odometry and the sightings are observed, at the values the whole model drew.',
            code: ['let appended = window.build(', 'live.append('],
            show: ['tu1', 'x1', 'ty0', 'y0', 'tz11', 'z11'],
        },
        {
            note: 'Tick 2: a third pose, and a new landmark $\\bfm_2$.',
            code: 'let appended = window.build(',
            show: ['tu2', 'x2', 'ty1', 'y1', 'tz22', 'z22', 'm2', 'tm2'],
        },
        {
            note: 'The window solves for its mode, from the last mode and the starts of what was just appended. It now holds three poses, more than $w = 2$.',
            code: 'let minimized = method.minimize(',
        },
        {
            note: 'The oldest pose $\\randomPoint_0$ retires, with the odometry and the sighting taken from it, observed. No landmark retires: $\\randomPoint_1$ still sees $\\bfm_1$.',
            code: 'let (addresses, observed) = live.retiring(a);',
            kind: { x0: 'retiring', y0: 'retiringObserved', z01: 'retiringObserved' },
        },
        {
            note: 'Every term joined to them goes: the prior $t_0$, the odometry and sighting terms, and the motion term $t^u_1$, which scores $\\randomPoint_1$ but is conditioned on $\\randomPoint_0$. What they involve beyond the retired quantities is the Markov blanket: $\\randomPoint_1$ and $\\bfm_1$.',
            code: 'let retirement = window.retire(addresses, observed)?;',
            kind: {
                t0: 'joined',
                tu1: 'joined',
                ty0: 'joined',
                tz01: 'joined',
                x1: 'blanket',
                m1: 'blanket',
            },
            emphasize: ['x0->tu1', 'tu1->x1', 'x1->ty0', 'm1->tz01'],
        },
        {
            note: "The retired terms, linearized at the window's mode and integrated over $\\randomPoint_0$, leave one factor $\\psi_1$ over the blanket, and the retirement puts it in their place. $\\randomPoint_1$ is now scored by nothing but is held by $\\psi_1$.",
            code: [
                'let replacing = Replacing::new(',
                'let replacement = replacing.at(&live.point)?;',
                'retirement.replace(|_| replacement.build())?;',
            ],
            hide: ['x0', 't0', 'tu1', 'ty0', 'y0', 'tz01', 'z01'],
            show: ['psi1'],
            kind: { x1: 'quantity', m1: 'quantity' },
        },
        {
            note: 'Tick 3: the window appends a pose and its sighting of $\\bfm_2$, and solves again.',
            code: ['let appended = window.build(', 'let minimized = method.minimize('],
            show: ['tu3', 'x3', 'ty2', 'y2', 'tz32', 'z32'],
        },
        {
            note: 'Now $\\randomPoint_1$ retires, and with it $\\bfm_1$, which no other pose of the window sees. A path that never returns never sights it again.',
            code: 'let (addresses, observed) = live.retiring(a);',
            kind: {
                x1: 'retiring',
                m1: 'retiring',
                y1: 'retiringObserved',
                z11: 'retiringObserved',
            },
        },
        {
            note: 'The terms joined to them include the replacement $\\psi_1$ and the prior of $\\bfm_1$. Their blanket is $\\randomPoint_2$ alone.',
            code: 'let retirement = window.retire(addresses, observed)?;',
            kind: {
                psi1: 'joinedFactor',
                tm1: 'joined',
                tu2: 'joined',
                ty1: 'joined',
                tz11: 'joined',
                x2: 'blanket',
            },
            emphasize: ['x1->tu2', 'tu2->x2', 'x2->ty1'],
        },
        {
            note: 'One factor $\\psi_2$ over $\\randomPoint_2$ replaces them all, and the window holds its last two poses again. Everything before them is in $\\psi_2$.',
            code: ['retirement.replace(|_| replacement.build())?;', 'live.retire(a);'],
            hide: ['x1', 'm1', 'tm1', 'psi1', 'tu2', 'ty1', 'y1', 'tz11', 'z11'],
            show: ['psi2'],
            kind: { x2: 'quantity' },
        },
    ],
});
