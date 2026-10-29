import { probabilistic } from '../../archive/core/model.graph';
import { defineWalkthrough } from '$lib/walk/spec';

// The walkthrough of the page on terms: the Markov blanket of the first tick, read off the
// probabilistic graph drawn above it.

/** `blanket_of_the_first_tick`, in `tests/terms.rs`. */
export const blanket = defineWalkthrough({
    ...probabilistic,
    title: 'The Markov blanket of the first tick',
    caption:
        'The probabilistic graph drawn above, asked for the blanket of $\\randomPoint_0$ and $\\bfz_0$.',
    nodeKinds: {
        ...probabilistic.nodeKinds,
        asked: { name: 'Quantity asked about', hue: 'orange', fill: 'shaded' },
        joined: { name: 'Term joined to them', hue: 'purple', shape: 'square', fill: 'shaded' },
        joinedFactor: {
            name: 'Likelihood factor joined to them',
            hue: 'purple',
            shape: 'square',
            outline: 'dashed',
            fill: 'shaded',
        },
        inBlanket: { name: 'Quantity in the blanket', hue: 'teal' },
        parameterInBlanket: { name: 'Parameter in the blanket', hue: 'teal', outline: 'dashed' },
    },
    steps: [
        {
            note: 'A sliding window that retires the first tick asks for the blanket of $\\randomPoint_0$ and $\\bfz_0$.',
            code: 'pub fn blanket',
            show: 'all',
            kind: { x0: 'asked', z0: 'asked' },
        },
        {
            note: '`terms_of` gives the terms a quantity is joined to: the one that scores it, those conditioned on it, and the factors that involve it. $\\randomPoint_0$ is joined to $t_1$, $t_3$, $t_4$ and $t_6$, and $\\bfz_0$ to $t_4$.',
            code: ['pub fn terms_of', '.flat_map(|q| self.terms_of(*q))'],
            kind: { tx0: 'joined', tx1: 'joined', tz0: 'joined', physics: 'joinedFactor' },
            emphasize: ['tx0->x0', 'x0->tx1', 'x0->tz0', 'x0->physics', 'tz0->z0'],
        },
        {
            note: "Each term's `involves` lists the quantities its log-density is computed from. Those not asked about make the blanket: $\\randomPoint_1$, $\\bfm$, $\\bfzeta_1$ and $\\bfsigma$.",
            code: ['.flat_map(|term| term.involves', '.filter(|q| !inside.contains(q))'],
            kind: {
                x1: 'inBlanket',
                m: 'inBlanket',
                zeta: 'parameterInBlanket',
                sigma: 'parameterInBlanket',
            },
            emphasize: ['zeta->tx1', 'tx1->x1', 'm->tz0', 'x1->physics', 'sigma->physics'],
        },
        {
            note: 'The other terms involve neither $\\randomPoint_0$ nor $\\bfz_0$. Given everything else, their conditional density is $t_1 + t_3 + t_4 + t_6$ up to a constant, which reads the blanket alone. `blanket` returns it in ancestral order.',
            code: 'blanket.sort_unstable();',
            mute: [
                'tm',
                'tz1',
                'z1',
                'score',
                'tm->m',
                'm->tz1',
                'x1->tz1',
                'tz1->z1',
                'sigma->score',
            ],
        },
    ],
});
