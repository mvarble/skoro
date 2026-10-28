import { describe, expect, it } from 'vitest';

import { describeWalkthrough, staleMatches } from './describe';
import { frameAt, kindsUsed } from './frames';
import { defineWalkthrough, matchLines, type StepSpec, type WalkthroughSpec } from './spec';

// A constant `c` and a quantity `x` feed `y`, inside a scope; `y` folds once `x` is observed.
function walk(steps: StepSpec[], extra: Partial<WalkthroughSpec> = {}): WalkthroughSpec {
    return {
        title: 'w',
        width: 300,
        height: 200,
        nodeKinds: {
            constant: { name: 'Constant', hue: 'gray' },
            quantity: { name: 'Quantity', hue: 'orange' },
            observed: { name: 'Observed quantity', hue: 'orange', fill: 'shaded' },
            mapped: { name: 'Mapped', hue: 'blue' },
            folded: { name: 'Folded', hue: 'blue', fill: 'shaded' },
        },
        edgeKinds: { input: { name: 'input of' } },
        nodes: [
            { id: 'c', x: 50, y: 50, kind: 'constant', label: 'c' },
            { id: 'x', x: 50, y: 150, kind: 'quantity', label: '$x$' },
            { id: 'y', x: 200, y: 100, kind: 'mapped', label: 'y' },
        ],
        edges: [
            { from: 'c', to: 'y', kind: 'input' },
            { from: 'x', to: 'y', kind: 'input', id: 'xy' },
        ],
        regions: [{ id: 'scope', x: 10, y: 10, width: 280, height: 180, label: 'scope' }],
        steps,
        ...extra,
    };
}

describe('defineWalkthrough', () => {
    it('accepts a walkthrough whose steps name what it has', () => {
        const spec = walk([
            { note: 'open', show: ['scope'] },
            { note: 'build', show: ['c', 'x', 'y'], code: 'let y' },
            { note: 'fold', kind: { x: 'observed', y: 'folded' }, flow: ['xy'] },
        ]);
        expect(defineWalkthrough(spec)).toBe(spec);
    });

    it.each<[string, StepSpec, RegExp]>([
        ['an unknown id', { note: 'n', show: ['z'] }, /`show` names "z"/],
        ['an undefined kind', { note: 'n', kind: { y: 'nope' } }, /"y" becomes an undefined kind/],
        ['flow along a node', { note: 'n', flow: ['y'] }, /`flow` names "y", which is no edge/],
        ['a label on an edge', { note: 'n', label: { xy: 'q' } }, /`label` names "xy"/],
        ['an empty note', { note: ' ' }, /step 1 has no note/],
    ])('refuses %s, naming the walkthrough', (_, step, message) => {
        expect(() => defineWalkthrough(walk([step]))).toThrow(/^Walkthrough "w": /);
        expect(() => defineWalkthrough(walk([step]))).toThrow(message);
    });

    it('refuses an id that names two things, and a walkthrough with no steps', () => {
        const twice = walk([{ note: 'n' }], {
            regions: [{ id: 'x', x: 0, y: 0, width: 1, height: 1 }],
        });
        expect(() => defineWalkthrough(twice)).toThrow(/"x" names two things/);
        expect(() => defineWalkthrough(walk([]))).toThrow(/no steps/);
    });
});

describe('frameAt', () => {
    const spec = walk([
        { note: 'open', show: ['scope'] },
        { note: 'roots', show: ['c', 'x'] },
        { note: 'map', show: ['y'], emphasize: ['y'] },
        { note: 'observe', kind: { x: 'observed' }, annotate: { x: '= 1' }, mute: ['c'] },
        { note: 'fold', kind: { y: 'folded' }, label: { scope: 'snapshot' }, flow: ['xy'] },
        { note: 'drop', hide: ['xy'], unmute: ['c'] },
    ]);

    it('starts with nothing shown until a step shows it', () => {
        expect([...frameAt(spec, 0).visible]).toEqual(['scope']);
        expect(frameAt(spec, 1).visible).toEqual(new Set(['scope', 'c', 'x']));
    });

    it('draws an edge once both its nodes are drawn, until a step hides it', () => {
        expect(frameAt(spec, 2).visible).toEqual(new Set(['scope', 'c', 'x', 'y', 'c->y', 'xy']));
        expect(frameAt(spec, 5).visible.has('xy')).toBe(false);
        expect(frameAt(spec, 5).visible.has('c->y')).toBe(true);
    });

    it('carries kinds, annotations, labels and muting, but not emphasis or flow', () => {
        const folded = frameAt(spec, 4);
        const node = (id: string) => folded.graph.nodes.find((n) => n.id == id)!;
        expect(node('x').kind).toBe('observed');
        expect(node('x').annotation).toBe('= 1');
        expect(node('y').kind).toBe('folded');
        expect(folded.regionLabels.get('scope')).toBe('snapshot');
        expect(folded.muted).toEqual(new Set(['c']));
        expect(folded.flow).toEqual(new Set(['xy']));
        expect(folded.emphasized.size).toBe(0);
        expect(frameAt(spec, 2).emphasized).toEqual(new Set(['y']));
        expect(frameAt(spec, 5).muted.size).toBe(0);
    });

    it('shows everything for `all`, and clamps the step', () => {
        const all = walk([{ note: 'all', show: 'all' }]);
        expect(frameAt(all, 7).visible).toEqual(new Set(['c', 'x', 'y', 'scope', 'c->y', 'xy']));
        expect(frameAt(all, 7).step).toBe(0);
    });

    it('lists the kinds a step changes to among those the legend shows', () => {
        expect(kindsUsed(spec).nodes).toEqual(
            new Set(['constant', 'quantity', 'mapped', 'observed', 'folded']),
        );
    });
});

describe('matchLines', () => {
    const lines = ['fn f() {', '    let a = 1;', '    let b = a;', '}', 'let a = 2;'];

    it('finds every line containing a string', () => {
        expect(matchLines(lines, 'let a')).toEqual([1, 4]);
        expect(matchLines(lines, 'nowhere')).toEqual([]);
    });

    it('spans the lines from one match to the next', () => {
        expect(matchLines(lines, { from: 'fn f', to: '}' })).toEqual([0, 1, 2, 3]);
        expect(matchLines(lines, { from: 'let b', to: 'missing' })).toEqual([]);
    });
});

describe('describeWalkthrough', () => {
    it('lists each step with what appears, changes and flows', () => {
        const spec = walk(
            [
                { note: 'Build `y`.', show: ['c', 'x', 'y'], code: 'let y' },
                { note: 'Observe $x$.', kind: { x: 'observed' }, flowBack: ['xy'] },
            ],
            { caption: 'A fold.' },
        );
        expect(describeWalkthrough(spec)).toBe(
            [
                '**Walkthrough: w.** A fold.',
                '',
                '1. Build `y`. (at `let y`; appears: `c` (Constant), $x$ (Quantity), `y` (Mapped))',
                '2. Observe $x$. (becomes: $x$ → Observed quantity; flows back along: $x$ → `y`)',
            ].join('\n'),
        );
    });
});

describe('staleMatches', () => {
    it('names each code match that finds no line of the quoted code', () => {
        const spec = walk([
            { note: 'a', code: ['let y', 'let z'] },
            { note: 'b', code: 'let y' },
        ]);
        expect(staleMatches(spec, 'let x = 1;\nlet y = x;')).toEqual([
            'step 1: `let z` matches no line of the quoted code',
        ]);
    });
});
