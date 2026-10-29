import { describe, expect, it } from 'vitest';

import { cycles, layoutReadingOrder, ROW } from './layout';
import { defineReadingOrder, itemText, type ReadingOrderSpec } from './spec';

// a ← b ← c, with c also using a directly and a using c ahead of it: one cycle of all three.
function threeModules(extra: Partial<ReadingOrderSpec> = {}): ReadingOrderSpec {
    return {
        title: 't',
        modules: [{ id: 'a' }, { id: 'b' }, { id: 'c' }],
        uses: [
            { from: 'b', to: 'a', items: ['A'] },
            { from: 'c', to: 'b', items: ['B'] },
            { from: 'c', to: 'a', items: ['A'] },
            { from: 'a', to: 'c', items: ['make', 'C'] },
        ],
        ...extra,
    };
}

const use = (layout: ReturnType<typeof layoutReadingOrder>, from: string, to: string) =>
    layout.uses.find((u) => u.from == from && u.to == to)!;

describe('layoutReadingOrder', () => {
    it('draws uses of earlier modules on the left and of later ones on the right', () => {
        const layout = layoutReadingOrder(threeModules());
        const ahead = use(layout, 'a', 'c');
        const back = use(layout, 'b', 'a');
        expect(ahead.later).toBe(true);
        expect(back.later).toBe(false);
        expect(ahead.d.startsWith(`M${layout.laterX} `)).toBe(true);
        expect(back.d.startsWith(`M${layout.dotX} `)).toBe(true);
    });

    it('marks a use of an earlier module that a longer chain implies', () => {
        const layout = layoutReadingOrder(threeModules());
        expect(use(layout, 'c', 'a').implied).toBe(true);
        expect(use(layout, 'c', 'b').implied).toBe(false);
        expect(use(layout, 'a', 'c').implied).toBe(false);
    });

    it('names the items of later modules beside the module that uses them', () => {
        const layout = layoutReadingOrder(threeModules());
        expect(layout.rows.map((r) => r.later)).toEqual(['make(), C', '', '']);
    });

    it('bands each cycle, one band to each run of its modules', () => {
        const whole = layoutReadingOrder(threeModules());
        expect(whole.bands).toHaveLength(1);
        expect(whole.bands[0]!.size).toBe(3);
        expect(whole.bands[0]!.height).toBe(3 * ROW - 4);

        const split = layoutReadingOrder({
            title: 't',
            modules: [{ id: 'a' }, { id: 'x' }, { id: 'b' }],
            uses: [
                { from: 'a', to: 'b', items: ['B'] },
                { from: 'b', to: 'a', items: ['A'] },
            ],
        });
        expect(split.bands.map((b) => b.size)).toEqual([2, 2]);
    });

    it('sets the modules set apart below a gap', () => {
        const layout = layoutReadingOrder(
            threeModules({ modules: [{ id: 'a' }, { id: 'b' }, { id: 'c', apart: true }] }),
        );
        const [a, b, c] = layout.rows;
        expect(b!.y - a!.y).toBe(ROW);
        expect(c!.y - b!.y).toBeGreaterThan(ROW);
    });
});

describe('cycles', () => {
    it('finds the strongly connected components of more than one module', () => {
        const found = cycles(
            ['a', 'b', 'c', 'd'],
            [
                { from: 'a', to: 'b', items: [] },
                { from: 'b', to: 'a', items: [] },
                { from: 'c', to: 'a', items: [] },
            ],
        );
        expect(found.map((c) => [...c].sort())).toEqual([['a', 'b']]);
    });
});

describe('defineReadingOrder', () => {
    it('refuses a use of a module it does not list', () => {
        expect(() =>
            defineReadingOrder({
                title: 'Core',
                modules: [{ id: 'a' }],
                uses: [{ from: 'a', to: 'z', items: [] }],
            }),
        ).toThrow(/Core.*`z`/);
    });

    it('refuses a module set apart before one that is not', () => {
        expect(() =>
            defineReadingOrder({
                title: 'Core',
                modules: [{ id: 'a', apart: true }, { id: 'b' }],
                uses: [],
            }),
        ).toThrow(/set apart/);
    });
});

describe('itemText', () => {
    it('marks functions and methods with parentheses', () => {
        expect(
            ['Space', 'narrow', 'Lowering.compile', '.compile', 'Space::Real'].map(itemText),
        ).toEqual(['Space', 'narrow()', 'Lowering.compile()', '.compile()', 'Space::Real']);
    });
});
