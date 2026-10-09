import { describe, expect, it } from 'vitest';

import { describeGraph } from './describe';
import { alongPolyline, boundaryPoint, layoutGraph } from './geometry';
import type { EdgeSpec, GraphSpec } from './spec';

const close = (actual: number, expected: number) => expect(actual).toBeCloseTo(expected, 2);

/** Parses the numbers out of an SVG path, as [x, y] pairs. */
const pathPoints = (d: string) =>
    [...d.matchAll(/(-?[\d.]+),(-?[\d.]+)/g)].map((m) => [Number(m[1]), Number(m[2])]);
const arrowTip = (points: string) => points.split(' ')[0].split(',').map(Number);

// Two nodes on a horizontal line in a 200×100 viewport: `a` at (50, 50), `b` at (150, 50).
function twoNodes(edge: Partial<EdgeSpec> = {}, extra: Partial<GraphSpec> = {}): GraphSpec {
    return {
        title: 't',
        width: 200,
        height: 100,
        nodes: [
            { id: 'a', x: 50, y: 50, kind: 'parameter' },
            { id: 'b', x: 150, y: 50, kind: 'mapped' },
        ],
        edges: [{ from: 'a', to: 'b', ...edge }],
        nodeRadius: 10,
        arrowSize: 4,
        ...extra,
    };
}

describe('boundaryPoint', () => {
    it('measures angles counter-clockwise from +X', () => {
        const [x, y] = boundaryPoint({ x: 0, y: 0, r: 2, shape: 'circle' }, 90);
        close(x, 0);
        close(y, 2);
    });

    it('reaches the corner of a square along its diagonal', () => {
        const [x, y] = boundaryPoint({ x: 0, y: 0, r: 1, shape: 'square' }, 45);
        close(x, 1);
        close(y, 1);
    });
});

describe('layoutGraph', () => {
    it('flips +Y up into SVG coordinates', () => {
        const spec = twoNodes();
        spec.nodes[0].y = 80;
        const [a] = layoutGraph(spec).nodes;
        expect(a.cy).toBe(20);
    });

    it('runs a straight edge between the boundaries, with the arrow tip on the target', () => {
        const [edge] = layoutGraph(twoNodes()).edges;
        const [start, end] = pathPoints(edge.d);
        expect(start).toEqual([60, 50]);
        // The line stops at the arrowhead's base; the tip is on `b`'s boundary.
        expect(end).toEqual([136, 50]);
        expect(arrowTip(edge.arrows[0])).toEqual([140, 50]);
    });

    it('honours explicit endpoint angles', () => {
        const [edge] = layoutGraph(twoNodes({ fromAngle: 90, toAngle: 90, arrow: 'none' })).edges;
        const [start, end] = pathPoints(edge.d);
        // (50, 60) and (150, 60) in math coordinates are y = 40 in SVG.
        expect(start).toEqual([50, 40]);
        expect(end).toEqual([150, 40]);
    });

    it('bends symmetrically, to the left of travel, as a Bézier curve', () => {
        const [edge] = layoutGraph(twoNodes({ bend: 90, arrow: 'none' })).edges;
        expect(edge.d).toContain('C');
        const [start, , , end] = pathPoints(edge.d);
        // Leaving `a` straight up and entering `b` from straight up: both at SVG y = 40.
        expect(start).toEqual([50, 40]);
        expect(end).toEqual([150, 40]);
    });

    it('aims a default endpoint at the nearest via point', () => {
        const [edge] = layoutGraph(twoNodes({ via: [[50, 90]], arrow: 'none' })).edges;
        const [start, via, end] = pathPoints(edge.d);
        expect(start).toEqual([50, 40]); // straight up from `a`
        expect(via).toEqual([50, 10]);
        // From `b` toward (50, 90): up and to the left.
        const angle = Math.atan2(90 - 50, 50 - 150);
        close(end[0], 150 + 10 * Math.cos(angle));
        close(end[1], 100 - (50 + 10 * Math.sin(angle)));
    });

    it('puts an arrowhead at each end when asked', () => {
        const [edge] = layoutGraph(twoNodes({ arrow: 'both' })).edges;
        expect(edge.arrows.map(arrowTip)).toEqual([
            [140, 50],
            [60, 50],
        ]);
    });

    it('places a label along the edge by length', () => {
        const [edge] = layoutGraph(twoNodes({ label: 'x', arrow: 'none' })).edges;
        expect(edge.label).toEqual({ x: 100, y: 50 });
    });

    it('draws a log-likelihood term square by default', () => {
        const spec = twoNodes();
        spec.nodes[1].kind = 'log-likelihood';
        expect(layoutGraph(spec).nodes[1].shape).toBe('square');
    });

    it('refuses an edge to a missing node', () => {
        expect(() => layoutGraph(twoNodes({ to: 'nowhere' }))).toThrow(/no node "nowhere"/);
    });

    it('refuses a binding without a role', () => {
        expect(() => layoutGraph(twoNodes({ kind: 'binding' }))).toThrow(/has no role/);
    });
});

describe('alongPolyline', () => {
    it('walks by length, not by vertex', () => {
        expect(
            alongPolyline(
                [
                    [0, 0],
                    [10, 0],
                    [10, 30],
                ],
                0.5,
            ),
        ).toEqual([10, 10]);
    });
});

describe('describeGraph', () => {
    it('names every node and edge with its kind', () => {
        const text = describeGraph(twoNodes({ kind: 'binding', role: 'fisher_root' }));
        expect(text).toContain('- `a`: Parameter');
        expect(text).toContain('- `a` → `b`: binding FISHER_ROOT');
    });
});
