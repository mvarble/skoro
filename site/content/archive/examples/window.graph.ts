import { defineGraph } from '$lib/graph/spec';

// The window's factor structure at a retirement, with a window of three poses: the oldest pose
// x_a retires with the landmark m_1 that no other pose of the window sees, and its blanket is the
// next pose x_{a+1} and the landmark m_2, which x_{a+2} still sees.

export const retirement = defineGraph({
    title: 'The window at a retirement',
    caption: String.raw`The factor structure of a window of three poses, $\randomPoint_a$, $\randomPoint_{a+1}$ and $\randomPoint_{a+2}$, with the odometry and the sightings observed. $\randomPoint_a$ retires together with $\bfm_1$, which no other pose of the window sees; the factors joined to them (the earlier replacement $\psi_a$, the motion and odometry terms $t^u_a$ and $t^y_a$, the sightings $t^z_{a1}$ and $t^z_{a2}$ and the prior $t^m_1$ of $\bfm_1$) are integrated, and the replacement they leave is joined to their blanket $\{\randomPoint_{a+1}, \bfm_2\}$. The factors of the later poses stay as they are.`,
    width: 640,
    height: 320,
    nodeKinds: {
        retired: { name: 'Retired quantity', hue: 'red' },
        blanket: { name: 'Blanket quantity', hue: 'orange' },
        kept: { name: 'Other free quantity', hue: 'blue' },
        factor: { name: 'Factor', hue: 'purple', shape: 'square' },
    },
    edgeKinds: {
        touches: { name: 'has a block for' },
    },
    nodes: [
        { id: 'rho', x: 40, y: 100, kind: 'factor', label: String.raw`$\psi_a$` },
        { id: 'xa', x: 150, y: 100, kind: 'retired', label: String.raw`$\randomPoint_a$` },
        { id: 'ua', x: 270, y: 40, kind: 'factor', label: String.raw`$t^u_a$` },
        { id: 'oa', x: 270, y: 100, kind: 'factor', label: String.raw`$t^y_a$` },
        { id: 'xb', x: 390, y: 100, kind: 'blanket', label: String.raw`$\randomPoint_{a+1}$` },
        { id: 'ub', x: 500, y: 40, kind: 'factor', label: String.raw`$t^u_{a+1}$` },
        { id: 'ob', x: 500, y: 100, kind: 'factor', label: String.raw`$t^y_{a+1}$` },
        { id: 'xc', x: 600, y: 100, kind: 'kept', label: String.raw`$\randomPoint_{a+2}$` },
        { id: 's1', x: 110, y: 200, kind: 'factor', label: String.raw`$t^z_{a1}$` },
        { id: 'm1', x: 110, y: 280, kind: 'retired', label: String.raw`$\bfm_1$` },
        { id: 'p1', x: 30, y: 280, kind: 'factor', label: String.raw`$t^m_1$` },
        { id: 's2', x: 270, y: 200, kind: 'factor', label: String.raw`$t^z_{a2}$` },
        { id: 'm2', x: 420, y: 280, kind: 'blanket', label: String.raw`$\bfm_2$` },
        { id: 'p2', x: 330, y: 280, kind: 'factor', label: String.raw`$t^m_2$` },
        { id: 's3', x: 560, y: 200, kind: 'factor', label: String.raw`$t^z_{a+2,2}$` },
    ],
    edges: [
        { from: 'rho', to: 'xa', kind: 'touches', arrow: 'none' },
        { from: 'ua', to: 'xa', kind: 'touches', arrow: 'none' },
        { from: 'ua', to: 'xb', kind: 'touches', arrow: 'none' },
        { from: 'oa', to: 'xa', kind: 'touches', arrow: 'none' },
        { from: 'oa', to: 'xb', kind: 'touches', arrow: 'none' },
        { from: 'ub', to: 'xb', kind: 'touches', arrow: 'none' },
        { from: 'ub', to: 'xc', kind: 'touches', arrow: 'none' },
        { from: 'ob', to: 'xb', kind: 'touches', arrow: 'none' },
        { from: 'ob', to: 'xc', kind: 'touches', arrow: 'none' },
        { from: 's1', to: 'xa', kind: 'touches', arrow: 'none' },
        { from: 's1', to: 'm1', kind: 'touches', arrow: 'none' },
        { from: 'p1', to: 'm1', kind: 'touches', arrow: 'none' },
        { from: 's2', to: 'xa', kind: 'touches', arrow: 'none' },
        { from: 's2', to: 'm2', kind: 'touches', arrow: 'none' },
        { from: 'p2', to: 'm2', kind: 'touches', arrow: 'none' },
        { from: 's3', to: 'xc', kind: 'touches', arrow: 'none' },
        { from: 's3', to: 'm2', kind: 'touches', arrow: 'none' },
    ],
});
