import { defineGraph } from '$lib/graph/spec';

// Example 1: three poses on SE(3) linked by odometry, and one landmark seen from the first and last.
export const poseMap = defineGraph({
    title: 'Poses, odometry, and a landmark',
    caption:
        'Each pose $x_{k+1}$ is Gaussian about $x_k \\circ u_k$, and each observation $z$ about ' +
        'the landmark $\\ell$ seen from its pose. Every stochastic and observed node carries its ' +
        'own Gaussian term; the terms and the covariance factors are not drawn.',
    width: 740,
    height: 330,
    nodes: [
        { id: 'u0', x: 230, y: 290, kind: 'constant', label: '$u_0$', annotation: 'odometry', annotationAngle: 180 },
        { id: 'u1', x: 520, y: 290, kind: 'constant', label: '$u_1$' },
        { id: 'x0', x: 80, y: 190, kind: 'stochastic', label: '$x_0$', annotation: '$\\mathrm{SE}(3)$', annotationAngle: 90 },
        { id: 'm1', x: 230, y: 190, kind: 'mapped', label: '$m_1$', annotation: '$x_0 \\circ u_0$', annotationAngle: -60 },
        { id: 'x1', x: 370, y: 190, kind: 'stochastic', label: '$x_1$', annotation: '$\\mathrm{SE}(3)$', annotationAngle: 90 },
        { id: 'm2', x: 520, y: 190, kind: 'mapped', label: '$m_2$', annotation: '$x_1 \\circ u_1$', annotationAngle: -120 },
        { id: 'x2', x: 660, y: 190, kind: 'stochastic', label: '$x_2$', annotation: '$\\mathrm{SE}(3)$', annotationAngle: 90 },
        { id: 'l', x: 370, y: 45, kind: 'parameter', label: '$\\ell$', annotation: 'landmark, $\\R^3$', annotationAngle: -90 },
        { id: 'h0', x: 200, y: 90, kind: 'mapped', label: '$h_0$', annotation: '$x_0^{-1} \\cdot \\ell$', annotationAngle: 20 },
        { id: 'z0', x: 70, y: 55, kind: 'observed', label: '$z_0$' },
        { id: 'h2', x: 540, y: 90, kind: 'mapped', label: '$h_2$', annotation: '$x_2^{-1} \\cdot \\ell$', annotationAngle: 160 },
        { id: 'z2', x: 670, y: 55, kind: 'observed', label: '$z_2$' },
    ],
    edges: [
        { from: 'u0', to: 'm1' },
        { from: 'x0', to: 'm1' },
        { from: 'm1', to: 'x1' },
        { from: 'u1', to: 'm2' },
        { from: 'x1', to: 'm2' },
        { from: 'm2', to: 'x2' },
        { from: 'x0', to: 'h0' },
        { from: 'l', to: 'h0' },
        { from: 'h0', to: 'z0' },
        { from: 'x2', to: 'h2' },
        { from: 'l', to: 'h2' },
        { from: 'h2', to: 'z2' },
    ],
});
