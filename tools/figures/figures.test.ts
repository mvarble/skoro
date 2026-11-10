import assert from 'node:assert/strict';
import test from 'node:test';

import {
    CHI2_90,
    calibrationFigure,
    chiSquaredDensity,
    ellipse90,
    histogram,
    largestEigenvalue,
    poseGraphFigure,
    slidingWindowFigure,
} from './lib.ts';
import type { Estimated, PoseGraph, SlidingWindow, Vec3 } from './lib.ts';

/** `∫₀^x f_d` by the midpoint rule. */
function cdf(d: number, x: number, n = 200_000): number {
    const h = x / n;
    let sum = 0;
    for (let i = 0; i < n; i++) sum += chiSquaredDensity(d, (i + 0.5) * h);
    return sum * h;
}

test('the chi-squared densities put 0.9 below their quantiles', () => {
    for (const d of [2, 3, 6]) {
        assert.ok(Math.abs(cdf(d, CHI2_90[d]!) - 0.9) < 1e-4, `d = ${d}`);
        assert.ok(Math.abs(cdf(d, 200) - 1) < 1e-4, `d = ${d} integrates to one`);
    }
});

test('the 90% ellipse follows the covariance', () => {
    const k = Math.sqrt(CHI2_90[2]!);
    const wide = ellipse90([4, 0, 0, 0, 1, 0, 0, 0, 1]);
    assert.ok(Math.abs(wide.rx - 2 * k) < 1e-12 && Math.abs(wide.ry - k) < 1e-12);
    assert.equal(wide.angle, 0);
    const tall = ellipse90([1, 0, 0, 0, 4, 0, 0, 0, 1]);
    assert.equal(tall.angle, 90);
    // Along the diagonal: variance 3 along (1, 1) and 1 across it.
    const diagonal = ellipse90([2, 1, 0, 1, 2, 0, 0, 0, 1]);
    assert.ok(Math.abs(diagonal.rx - Math.sqrt(3) * k) < 1e-12);
    assert.ok(Math.abs(diagonal.angle - 45) < 1e-9);
});

test('a histogram counts each value once, and leaves out what is past its range', () => {
    assert.deepEqual(histogram([0, 0.4, 0.6, 1.9, 2, 5], 2, 2), [3, 1]);
});

test('the largest eigenvalue of a symmetric matrix', () => {
    assert.ok(Math.abs(largestEigenvalue([2, 1, 0, 1, 2, 0, 0, 0, 1]) - 3) < 1e-9);
});

const at = (x: number, y: number): Vec3 => [x, y, 0];
const quantity = (x: number, y: number): Estimated => ({
    truth: at(x, y),
    start: at(x + 0.1, y),
    mode: at(x, y + 0.01),
    covariance: [0.01, 0, 0, 0, 0.004, 0, 0, 0, 0.01],
});

test('each figure renders as one SVG with its title', () => {
    const poses = Array.from({ length: 12 }, (_, k) => quantity(Math.cos(k / 2), Math.sin(k / 2)));
    const graph: PoseGraph = {
        rows: 2,
        poses,
        landmarks: [quantity(0, 0), quantity(2, 1)],
        sightings: [
            [0, 0],
            [5, 1],
        ],
        pose_errors: [1, 5, 12],
        landmark_errors: [0.5, 7],
        pose_calibration: { coverage: 0.9, tolerance: 0.05 },
        landmark_calibration: { coverage: 0.91, tolerance: 0.05 },
    };
    const window: SlidingWindow = {
        rows: 2,
        window: 3,
        truth: poses.map((p) => p.truth),
        start: poses.map((p) => p.start),
        batch: poses.map((p) => p.mode),
        filtered: poses.map((p) => ({ mode: p.mode, covariance: p.covariance })),
        landmark_truth: [at(0, 0)],
        landmark_batch: [at(0, 0)],
        filtering_errors: [1, 2],
        filtering_calibration: { coverage: 0.92, tolerance: 0.04 },
        gaps: [0.1, 0.2],
    };
    for (const svg of [poseGraphFigure(graph), calibrationFigure(graph), slidingWindowFigure(window)]) {
        assert.match(svg, /^<svg [^>]*viewBox="0 0 \d+ \d+"/);
        assert.match(svg, /<title>[^<]+<\/title>/);
        assert.ok(svg.trimEnd().endsWith('</svg>'));
        assert.doesNotMatch(svg, /NaN|undefined/);
    }
});
