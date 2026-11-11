import assert from 'node:assert/strict';
import { spawnSync } from 'node:child_process';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import test from 'node:test';

import {
    CHI2_90,
    PALETTE,
    calibrationFigure,
    chiSquaredDensity,
    ellipse90,
    figureStyles,
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

/** A small pose graph and window, as the examples' data would describe them. */
function sample(): { graph: PoseGraph; window: SlidingWindow } {
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
    return { graph, window };
}

test('each figure renders as one SVG with its title', () => {
    const { graph, window } = sample();
    for (const svg of [poseGraphFigure(graph), calibrationFigure(graph), slidingWindowFigure(window)]) {
        assert.match(svg, /^<svg [^>]*viewBox="0 0 \d+ \d+"/);
        assert.match(svg, /<title>[^<]+<\/title>/);
        assert.ok(svg.trimEnd().endsWith('</svg>'));
        assert.doesNotMatch(svg, /NaN|undefined/);
        // Every colour of a mark is a variable; hexadecimal colours appear only in the figure's
        // own stylesheet, which defines every variable a mark uses, in both themes.
        const style = /<style>([\s\S]*?)<\/style>/.exec(svg)?.[1] ?? '';
        const marks = svg.replace(/<style>[\s\S]*?<\/style>/, '');
        assert.doesNotMatch(marks, /#[0-9a-f]{3,6}\b/i);
        for (const used of new Set(marks.match(/--fig-[a-z]+/g))) {
            assert.equal(style.split(`${used}:`).length, 3, `${used} in both themes`);
        }
    }
});

test("the site's stylesheet sets every role in both themes", () => {
    const css = figureStyles();
    for (const theme of ['light', 'dark'] as const) {
        for (const [name, value] of Object.entries(PALETTE[theme])) {
            assert.ok(css.includes(`--fig-${name}: ${value};`), `${theme} ${name}`);
        }
    }
    assert.match(css, /^:root\.dark svg\.skoro-figure \{/m);
});

test('bin/figures draws the figures of whichever examples wrote their data', () => {
    const main = path.join(import.meta.dirname, 'main.ts');
    const run = (data: string, out: string) =>
        spawnSync(process.execPath, [main, data, out], { encoding: 'utf8' });
    const data = fs.mkdtempSync(path.join(os.tmpdir(), 'figures-data-'));
    const out = fs.mkdtempSync(path.join(os.tmpdir(), 'figures-out-'));
    try {
        // Neither example's data: an error, and nothing written.
        const none = run(data, out);
        assert.equal(none.status, 1, none.stderr);
        assert.deepEqual(fs.readdirSync(out), []);
        // The pose graph's alone: its two figures, and the window's skipped with how to make it.
        fs.writeFileSync(path.join(data, 'pose_graph.json'), JSON.stringify(sample().graph));
        const one = run(data, out);
        assert.equal(one.status, 0, one.stderr);
        assert.deepEqual(fs.readdirSync(out).sort(), ['calibration.svg', 'pose-graph.svg']);
        assert.match(one.stdout, /skipped sliding-window\.svg: no sliding_window\.json/);
        // Both: all three.
        fs.writeFileSync(path.join(data, 'sliding_window.json'), JSON.stringify(sample().window));
        const both = run(data, out);
        assert.equal(both.status, 0, both.stderr);
        assert.deepEqual(fs.readdirSync(out).sort(), [
            'calibration.svg',
            'pose-graph.svg',
            'sliding-window.svg',
        ]);
    } finally {
        fs.rmSync(data, { recursive: true, force: true });
        fs.rmSync(out, { recursive: true, force: true });
    }
});
