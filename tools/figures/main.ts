// The entry point `bin/figures` runs: `bin/figures <data directory> [<output directory>]`. It reads
// `pose_graph.json` and `sliding_window.json`, which the examples write when run with
// `SKORO_FIGURES=<data directory>`, and writes the figures, by default to `site/static/figures/`,
// with the site's stylesheet for them. Each example's figures are drawn from its own data, so
// running one example and then this redraws that example's figures and leaves the others as they
// are; a directory with neither example's data is an error.
import fs from 'node:fs';
import path from 'node:path';

import { calibrationFigure, figureStyles, poseGraphFigure, slidingWindowFigure } from './lib.ts';
import type { PoseGraph, SlidingWindow } from './lib.ts';

const [data, out = path.join(import.meta.dirname, '../../site/static/figures')] = process.argv.slice(2);
if (!data) {
    process.stderr.write('usage: bin/figures <data directory> [<output directory>]\n');
    process.exit(2);
}

/** An example's data, or nothing when that example was not run into `data`. */
const read = <T>(name: string): T | undefined => {
    const file = path.join(data, `${name}.json`);
    return fs.existsSync(file) ? (JSON.parse(fs.readFileSync(file, 'utf8')) as T) : undefined;
};
const write = (name: string, svg: string) => {
    fs.mkdirSync(out, { recursive: true });
    const file = path.join(out, `${name}.svg`);
    fs.writeFileSync(file, svg + '\n');
    process.stdout.write(`wrote ${file}\n`);
};
const skip = (example: string, figures: string) =>
    process.stdout.write(
        `skipped ${figures}: no ${example}.json in ${data} ` +
            `(run SKORO_FIGURES=${data} cargo run --release -p skoro --example ${example})\n`,
    );

const poseGraph = read<PoseGraph>('pose_graph');
if (poseGraph) {
    write('pose-graph', poseGraphFigure(poseGraph));
    write('calibration', calibrationFigure(poseGraph));
} else {
    skip('pose_graph', 'pose-graph.svg and calibration.svg');
}
const slidingWindow = read<SlidingWindow>('sliding_window');
if (slidingWindow) {
    write('sliding-window', slidingWindowFigure(slidingWindow));
} else {
    skip('sliding_window', 'sliding-window.svg');
}
if (!poseGraph && !slidingWindow) {
    process.stderr.write(`bin/figures: no example's data in ${data}\n`);
    process.exit(1);
}

// The site's stylesheet for the figures, from the same palette, unless the figures were written
// somewhere else.
if (process.argv[3] === undefined) {
    const styles = path.join(import.meta.dirname, '../../site/src/styles/utils/figures.css');
    fs.writeFileSync(styles, figureStyles());
    process.stdout.write(`wrote ${styles}\n`);
}
